# --- scarcity.py --------------------------------
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .summaries import _resolve_region, _regional_price, DEFAULT_YEAR
from .groups import NUCLEAR_PT, NUCLEAR_IDS, PT_RENAME, DCIIA_TYPES, DH_DCIIA_TYPES
from .energy_balance_build import relabel_nuclear, cluster_members

HOURS_PER_YEAR = 8760.0
EX_PT = "EX"                         # extraction: ECap is back-pressure-equivalent

# Electricity-generating types on the SEh frame. Heat-only / pure consumers
# never appear there. Nuclear is relabelled from NUCLEAR_IDS in the build.
EL_PTS = (NUCLEAR_PT, "CD", "BP", "BB", "EX", "EPE", "FD",
          "WS", "WL", "PV", "RO", "ES", "EV_V2G")
# Heat-producing types on the SHh frame. EH has no HCap (electricity-intake
# basis) -> its capacity factor comes back NaN by design, not faked.
DH_PTS = ("HS", "SH", "BH", "EPH", "BP", "BB", "EX", NUCLEAR_PT, "PX", "EH")


@dataclass(frozen=True)
class _MarketSpec:
    """Everything that differs between the electricity and district-heat tables."""
    area_key: str        # "EA" / "HA"
    price: str           # "pE" / "pH"
    cap_sym: str         # "ECap" / "HCap"  (installed capacity X, GW)
    weight_cols: tuple   # load weight for the demand-weighted regional price
    pts: tuple           # supply types present on the frame
    noun: str            # header noun for the printers
    ex_uplift: bool      # scale EX ids by (1 + Cv/Cm) -> condensing-equivalent


EL_SPEC = _MarketSpec("EA", "pE", "ECap", DCIIA_TYPES, EL_PTS, "el", True)
DH_SPEC = _MarketSpec("HA", "pH", "HCap", DH_DCIIA_TYPES, DH_PTS, "fjernvarme", False)


def _year_filter(df, years):
    """Subset to `years` (int, iterable, or None = all)."""
    if years is None:
        return df
    yrs = [years] if isinstance(years, int) else list(years)
    return df[df["t"].isin(yrs)]


def _ecap_years(df, years):
    """Year list (ints) matching `_year_filter`, for slicing the int64 `t` on cap."""
    if years is None:
        yrs = sorted(df["t"].unique())
    else:
        yrs = [years] if isinstance(years, int) else list(years)
    return [int(y) for y in yrs]


def _id2area_sym(db, area_key):
    sym = "id2" + area_key
    if sym not in db.symbols:
        cand = [k for k in db.symbols.keys() if "id2" in str(k)]
        raise KeyError(f"{sym} not in db; available id2*: {cand}")
    return sym


# =========================================================================== #
# cores  (market-agnostic; driven entirely by a _MarketSpec)
# =========================================================================== #
def _capture_core(sup, dem, spec, want):
    """gen / capture / vf_load / vf_flat for one state. Numerator: zonal price.
    Denominators: demand-weighted regional price (load) and its flat mean."""
    reg = _regional_price(dem, price=spec.price, weight_cols=spec.weight_cols)
    pcol = spec.price + "_region"
    p_load = (reg["w"] * reg[pcol]).sum() / reg["w"].sum()
    p_flat = reg[pcol].mean()                        # baseload convention

    cols = [c for c in want if c in sup.columns]
    q = sup[cols].clip(lower=0)                       # ES/HS: discharge only
    gen = q.sum()                                    # plain count, hFull-expanded
    rev = q.mul(sup[spec.price], axis=0).sum()

    df = pd.DataFrame({
        "Pt": cols,
        "gen": gen.reindex(cols).to_numpy(),
        "capture": (rev / gen.where(gen > 0)).reindex(cols).to_numpy(),
    })
    df["vf_load"] = df["capture"] / p_load
    df["vf_flat"] = df["capture"] / p_flat
    return df, (p_load, p_flat)


def _ex_multiplier(db, cap, yrs):
    """Per-row nameplate multiplier: (1 + Cv/Cm) on EX ids, 1.0 elsewhere.

    EX plants can run in extraction mode at ECap x (1 + Cv/Cm) electricity, so an
    uncorrected el capacity factor can exceed 100 pct. `cap` must still carry its
    original `Pt` (i.e. call before the nuclear relabel) and a clean RangeIndex."""
    mult = pd.Series(1.0, index=cap.index)
    is_ex = (cap["Pt"] == EX_PT).to_numpy()
    if not is_ex.any():
        return mult

    missing = [s for s in ("Cm", "Cv") if s not in db.symbols]
    if missing:                                      # fail loudly, never silently 1.0
        raise KeyError(
            f"EX ids present but {missing} not in db; cannot convert "
            "back-pressure-equivalent ECap to condensing-equivalent nameplate.")

    par = {}
    for sym in ("Cm", "Cv"):
        s = db.symbols[sym].vals.rename(sym).reset_index()
        s["t"] = s["t"].astype(int)
        s["id"] = s["id"].astype(str)
        par[sym] = s[s["t"].isin(yrs)][["t", "id", sym]]

    m = (cap[["t", "id"]]
         .merge(par["Cm"], on=["t", "id"], how="left")
         .merge(par["Cv"], on=["t", "id"], how="left"))
    bad = is_ex & (m["Cm"].isna() | m["Cv"].isna() | (m["Cm"] <= 0)).to_numpy()
    if bad.any():
        raise ValueError(f"EX ids without usable Cm/Cv: {cap.loc[bad, 'id'].tolist()}")

    mult[is_ex] = (1.0 + m.loc[is_ex, "Cv"] / m.loc[is_ex, "Cm"]).to_numpy()
    return mult


def _nameplate(db, spec, target_areas, yrs):
    """Σ (installed capacity X x 8760) per Pt, region-filtered via id2<area>.
    On the el side, EX ids are lifted to condensing-equivalent capacity."""
    id2Pt = db.symbols["id2Pt"].vals.to_frame(index=False)[["id", "Pt"]]
    amsym = _id2area_sym(db, spec.area_key)
    id2a = db.symbols[amsym].vals.to_frame(index=False)[["id", spec.area_key]]

    cap = db.symbols[spec.cap_sym].vals.rename(spec.cap_sym).reset_index()
    cap["t"] = cap["t"].astype(int)                  # int64 t, coerce
    cap = cap[cap["t"].isin(yrs)]
    cap = (cap.merge(id2a, on="id", how="left")
              .merge(id2Pt, on="id", how="left"))
    cap["id"] = cap["id"].astype(str)
    # reset_index is load-bearing: _ex_multiplier aligns a merge result positionally
    cap = cap[cap[spec.area_key].isin(target_areas)].reset_index(drop=True)
    if cap.empty:                                    # fail loudly
        raise ValueError(
            f"{spec.cap_sym} empty after {amsym} filter on {list(target_areas)}. "
            "Likely a namespace mismatch: supply uses a collapsed label (e.g. 'DK') "
            f"while {amsym} holds raw zones. Pass raw zones via ea=, or build ctx "
            "per-area.")

    # PV-battery cluster members are not in the supply frame (their net injection is
    # PVES), so they are not in the nameplate either.
    cap = cap[~cap["id"].isin(cluster_members(db))].reset_index(drop=True)
    cap["nameplate"] = cap[spec.cap_sym] * HOURS_PER_YEAR
    if spec.ex_uplift:                               # before the nuclear relabel:
        cap["nameplate"] *= _ex_multiplier(db, cap, yrs)   # KV ids are EX-typed

    cap["Pt"] = cap["Pt"].replace(PT_RENAME)
    cap = relabel_nuclear(cap, NUCLEAR_IDS)
    return cap.groupby("Pt")["nameplate"].sum()


def _capacity_factor_core(ctx, spec, target_areas, years, want, state):
    """realized / nameplate / cap_factor for one state (single-state slice)."""
    sup = _year_filter(ctx.supply_dict[state], years)
    sup = sup[sup[spec.area_key].isin(target_areas)]
    if sup.empty:
        raise ValueError(f"no supply rows for {list(target_areas)} / years={years} / {state}")
    cols = [c for c in want if c in sup.columns]
    realized = sup[cols].clip(lower=0).sum().reindex(cols)      # = capture's `gen`

    yrs = _ecap_years(ctx.supply_dict[state], years)
    name = _nameplate(ctx.db_dict[state], spec, target_areas, yrs)

    df = pd.DataFrame({
        "run": state, "Pt": cols,
        "realized": realized.to_numpy(),
        "nameplate": name.reindex(cols).to_numpy(),            # NaN if no cap (e.g. EH)
    })
    df["cap_factor"] = 100 * df["realized"] / df["nameplate"]
    return df


# =========================================================================== #
# shared reshape: long (run as a column) -> wide (baseline/shock side by side)
# =========================================================================== #
def _widen(res, vals, deltas, label, ndigits):
    """Pivot `run` into column suffixes, append deltas, index on (region[, t], Pt)."""
    key = ["Pt"] + [c for c in ("t",) if c in res.columns]
    wide = res.pivot_table(index=key, columns="run", values=vals, dropna=False)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([vals, ["baseline", "shock"]]))
    wide.columns = [f"{v}_{r[:5]}" for v, r in wide.columns]   # _basel / _shock
    for new, src in deltas.items():
        wide[new] = wide[f"{src}_shock"] - wide[f"{src}_basel"]
    return (wide.assign(region=label)
                .set_index("region", append=True)
                .reorder_levels(["region"] + key).sort_index()
                .round(ndigits))


def _slice_for_print(df, region, year):
    """Drop the region/year index levels so only Pt remains."""
    if region is not None:
        df = df.xs(region, level="region")
    if year is not None and "t" in df.index.names:
        df = df.xs(year, level="t")
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel([n for n in df.index.names if n != "Pt"])
    return df


def _print_table(df, cols, heads, *, title, sort_on, w, ndigits, total_cols):
    """The producer-profit table layout: right-aligned columns, rules, total row."""
    print(title)
    hdr = f'{"":<8}' + "".join(f"{h:>{w}}" for h in heads)
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for pt_, row in df[cols].sort_values(sort_on, ascending=False).iterrows():
        print(f"  {pt_:<6}" + "".join(
            f"{row[c]:>{w},.{ndigits}f}" if pd.notna(row[c]) else f'{"—":>{w}}'
            for c in cols))
    print("  " + "─" * (len(hdr) - 2))
    tot = df[total_cols].sum()
    print(f'  {"Total":<6}'
          + "".join(f"{tot[c]:>{w},.{ndigits}f}" for c in total_cols)
          + "".join(f'{"":>{w}}' for _ in cols[len(total_cols):]))
    print()


# =========================================================================== #
# summary engines
# =========================================================================== #
def _capture_summary(ctx, spec, ea, country, years, pts, verbose=False, wide=True):
    """Capture price / value factor by Pt. `wide=False` returns the long frame
    (run as a column) that the *_delta helpers consume. `verbose` unused (compat)."""
    target, label = _resolve_region(ea, country)
    want = list(pts) if pts is not None else list(spec.pts)
    out = []
    for state in ("baseline", "shock"):
        sup = _year_filter(ctx.supply_dict[state], years)
        sup = sup[sup[spec.area_key].isin(target)]
        if sup.empty:
            raise ValueError(f"no supply rows for {label} / years={years} / {state}")
        dem = _year_filter(ctx.demand_dict[state], years)
        dem = dem[dem[spec.area_key].isin(target)][
            ["t", spec.area_key, "hFull", spec.price, *spec.weight_cols]
        ]
        df, _ = _capture_core(sup, dem, spec, [c for c in want if c in sup.columns])
        df.insert(0, "run", state)
        out.append(df)
    res = pd.concat(out, ignore_index=True)

    # drop Pt rows with no capture price in either run (no generation at all)
    key = ["Pt"] + [c for c in ("t",) if c in res.columns]
    res = res[res.groupby(key)["capture"].transform(lambda s: s.notna().any())]

    if not wide:
        return (res.assign(region=label)
                   .set_index(["region", "run", "Pt"]).sort_index()
                   .round({"gen": 1, "capture": 2, "vf_load": 2, "vf_flat": 2}))
    return _widen(res, ["gen", "capture", "vf_load"],
                  {"dGen": "gen", "dCapture": "capture", "dVf": "vf_load"},
                  label, 2)


def _capacity_factor_summary(ctx, spec, ea, country, years, pts, verbose=False, wide=True):
    """Capacity factor by Pt. `wide=False` returns the long frame for *_delta."""
    if ctx.db_dict is None:
        raise ValueError(
            "capacity_factor needs ctx.db_dict (installed capacity). Build via "
            "ElBalanceContext.from_data(...); a from_dicts/unpickled ctx has no db.")
    target, label = _resolve_region(ea, country)
    want = list(pts) if pts is not None else list(spec.pts)
    res = pd.concat([_capacity_factor_core(ctx, spec, target, years, want, s)
                     for s in ("baseline", "shock")], ignore_index=True)

    # drop Pt rows with no cap_factor in either run (e.g. EH has no HCap)
    key = ["Pt"] + [c for c in ("t",) if c in res.columns]
    res = res[res.groupby(key)["cap_factor"].transform(lambda s: s.notna().any())]

    if not wide:
        return (res.assign(region=label)
                   .set_index(["region", "run", "Pt"]).sort_index()
                   .round({"realized": 1, "nameplate": 1, "cap_factor": 1}))
    return _widen(res, ["realized", "cap_factor"],
                  {"dRealized": "realized", "dCapFactor": "cap_factor"},
                  label, 1)


# =========================================================================== #
# printers  (identical layout for electricity and heat; spec picks the wording)
# =========================================================================== #
def print_capture(cap, *, region=None, year=None, spec=EL_SPEC):
    """Capture price by Pt in the producer-profit table style."""
    df = _slice_for_print(cap, region, year)
    cols = ["gen_basel", "gen_shock", "dGen",
            "capture_basel", "capture_shock", "dCapture", "dVf"]
    heads = ["TWh basis", "TWh stød", "ΔTWh", "fangst basis", "fangst stød",
             "Δfangst", "ΔVF"]
    yr_ = f"{year}, " if year is not None else ""
    _print_table(
        df, cols, heads,
        title=f"Capture price ({spec.noun}) — {yr_}{region or 'alle'}  (GWh / DKK per MWh)",
        sort_on="gen_basel", w=13, ndigits=2,
        total_cols=["gen_basel", "gen_shock", "dGen"],
    )


def print_capacity_factor(cf, *, region=None, year=None, spec=EL_SPEC, unit="GWh"):
    """Capacity factor by Pt in the producer-profit table style."""
    df = _slice_for_print(cf, region, year)
    cols = ["realized_basel", "realized_shock", "dRealized",
            "cap_factor_basel", "cap_factor_shock", "dCapFactor"]
    heads = ["basis", "stød", "Δ", "CF basis", "CF stød", "ΔCF"]
    yr_ = f"{year}, " if year is not None else ""
    _print_table(
        df, cols, heads,
        title=(f"Capacity factor ({spec.noun}, {spec.cap_sym}) — "
               f"{yr_}{region or 'alle'}  ({unit} / pct.)"),
        sort_on="realized_basel", w=12, ndigits=1,
        total_cols=["realized_basel", "realized_shock", "dRealized"],
    )


def print_heat_capture(cap, **kw):
    """District-heat capture table (same layout as :func:`print_capture`)."""
    print_capture(cap, spec=DH_SPEC, **kw)


def print_heat_capacity_factor(cf, **kw):
    """District-heat capacity-factor table (same layout as electricity)."""
    print_capacity_factor(cf, spec=DH_SPEC, **kw)


def _make_delta(summary, metrics, pct_on):
    """baseline vs. shock -> wide with per-metric deltas and one pct column."""
    w = summary.unstack("run")
    for col in metrics:
        w[(col, "delta")] = w[(col, "shock")] - w[(col, "baseline")]
    w[(pct_on, "pct")] = 100 * w[(pct_on, "delta")] / w[(pct_on, "baseline")]
    return w.sort_index(axis=1).round(2)


# =========================================================================== #
# public API  (thin wrappers -- signature identical across E and H)
# =========================================================================== #
def capture_summary(ctx, ea=None, *, country=None, years=DEFAULT_YEAR,
                    pts=None, verbose=True):
    """Electricity generation-weighted capture price and value factor per Pt."""
    return _capture_summary(ctx, EL_SPEC, ea, country, years, pts, verbose)


def heat_capture_summary(ctx, ea=None, *, country=None, years=DEFAULT_YEAR,
                         pts=None, verbose=True):
    """District-heat capture price and value factor per Pt (pH / DCIIA weight)."""
    return _capture_summary(ctx, DH_SPEC, ea, country, years, pts, verbose)


def capacity_factor_summary(ctx, ea=None, *, country=None, years=DEFAULT_YEAR,
                            pts=None, verbose=True):
    """Electricity capacity factor: realized / (ECap x 8760) per Pt, EX ids lifted
    to condensing-equivalent capacity via (1 + Cv/Cm)."""
    return _capacity_factor_summary(ctx, EL_SPEC, ea, country, years, pts, verbose)


def heat_capacity_factor_summary(ctx, ea=None, *, country=None, years=DEFAULT_YEAR,
                                 pts=None, verbose=True):
    """District-heat capacity factor: realized / (HCap x 8760) per Pt."""
    return _capacity_factor_summary(ctx, DH_SPEC, ea, country, years, pts, verbose)


def capture_delta(ctx, ea=None, *, country=None, years=DEFAULT_YEAR, pts=None):
    return _make_delta(_capture_summary(ctx, EL_SPEC, ea, country, years, pts,
                                        verbose=False, wide=False),
                       ("capture", "vf_load", "vf_flat"), "capture")


def heat_capture_delta(ctx, ea=None, *, country=None, years=DEFAULT_YEAR, pts=None):
    return _make_delta(_capture_summary(ctx, DH_SPEC, ea, country, years, pts,
                                        verbose=False, wide=False),
                       ("capture", "vf_load", "vf_flat"), "capture")


def capacity_factor_delta(ctx, ea=None, *, country=None, years=DEFAULT_YEAR, pts=None):
    return _make_delta(_capacity_factor_summary(ctx, EL_SPEC, ea, country, years, pts,
                                                verbose=False, wide=False),
                       ("realized", "nameplate", "cap_factor"), "cap_factor")


def heat_capacity_factor_delta(ctx, ea=None, *, country=None, years=DEFAULT_YEAR, pts=None):
    return _make_delta(_capacity_factor_summary(ctx, DH_SPEC, ea, country, years, pts,
                                                verbose=False, wide=False),
                       ("realized", "nameplate", "cap_factor"), "cap_factor")