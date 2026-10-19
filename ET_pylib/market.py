"""
Market outcomes — prices, scarcity, trade, terms of trade, capture and capacity
================================================================================

Tables of baseline vs. shock read off an `ElBalanceContext`. Regional prices are
demand-weighted on the DC-IIA consumers; capture prices and capacity factors
work for electricity (EL_SPEC) and district heat (DH_SPEC) alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .groups import (
    c_aggs, EA_TO_COUNTRY, DCIIA_TYPES, DH_DCIIA_TYPES, ANALYSIS_YEAR,
    NUCLEAR_PT, NUCLEAR_IDS, PT_RENAME, STATES,
)
from ._util import (
    filter_years, years_list, level_isin, relabel_nuclear, cluster_members, print_table,
)

HOURS_PER_YEAR = 8760.0


# ==================== ==================== ==================== ====================
# 0. regions and regional prices

def _resolve_region(ea, country):
    """(ea | country) -> (tuple of EAs, label). Fails loudly on an unknown country."""
    if country is not None:
        if country not in c_aggs:
            raise KeyError(f"unknown country {country!r}; known: {sorted(c_aggs)}")
        return tuple(c_aggs[country]), country
    if ea is None:
        raise ValueError("pass ea= or country=")
    return (ea,), ea


def _regional_price(df, *, price="pE", weight_cols=DCIIA_TYPES):
    """Areas collapsed to one demand-weighted price per (t, hFull).
    Returns t / hFull / <price>_region / w; weight = the DC-IIA consumers."""
    weight_cols = [c for c in weight_cols if c in df.columns]
    if not weight_cols:
        raise KeyError("no demand-weight columns in the frame")
    w = df[weight_cols].sum(axis=1).abs()
    pcol = f"{price}_region"
    g = (df.assign(_w=w, _wp=w * df[price])
           .groupby(["t", "hFull"], as_index=False)
           .agg(_wp=("_wp", "sum"), _w=("_w", "sum")))
    g[pcol] = g["_wp"] / g["_w"]
    return g.rename(columns={"_w": "w"})[["t", "hFull", pcol, "w"]]


def _region_demand(ctx, state, target_eas, years):
    """One state's demand frame, cut to `years` and `target_eas`, price + weight columns."""
    dem = ctx.demand_dict[state]
    df = dem[["t", "EA", "hFull", "pE", *[c for c in DCIIA_TYPES if c in dem.columns]]]
    df = filter_years(df, years, what="demand")
    return df[df["EA"].isin(target_eas)]


# ==================== ==================== ==================== ====================
# 1. prices and scarcity

def price_summary(ctx, ea=None, *, country=None, years=ANALYSIS_YEAR):
    """Demand-weighted moments (mean, std, skew, excess kurtosis) and min/max of the
    regional hourly price."""
    target_eas, label = _resolve_region(ea, country)
    rows = []
    for state in STATES:
        reg = _regional_price(_region_demand(ctx, state, target_eas, years))
        p, w = reg["pE_region"], reg["w"]
        W = w.sum()
        mean = (w * p).sum() / W
        var = (w * (p - mean) ** 2).sum() / W
        skew = (w * (p - mean) ** 3).sum() / W / var ** 1.5
        exkurt = (w * (p - mean) ** 4).sum() / W / var ** 2 - 3
        rows.append({"run": state, "mean": mean,
                     "min": p.min(), "max": p.max(), "std": var ** 0.5,
                     "skew": skew, "exkurt": exkurt})
    return (pd.DataFrame(rows).assign(region=label)
              .set_index(["region", "run"]).round(3))


def scarcity_summary(ctx, ea=None, *, country=None, years=ANALYSIS_YEAR,
                     null_threshold=20.0, mangel_factor=2.0):
    """Share of hours (pct.) with a near-zero price (< null_threshold) and with a
    scarcity price (> mangel_factor x annual mean), on the regional price."""
    target_eas, label = _resolve_region(ea, country)
    rows = []
    for state in STATES:
        per_hour = _regional_price(_region_demand(ctx, state, target_eas, years))
        wp = (per_hour["w"] * per_hour["pE_region"]).groupby(per_hour["t"]).sum()
        annual = wp / per_hour.groupby("t")["w"].sum()
        per_hour = per_hour.assign(annual_mean=per_hour["t"].map(annual))

        is_null = per_hour["pE_region"] < null_threshold
        is_mangel = per_hour["pE_region"] > mangel_factor * per_hour["annual_mean"]
        N = len(per_hour)                            # real hours in the window
        rows.append({"run": state,
                     "nulpristime": 100 * is_null.sum() / N,
                     "mangeltime": 100 * is_mangel.sum() / N})
    return (pd.DataFrame(rows).assign(region=label)
              .set_index(["region", "run"]).sort_index().round(4))


def full_summary(ctx, ea=None, *, country=None, years=ANALYSIS_YEAR):
    """Price moments and scarcity shares side by side."""
    p = price_summary(ctx, ea=ea, country=country, years=years)
    s = scarcity_summary(ctx, ea=ea, country=country, years=years)
    return p.join(s)


# ==================== ==================== ==================== ====================
# 2. trade

def trade_summary(ctx, ea=None, *, country=None, years=ANALYSIS_YEAR, value=False,
                  drop_empty=True):
    """Trade by partner for one EA or one country. Export is valued at the origin's
    price (EA), import at the partner's price (EAEA)."""
    if country is not None:
        target_eas, origin_label = _resolve_region(None, country)
        target_eas = set(target_eas)

    rows = []
    for state in STATES:
        df = filter_years(ctx.NX_dict[state], years, what="NX")
        if ea is not None:
            exp = df[df.index.get_level_values("EA") == ea]
            imp = df[df.index.get_level_values("EAEA") == ea]
            exp_q = exp["lineExport"].clip(lower=0)
            imp_q = imp["lineExport"].clip(lower=0)
            if value:
                exp_q = exp_q * exp["pE_EA"]
                imp_q = imp_q * imp["pE_EA"]
            g = pd.DataFrame({
                "export": exp_q.groupby(level="EAEA").sum(),
                "import": imp_q.groupby(level="EA").sum(),
            }).fillna(0.0)
            g.index.name = "partner"
        else:
            # origin == country iff EA is a member; partners outside it map to their country
            mask = (level_isin(df.index, "EA", target_eas)
                    & ~level_isin(df.index, "EAEA", target_eas))
            s = df.loc[mask, ["lineExport", "pE_EA", "pE_EAEA"]].reset_index()
            partner = s["EAEA"].map({x: EA_TO_COUNTRY.get(x, x) for x in s["EAEA"].unique()})
            flow = s["lineExport"]
            exp_q = flow.clip(lower=0)
            imp_q = (-flow).clip(lower=0)
            if value:
                exp_q = exp_q * s["pE_EA"]
                imp_q = imp_q * s["pE_EAEA"]
            g = pd.DataFrame({
                "export": exp_q.groupby(partner).sum(),
                "import": imp_q.groupby(partner).sum(),
            }).fillna(0.0)
            g.index.name = "partner"
        g["net"] = g["export"] - g["import"]
        g.insert(0, "run", state)
        rows.append(g.reset_index())
    out = (pd.concat(rows, ignore_index=True)
             .set_index(["partner", "run"]).sort_index().round(2))
    if drop_empty:
        active = (out[["export", "import"]].abs().groupby(level="partner").max() > 0).any(axis=1)
        out = out[out.index.get_level_values("partner").isin(active[active].index)]
    return out


def full_trade_summary(ctx, ea=None, *, country=None, years=ANALYSIS_YEAR, drop_empty=True):
    """Quantity (TWh) and value (mDKK) side by side per partner, plus a TOTAL row."""
    kw = dict(ea=ea, country=country, years=years, drop_empty=drop_empty)
    out = trade_summary(ctx, value=False, **kw).join(
        trade_summary(ctx, value=True, **kw), lsuffix="_TWh", rsuffix="_mDKK")
    totals = out.groupby(level="run").sum(numeric_only=True)
    totals.index = pd.MultiIndex.from_product([["TOTAL"], totals.index], names=out.index.names)
    return pd.concat([out, totals]).round(2)


# ==================== ==================== ==================== ====================
# 3. terms of trade

_TOT_COLS = ["X_TWh_base", "X_TWh_shock", "M_TWh_base", "M_TWh_shock",
             "pX_base", "pX_shock", "pM_base", "pM_shock",
             "ToT_base", "ToT_shock", "dToT", "dToT_pct"]


def _tot_record(df):
    """Unit values and terms of trade from one partner's frame, indexed by state."""
    if isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"expected a flat state index, got levels {df.index.names}")
    ratio = lambda a, b: a / b if b else float("nan")
    rec = {}
    for tag, state in (("base", "baseline"), ("shock", "shock")):
        r = df.loc[state]
        if isinstance(r, pd.DataFrame):
            raise ValueError(f"{len(r)} rows for state {state!r}; pass a single year")
        rec[f"X_TWh_{tag}"] = r["export_TWh"]
        rec[f"M_TWh_{tag}"] = r["import_TWh"]
        rec[f"pX_{tag}"] = ratio(r["export_mDKK"], r["export_TWh"])
        rec[f"pM_{tag}"] = ratio(r["import_mDKK"], r["import_TWh"])
        rec[f"ToT_{tag}"] = ratio(rec[f"pX_{tag}"], rec[f"pM_{tag}"])
    rec["dToT"] = rec["ToT_shock"] - rec["ToT_base"]
    rec["dToT_pct"] = 100 * rec["dToT"] / rec["ToT_base"] if rec["ToT_base"] else float("nan")
    return rec


def terms_of_trade_partner(ctx, *, country="DK", years=ANALYSIS_YEAR, include_total=True):
    """Bilateral terms of trade for `country` against each partner: pX is the
    country's own price in hours it exports there, pM the partner's price in hours
    it imports. TOTAL is the national terms of trade."""
    ts = full_trade_summary(ctx, country=country, years=years, drop_empty=True)
    partners = [p for p in ts.index.get_level_values("partner").unique() if p != "TOTAL"]
    if include_total:
        partners += ["TOTAL"]
    rows = {p: _tot_record(ts.xs(p, level="partner")) for p in partners}
    out = pd.DataFrame.from_dict(rows, orient="index")[_TOT_COLS]
    out.index.name = "partner"
    return out


def print_terms_of_trade_partner(ctx, *, country="DK", years=ANALYSIS_YEAR, include_total=True):
    """Bilateral terms-of-trade table, one row per partner."""
    yr = years if isinstance(years, int) else "+".join(map(str, years))
    print_table(
        terms_of_trade_partner(ctx, country=country, years=years, include_total=include_total),
        _TOT_COLS,
        ["X(b)", "X(s)", "M(b)", "M(s)", "pX(b)", "pX(s)", "pM(b)", "pM(s)",
         "ToT(b)", "ToT(s)", "ΔToT", "%Δ"],
        [",.2f", ",.2f", ",.2f", ",.2f", ",.0f", ",.0f", ",.0f", ",.0f",
         ".3f", ".3f", "+.3f", "+.2f"],
        title=(f"Terms of trade, {country} per partner — {yr}  "
               f"(X/M in TWh, pX/pM in DKK/MWh, ToT = pX/pM)"),
        w=8, rule_before="TOTAL", blank=False,
    )


# ==================== ==================== ==================== ====================
# 4. capture prices and capacity factors (electricity and district heat)

# Electricity-generating types on the SEh frame; nuclear is relabelled in the build.
EL_PTS = (NUCLEAR_PT, "CD", "BP", "BB", "EX", "EPE", "FD",
          "WS", "WL", "PV", "RO", "ES", "EV_V2G")
# Heat-producing types on the SHh frame. EH has no HCap, so its capacity factor is NaN.
DH_PTS = ("HS", "SH", "BH", "EPH", "BP", "BB", "EX", NUCLEAR_PT, "PX", "EH")


@dataclass(frozen=True)
class _MarketSpec:
    """Everything that differs between the electricity and district-heat tables."""
    area_key: str        # "EA" / "HA"
    price: str           # "pE" / "pH"
    cap_sym: str         # "ECap" / "HCap"
    weight_cols: tuple   # load weight for the regional price
    pts: tuple           # supply types on the frame
    noun: str            # header noun for the printers
    ex_uplift: bool      # scale EX ids by (1 + Cv/Cm) -> condensing-equivalent


EL_SPEC = _MarketSpec("EA", "pE", "ECap", DCIIA_TYPES, EL_PTS, "el", True)
DH_SPEC = _MarketSpec("HA", "pH", "HCap", DH_DCIIA_TYPES, DH_PTS, "fjernvarme", False)


def _capture_core(sup, dem, spec, want):
    """gen / capture / vf_load / vf_flat for one state. Numerator: zonal price;
    denominators: the regional load-weighted price and its flat mean."""
    reg = _regional_price(dem, price=spec.price, weight_cols=spec.weight_cols)
    pcol = spec.price + "_region"
    p_load = (reg["w"] * reg[pcol]).sum() / reg["w"].sum()
    p_flat = reg[pcol].mean()

    cols = [c for c in want if c in sup.columns]
    q = sup[cols].clip(lower=0)                       # storage: discharge only
    gen = q.sum()
    rev = q.mul(sup[spec.price], axis=0).sum()

    df = pd.DataFrame({
        "Pt": cols,
        "gen": gen.reindex(cols).to_numpy(),
        "capture": (rev / gen.where(gen > 0)).reindex(cols).to_numpy(),
    })
    df["vf_load"] = df["capture"] / p_load
    df["vf_flat"] = df["capture"] / p_flat
    return df


def _ex_multiplier(db, cap, yrs):
    """Per-row nameplate multiplier: (1 + Cv/Cm) on EX ids, 1.0 elsewhere. EX plants
    run at ECap x (1 + Cv/Cm) in extraction mode, so an uncorrected el capacity factor
    can exceed 100 pct. `cap` must carry its original Pt and a clean RangeIndex."""
    mult = pd.Series(1.0, index=cap.index)
    is_ex = (cap["Pt"] == "EX").to_numpy()
    if not is_ex.any():
        return mult

    missing = [s for s in ("Cm", "Cv") if s not in db.symbols]
    if missing:
        raise KeyError(f"EX ids present but {missing} not in db; cannot convert "
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
    """Sum of (installed capacity x 8760) per Pt in `target_areas`; on the el side EX
    ids are lifted to condensing-equivalent capacity."""
    id2Pt = db.symbols["id2Pt"].vals.to_frame(index=False)[["id", "Pt"]]
    amsym = "id2" + spec.area_key
    if amsym not in db.symbols:
        raise KeyError(f"{amsym} not in db")
    id2a = db.symbols[amsym].vals.to_frame(index=False)[["id", spec.area_key]]

    cap = db.symbols[spec.cap_sym].vals.rename(spec.cap_sym).reset_index()
    cap["t"] = cap["t"].astype(int)
    cap = cap[cap["t"].isin(yrs)]
    cap = (cap.merge(id2a, on="id", how="left")
              .merge(id2Pt, on="id", how="left"))
    cap["id"] = cap["id"].astype(str)
    # reset_index is load-bearing: _ex_multiplier aligns a merge result positionally
    cap = cap[cap[spec.area_key].isin(target_areas)].reset_index(drop=True)
    if cap.empty:
        raise ValueError(
            f"{spec.cap_sym} empty after the {amsym} filter on {list(target_areas)}: "
            "pass raw zones, not a collapsed label such as 'DK' from the country build.")

    # PV-battery cluster members are not in the supply frame (their net injection is PVES).
    cap = cap[~cap["id"].isin(cluster_members(db))].reset_index(drop=True)
    cap["nameplate"] = cap[spec.cap_sym] * HOURS_PER_YEAR
    if spec.ex_uplift:                               # before the nuclear relabel
        cap["nameplate"] *= _ex_multiplier(db, cap, yrs)

    cap["Pt"] = cap["Pt"].replace(PT_RENAME)
    cap = relabel_nuclear(cap, NUCLEAR_IDS)
    return cap.groupby("Pt")["nameplate"].sum()


def _supply_in(ctx, spec, state, target, years, label):
    sup = filter_years(ctx.supply_dict[state], years, what="supply")
    sup = sup[sup[spec.area_key].isin(target)]
    if sup.empty:
        raise ValueError(f"no supply rows for {label} / years={years} / {state}")
    return sup


def _widen(res, vals, deltas, label, ndigits, *, keep_if):
    """`run` pivoted into column suffixes (_basel / _shock), deltas appended. Pt rows
    with `keep_if` missing in both runs (no generation / no capacity) are dropped."""
    res = res[res.groupby("Pt")[keep_if].transform(lambda s: s.notna().any())]
    wide = res.pivot_table(index=["Pt"], columns="run", values=vals, dropna=False)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([vals, ["baseline", "shock"]]))
    wide.columns = [f"{v}_{r[:5]}" for v, r in wide.columns]
    for new, src in deltas.items():
        wide[new] = wide[f"{src}_shock"] - wide[f"{src}_basel"]
    return (wide.assign(region=label)
                .set_index("region", append=True)
                .reorder_levels(["region", "Pt"]).sort_index()
                .round(ndigits))


def _capture_summary(ctx, spec, ea, country, years, pts):
    target, label = _resolve_region(ea, country)
    want = list(pts) if pts is not None else list(spec.pts)
    out = []
    for state in STATES:
        sup = _supply_in(ctx, spec, state, target, years, label)
        dem = filter_years(ctx.demand_dict[state], years, what="demand")
        dem = dem[dem[spec.area_key].isin(target)][
            ["t", spec.area_key, "hFull", spec.price, *spec.weight_cols]]
        df = _capture_core(sup, dem, spec, [c for c in want if c in sup.columns])
        df.insert(0, "run", state)
        out.append(df)
    res = pd.concat(out, ignore_index=True)
    return _widen(res, ["gen", "capture", "vf_load"],
                  {"dGen": "gen", "dCapture": "capture", "dVf": "vf_load"}, label, 2,
                  keep_if="capture")


def _capacity_factor_summary(ctx, spec, ea, country, years, pts):
    target, label = _resolve_region(ea, country)
    want = list(pts) if pts is not None else list(spec.pts)
    out = []
    for state in STATES:
        sup = _supply_in(ctx, spec, state, target, years, label)
        cols = [c for c in want if c in sup.columns]
        realized = sup[cols].clip(lower=0).sum().reindex(cols)
        yrs = years_list(years) or sorted(int(t) for t in ctx.supply_dict[state]["t"].unique())
        name = _nameplate(ctx.db_dict[state], spec, target, yrs)
        df = pd.DataFrame({"run": state, "Pt": cols,
                           "realized": realized.to_numpy(),
                           "nameplate": name.reindex(cols).to_numpy()})
        df["cap_factor"] = 100 * df["realized"] / df["nameplate"]
        out.append(df)
    res = pd.concat(out, ignore_index=True)
    return _widen(res, ["realized", "cap_factor"],
                  {"dRealized": "realized", "dCapFactor": "cap_factor"}, label, 1,
                  keep_if="cap_factor")


def capture_summary(ctx, ea=None, *, country=None, years=ANALYSIS_YEAR, pts=None):
    """Electricity generation-weighted capture price and value factor per Pt."""
    return _capture_summary(ctx, EL_SPEC, ea, country, years, pts)


def heat_capture_summary(ctx, ea=None, *, country=None, years=ANALYSIS_YEAR, pts=None):
    """District-heat capture price and value factor per Pt."""
    return _capture_summary(ctx, DH_SPEC, ea, country, years, pts)


def capacity_factor_summary(ctx, ea=None, *, country=None, years=ANALYSIS_YEAR, pts=None):
    """Electricity capacity factor: realised / (ECap x 8760) per Pt, EX ids at
    condensing-equivalent capacity (1 + Cv/Cm)."""
    return _capacity_factor_summary(ctx, EL_SPEC, ea, country, years, pts)


def heat_capacity_factor_summary(ctx, ea=None, *, country=None, years=ANALYSIS_YEAR, pts=None):
    """District-heat capacity factor: realised / (HCap x 8760) per Pt."""
    return _capacity_factor_summary(ctx, DH_SPEC, ea, country, years, pts)


def _slice_for_print(df, region):
    if region is not None:
        df = df.xs(region, level="region")
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel([n for n in df.index.names if n != "Pt"])
    return df


def print_capture(cap, *, region=None, year=None, spec=EL_SPEC):
    """Capture-price table by Pt."""
    cols = ["gen_basel", "gen_shock", "dGen",
            "capture_basel", "capture_shock", "dCapture", "dVf"]
    df = _slice_for_print(cap, region)[cols].sort_values("gen_basel", ascending=False)
    yr_ = f"{year}, " if year is not None else ""
    print_table(
        df, cols,
        ["TWh basis", "TWh stød", "ΔTWh", "fangst basis", "fangst stød", "Δfangst", "ΔVF"],
        ",.2f", w=13, nan="—", total=["gen_basel", "gen_shock", "dGen"],
        title=f"Capture price ({spec.noun}) — {yr_}{region or 'alle'}  (GWh / DKK per MWh)",
    )


def print_capacity_factor(cf, *, region=None, year=None, spec=EL_SPEC, unit="GWh"):
    """Capacity-factor table by Pt."""
    cols = ["realized_basel", "realized_shock", "dRealized",
            "cap_factor_basel", "cap_factor_shock", "dCapFactor"]
    df = _slice_for_print(cf, region)[cols].sort_values("realized_basel", ascending=False)
    yr_ = f"{year}, " if year is not None else ""
    print_table(
        df, cols, ["basis", "stød", "Δ", "CF basis", "CF stød", "ΔCF"],
        ",.1f", w=12, nan="—", total=["realized_basel", "realized_shock", "dRealized"],
        title=(f"Capacity factor ({spec.noun}, {spec.cap_sym}) — "
               f"{yr_}{region or 'alle'}  ({unit} / pct.)"),
    )


def print_heat_capture(cap, **kw):
    print_capture(cap, spec=DH_SPEC, **kw)


def print_heat_capacity_factor(cf, **kw):
    print_capacity_factor(cf, spec=DH_SPEC, **kw)
