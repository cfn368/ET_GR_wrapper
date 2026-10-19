"""Annuitised CAPEX + O&M and system costs for capacity shocks, allocated to
countries. Constants are shock-specific; functions are generic over the tables."""

from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
import math
from .groups import NUCLEAR_IDS, c_aggs

# =========================================================================== #
# 1. assumptions
# =========================================================================== #
R, T = 0.04, 40                      # real discount rate, technical lifetime (yr)
EUR_DKK = 7.46                       # fixed-rate peg, mio. kr. per MEUR
KV_ECAP_FACTOR = 1.25                # KV enters ECap as back-pressure-equivalent

CABLE_UNITS = {                      # CAPEX mio.kr/MW, annual O&M as share of CAPEX
    "Øresund_1-3":   (1.846,  0.00833),
    "Kontiskan":     (11.429, 0.01400),
    "Skagerrak_1-4": (11.429, 0.01250),
}

CABLE_OWNERS = {                     # physical line -> countries sharing the cost
    "Øresund_1-3":   ("DK", "SE"),
    "Kontiskan":     ("DK", "SE"),
    "Skagerrak_1-4": ("DK", "NO"),
}

PLANT_UNITS = {                      # CAPEX MEUR/MW, lifetime yr, annual FOM as % of CAPEX
    "Nuclear":      (8,     60, 164/100),   # ET / FOA / OECD-IEA
    "WindOffshore": (25/10, 30, 192/100),   # ET / FOA / ET
}

PLANT_OWNERS = {                     # technology -> countries carrying the cost
    "Nuclear":      ("DK",),
    "WindOffshore": ("DK",),
}

IDC_YEARS = {                        # byggerenter: effektive år med renteeksponering
    "Nuclear":      2.0,             # ~4 års byggeperiode, jævn udbetaling
    "WindOffshore": 0.0,             # ingen tillæg indtil videre
}


# --- system costs (grid, balancing, reserves) -- NOT in UP/ΔPS -------------- #
ORE_KWH_PR_TWH = 10.0                # 1 øre/kWh x 1 TWh = 10 mio. kr.
VRE_PTS = ("PV", "WL", "WS")         # VE = sol + land- og havvind
VRE_PTS_AGG = ("Sol", "Vind")        # samme sæt, hvis output er PT_AGGS-aggregeret
NUC_PTS = ("Nuclear",)

VRE_SYSCOST_ORE_KWH = 14.5            # systemomkostning pr. kWh VE
NUC_SYSCOST_ORE_KWH =  3.             # 10 DKK/MWh = 1 øre/kWh


# --- rentestrukturer ------------------------------------------------------- #
FM_STEPS = ((35, 0.035), (70, 0.025), (None, 0.015))   # Finansministeriet 2021

RATE_SCHEDULES = {
    "flat4":  0.04,
    "flat35": 0.035,
    "flat8":  0.08,
    "fm":     FM_STEPS,
}


# =========================================================================== #
# 2. cost tables
# =========================================================================== #
def annuity_factor(r=R, T=T):
    """Capital recovery factor for rate `r` over `T` years."""
    return r / (1 - (1 + r) ** -T)


def _steps_to_vector(steps, n, offset):
    """Trappefunktion ((graense_aar, rente), ...) -> aarsrenter for aar 1..n."""
    out = []
    for k in range(offset, offset + n):              # k = 0-indekseret kalenderaar
        for edge, rate in steps:
            if edge is None or k < edge:
                out.append(float(rate))
                break
        else:
            raise ValueError(f"rentetrappe daekker ikke aar {k}: {steps}")
    return out


def rate_vector(r, n, *, offset=0):
    """Aarsrenter for aar 1..n. Accepterer skalar, navn, trappe, callable, vektor."""
    n = int(n)
    if isinstance(r, str):
        r = RATE_SCHEDULES[r]                        # KeyError = ukendt navn, loud
    if isinstance(r, (int, float)):
        return [float(r)] * n
    if callable(r):
        return [float(r(k)) for k in range(offset, offset + n)]
    seq = list(r)
    if seq and isinstance(seq[0], (tuple, list)):
        return _steps_to_vector(seq, n, offset)
    if len(seq) < offset + n:
        raise ValueError(f"rentevektor har {len(seq)} aar, kraever {offset + n}")
    return [float(x) for x in seq[offset:offset + n]]


def discount_factors(rates):
    """Kumulerede diskonteringsfaktorer d_s for en liste af aarsrenter."""
    out, d = [], 1.0
    for r_k in rates:
        d /= (1.0 + r_k)
        out.append(d)
    return out


def annuity_factor(r=R, T=T, *, offset=0):
    """Capital recovery factor. Lukket form ved konstant rente, ellers 1/sum(d_s)."""
    rates = rate_vector(r, T, offset=offset)
    if len(set(rates)) == 1:                         # bagudkompatibel fast path
        r0 = rates[0]
        return r0 / (1 - (1 + r0) ** -int(T))
    return 1.0 / sum(discount_factors(rates))


def idc_factor(years, r=R, *, offset=0):
    """Byggerentefaktor prod(1+r_u)^dt: overnight cost -> CAPEX. Broek-aar tillades."""
    years = float(years)
    if years <= 0:
        return 1.0
    rates = rate_vector(r, math.ceil(years), offset=offset)
    phi, left = 1.0, years
    for r_k in rates:
        step = min(1.0, left)
        phi *= (1.0 + r_k) ** step
        left -= step
    return phi


def cable_cost(deltas, *, r=R, T=T, units=None):
    """Annual cost per physical line (mio. kr./yr) from a GW-delta Series."""
    units = CABLE_UNITS if units is None else units
    a = annuity_factor(r, T)
    rows = {}
    for line, gw in deltas.items():
        capex_pr_mw, om_rate = units[line]           # KeyError = unmapped line, loud
        capex = capex_pr_mw * float(gw) * 1000.0     # GW -> MW
        rows[line] = {"delta_GW": float(gw), "capex": capex,
                      "annuitet": capex * a, "om": capex * om_rate}
    df = pd.DataFrame(rows).T
    df["total"] = df["annuitet"] + df["om"]
    return df


def plant_cost(deltas, *, r=R, units=None, eur_dkk=EUR_DKK, idc=None):
    """Annual cost per technology (mio. kr./yr) from a signed GW-delta Series.
    `capex` er overnight; `capex_idc` er inkl. byggerenter og er grundlaget for
    annuiteten. FOM regnes af overnight -- det er en driftspost, ikke finansiering."""
    units = PLANT_UNITS if units is None else units
    idc = IDC_YEARS if idc is None else idc
    rows = {}
    for tech, gw in deltas.items():
        capex_meur_pr_mw, lifetime, fom_pct = units[tech]   # KeyError = unmapped tech, loud
        idc_years = idc[tech]                               # KeyError = unmapped tech, loud
        phi = idc_factor(idc_years, r)
        capex = capex_meur_pr_mw * eur_dkk * float(gw) * 1000.0   # MEUR/MW -> mio.kr/MW, GW -> MW
        rows[tech] = {"delta_GW": float(gw), "levetid": float(lifetime),
                      "idc_aar": float(idc_years), "capex": capex,
                      "capex_idc": capex * phi,
                      "annuitet": capex * phi * annuity_factor(r, lifetime),
                      "om": capex * fom_pct / 100.0}
    df = pd.DataFrame(rows).T
    df["total"] = df["annuitet"] + df["om"]
    return df


# =========================================================================== #
# 3. allocation to countries
# =========================================================================== #
def annuity_by_country(cost, *, owners=None, col="total"):
    """{country: mio. kr./yr}, splitting each line evenly across its owners."""
    owners = CABLE_OWNERS if owners is None else owners
    out = {}
    for line, row in cost.iterrows():
        parties = owners[line]
        for c in parties:
            out[c] = out.get(c, 0.0) + float(row[col]) / len(parties)
    return out


def annuity_by_tech(deltas, *, r=R, units=None, owners=None, scale=1e-3,
                    col="annuitet"):
    """Annuity per (country, technology). Index country, columns tech + 'total'.
    `col`: "annuitet" (CAPEX only -- FOM is already netted out of `UP`/ΔPS)
    or "total" (CAPEX + FOM, for a standalone resource-cost table)."""
    units = PLANT_UNITS if units is None else units
    owners = PLANT_OWNERS if owners is None else owners
    cost = plant_cost(deltas, r=r, units=units)

    out = {}
    for tech, row in cost.iterrows():
        parties = owners[tech]                       # KeyError = unmapped tech, loud
        share = float(row[col]) / len(parties) * scale
        for c in parties:
            bucket = out.setdefault(c, {})
            bucket[tech] = bucket.get(tech, 0.0) + share

    df = pd.DataFrame(out).T.fillna(0.0)
    df = df.reindex(columns=[t for t in units if t in df.columns])   # stable order
    df["total"] = df.sum(axis=1)
    return df


def welfare_annuity(deltas, *, r=R, units=None, owners=None, scale=1e-3,
                    col="annuitet"):
    """{country: total annuity} for `welfare_levels`. Default scale: mio. -> mia."""
    return annuity_by_tech(deltas, r=r, units=units, owners=owners,
                           scale=scale, col=col)["total"].to_dict()


def annuity_by_line(cost, *, owners=None, col="total", scale=1.0):
    """Annuitet pr. (land, fysisk linje). Index land, kolonner linje + 'total'.
    `col="total"` (CAPEX + O&M) er default: kabel-O&M ligger hos TSO'en og er
    -- i modsætning til anlæg-FOM -- ikke netto i `UP`/ΔPS."""
    owners = CABLE_OWNERS if owners is None else owners
    out = {}
    for line, row in cost.iterrows():
        parties = owners[line]                       # KeyError = unmapped line, loud
        share = float(row[col]) / len(parties) * scale
        for c in parties:
            bucket = out.setdefault(c, {})
            bucket[line] = bucket.get(line, 0.0) + share

    df = pd.DataFrame(out).T.fillna(0.0)
    df = df.reindex(columns=[l for l in cost.index if l in df.columns])
    df["total"] = df.sum(axis=1)
    return df


# =========================================================================== #
# 4. shock-specific delta builders
# =========================================================================== #
def nuclear_deltas(cd, kv, *, basis="nameplate"):
    """Signed GW deltas {Nuclear, WindOffshore} for the wind->nuclear substitution.
    `basis`: "nameplate" bills entered CD+KV, "installed" bills dispatchable ECap."""
    if basis not in ("nameplate", "installed"):
        raise ValueError(f"basis must be 'nameplate' or 'installed', got {basis!r}")
    nuc = cd + kv if basis == "nameplate" else cd + kv / KV_ECAP_FACTOR
    return pd.Series({"Nuclear": nuc, "WindOffshore": -(cd + kv)})   # AltCapReduction cuts entered CD+KV


def swap_deltas(nuc, cut, *, include_heat=True):
    """Signed GW deltas for `plant_cost` from `nuc_wind_swap` output.
    Nuclear capacity splits across ECap (el) and HCap (heat); the entered
    nameplate is their sum, so bill both."""
    gw = nuc["GW_added"] + (nuc["GW_heat"] if include_heat else 0.0)
    return pd.Series({"Nuclear": float(gw.sum()),
                      "WindOffshore": -float(cut["GW_cut"].sum())})


def cable_deltas(directed, *, col="delta", line_level="line", tol=1e-9):
    """GW-delta pr. fysisk linje fra den retningsopdelte trCap-tabel.
    Rejser hvis de to retninger på en linje ikke har samme delta."""
    d = directed[col] if hasattr(directed, "columns") else directed
    g = d.groupby(level=line_level)
    spread = (g.max() - g.min()).abs()
    bad = spread[spread > tol]
    if len(bad):
        raise ValueError(f"asymmetrisk delta pr. linje: {bad.to_dict()}")
    return g.first().astype(float)                   # én værdi pr. fysisk linje


# =========================================================================== #
# 5. generation lookups
# =========================================================================== #
DQ_COLS = ("dQ_TWh", "delta", "dQ")                        # delta-kolonnenavne
BASE_COLS, SHOCK_COLS = ("q_base", "baseline"), ("q_shock", "shock")


def _generation(baseline_output, shock_output, *, year, areas, id2Pt, id2EA,
                split_nuclear=False):
    """producer_generation, med valgfri nuclear-udskilning hvis den understøttes."""
    from .effect_figs import producer_generation
    kw = dict(id2Pt=id2Pt, id2EA=id2EA, areas=areas, year=year)
    if split_nuclear:
        try:
            return producer_generation(baseline_output, shock_output,
                                       split_nuclear=True, **kw)
        except TypeError:                                  # ældre signatur
            pass
    return producer_generation(baseline_output, shock_output, **kw)


def _gen_keys(gen):
    """Plant-type-nøgler som strenge, uanset om Pt er indeksniveau eller indeks."""
    lvl = "Pt" if "Pt" in (gen.index.names or []) else None
    return [str(k) for k in (gen.index.get_level_values(lvl) if lvl else gen.index)]


def _gen_series(gen, which):
    """Kolonne fra en generationsframe: 'base', 'shock' eller 'delta' (TWh)."""
    lookup = {str(c).lower(): c for c in gen.columns}
    pick = lambda names: next((lookup[n.lower()] for n in names if n.lower() in lookup), None)
    base, shock = pick(BASE_COLS), pick(SHOCK_COLS)
    if which == "delta":
        dq = pick(DQ_COLS)
        if dq is not None:
            return gen[dq]
        if base is None or shock is None:
            raise KeyError(f"kan ikke udlede delta af kolonner {list(gen.columns)}")
        return gen[shock] - gen[base]
    col = base if which == "base" else shock
    if col is None:
        raise KeyError(f"ingen '{which}'-kolonne i {list(gen.columns)}")
    return gen[col]


def _resolve_pts(keys, pts, pts_agg, what):
    """Vælg det typesæt der faktisk er i frame'en; rejs hvis intet eller begge."""
    raw = [p for p in pts if p in keys]
    agg = [p for p in pts_agg if p in keys]
    if raw and agg:
        raise KeyError(f"både rå og aggregerede {what}-typer til stede: {raw} + {agg}")
    if not (raw or agg):
        raise KeyError(f"ingen {what}-typer i generationstabellen; "
                       f"fandt {sorted(set(keys))}")
    return raw or agg


# =========================================================================== #
# 6. system costs
# =========================================================================== #
@dataclass(frozen=True)
class SysCost:
    """Én systemomkostningspost: sats i øre/kWh gange en mængde i TWh."""
    key: str                      # rækkenøgle i blokken
    label: str                    # dansk rækkelabel
    rate: float                   # øre/kWh
    pts: tuple                    # rå Pt-koder
    basis: str = "delta"          # "delta" (shock - baseline) eller "level" (shock)
    pts_agg: tuple = ()           # PT_AGGS-buckets, hvis output er aggregeret
    split_nuclear: bool = False   # kræver Nuclear udskilt af "Regulerbar"
    unit_label: str = "øre/kWh"


SYSCOST_SPECS = (
    SysCost("vre_syscost", "Systemomkostning, VE", VRE_SYSCOST_ORE_KWH,
            pts=VRE_PTS, pts_agg=VRE_PTS_AGG, basis="delta"),
    SysCost("nuc_syscost", "Systemomkostning, kernekraft", NUC_SYSCOST_ORE_KWH,
            pts=NUC_PTS, basis="delta", split_nuclear=True),
)


def syscost_twh(baseline_output, shock_output, spec, *, year, areas,
                id2Pt=None, id2EA=None, base_tol=1e-6):
    """Mængden bag `spec` (TWh). "delta" = shock - baseline, "level" = shock
    (og kræver at baseline er nul, ellers rejses)."""
    gen = _generation(baseline_output, shock_output, year=year, areas=areas,
                      id2Pt=id2Pt, id2EA=id2EA, split_nuclear=spec.split_nuclear)
    keys = _gen_keys(gen)
    what = spec.key.split("_")[0].upper()
    sel = [k in _resolve_pts(keys, spec.pts, spec.pts_agg, what) for k in keys]

    if spec.basis == "delta":
        return float(_gen_series(gen, "delta")[sel].sum())
    if spec.basis == "level":
        q_base = float(_gen_series(gen, "base")[sel].sum())
        if abs(q_base) > base_tol:
            raise ValueError(f"{spec.key}: baseline er {q_base:.6f} TWh, ikke nul "
                             "-- brug basis='delta'")
        return float(_gen_series(gen, "shock")[sel].sum())
    raise ValueError(f"basis must be 'delta' or 'level', got {spec.basis!r}")


def syscost_mio(twh, rate):
    """Velfærdsbidrag (mio. kr./yr): negativ = omkostning, positiv = sparet."""
    return -float(twh) * float(rate) * ORE_KWH_PR_TWH


# bagudkompatible tyndtvejere -- kaldes af ældre notebook-celler
def vre_delta_twh(baseline_output, shock_output, **kw):
    """Δ VE-produktion (TWh). Se :func:`syscost_twh`."""
    return syscost_twh(baseline_output, shock_output, SYSCOST_SPECS[0], **kw)


def nuc_gen_twh(baseline_output, shock_output, **kw):
    """Kernekraftproduktion i stødet (TWh). Se :func:`syscost_twh`."""
    return syscost_twh(baseline_output, shock_output, SYSCOST_SPECS[1], **kw)


# =========================================================================== #
# 7. welfare block -- rå model-output -> printbar tabel, ét kald
# =========================================================================== #
WELFARE_ROWS = (                     # modellens egne komponenter; sum == net
    ("dCS",     "Δ Forbrugeroverskud"),
    ("dPS",     "Δ Producentoverskud"),
    ("dCI",     "Δ Flaskehalsindtægter"),
    ("annuity", "Annuitet, netto"),
)
NET_ROW = ("net", "Netto velfærdseffekt")
TECH_LABELS = {"Nuclear": "Kernekraft (ny)", "WindOffshore": "Havvind (nedlagt)"}

REGION_COUNTRIES = {                 # region -> lande i annuitetstabellen
    "DK": ("DK",), "NO": ("NO",), "SE": ("SE",), "Scandi": ("DK", "NO", "SE"),
}
UNIT_SCALE = {"mio": 1000.0, "mia": 1.0}   # welfare_levels returnerer mia. kr.
UNIT_LABEL = {"mio": "mio. kr./år", "mia": "mia. kr./år"}


def welfare_block(
    baseline_output, shock_output, *, year, deltas=None, annuity=None,
    regions=("DK",), markets=("E"), include_ci=True, include_heat=True,
    id2Pt=None, id2EA=None,
    r=R, units=None, owners=None, col="annuitet", unit="mio",
    syscosts=SYSCOST_SPECS, rates=None, twh=None, nuc_u_min=0.3,
):
    """Velfærdsdekomponering + annuitet + systemomkostninger, komponenter x regioner.

    `annuity` overstyrer anlægsberegningen: enten {land: mio. kr./yr} eller en
    frame som :func:`annuity_by_tech` / :func:`annuity_by_line` (index land,
    kolonner poster + 'total'). `syscosts=()` slår systemposterne fra.
    """
    from .effect_figs import welfare_levels
    from .groups import c_aggs

    if unit not in UNIT_SCALE:
        raise ValueError(f"unit must be one of {tuple(UNIT_SCALE)}, got {unit!r}")
    if annuity is not None and r != R:
        raise ValueError("r= ignoreres naar annuity= gives; byg annuitetsframen "
                         "med den oenskede rente (annuity_by_tech(..., r=...))")

    scale = UNIT_SCALE[unit]
    regions = list(regions)
    id2Pt = shock_output["id2Pt"].vals if id2Pt is None else id2Pt
    id2EA = shock_output["id2EA"].vals if id2EA is None else id2EA
    rates, twh = dict(rates or {}), dict(twh or {})

    flex = None
    if nuc_u_min is not None:
        from .flex_fix import nuc_flex_value
        flex = nuc_flex_value(shock_output, year=year, u_min=nuc_u_min, syscost=0).loc["total"]

    if annuity is None:                              # anlægsstød: byg fra GW-deltas
        if deltas is None:
            from ET_shocks.nucShock import nuc_wind_swap
            nuc, cut = nuc_wind_swap(baseline_output, shock_output, yr=year, verbose=False)
            deltas = swap_deltas(nuc, cut, include_heat=include_heat)
        ann_mio = annuity_by_tech(deltas, r=r, units=units, owners=owners,
                                  scale=1.0, col=col)
    elif hasattr(annuity, "columns"):                # færdig frame (anlæg el. kabel)
        ann_mio = annuity.copy()
        if "total" not in ann_mio.columns:
            ann_mio["total"] = ann_mio.sum(axis=1)
    else:                                            # {land: mio. kr./yr}
        ann_mio = pd.DataFrame({"total": pd.Series(annuity, dtype=float)})

    ann = ann_mio * (scale / 1000.0)
    items = [c for c in ann.columns if c != "total"]  # tom => ingen "heraf"-blok

    w = welfare_levels(
        baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
        year=year, markets=markets, regions=regions, include_ci=include_ci,
        annuity=ann_mio["total"].to_dict(),           # mio. kr./yr
    ) * scale

    used_rate = {s.key: float(rates.get(s.key, s.rate)) for s in syscosts}
    used_twh = {s.key: {} for s in syscosts}

    out = {}
    for region in regions:
        s = {k: float(w.loc[region, k]) for k, _ in WELFARE_ROWS}

        net_model = float(w.loc[region, NET_ROW[0]])
        net_chk = sum(s[k] for k, _ in WELFARE_ROWS)
        assert abs(net_chk - net_model) < 1e-6, (
            f"{region}: net {net_model:.4f} != Σ komponenter {net_chk:.4f}")

        lands = REGION_COUNTRIES.get(region, (region,))
        has_nuc = flex is not None and "DK" in lands  # kernekraften står i DK
        if has_nuc:
            s["dPS"] += float(flex["dW"]) * (scale / 1000.0)

        for item in items:
            s["ann_" + item] = sum(float(ann.loc[c, item]) for c in lands if c in ann.index)
        if items:
            ann_sum = sum(s["ann_" + i] for i in items)
            assert abs(ann_sum + s["annuity"]) < 1e-6, (
                f"{region}: annuity {s['annuity']:.4f} != -Σ ann_* {-ann_sum:.4f}")

        for spec in syscosts:
            q = twh.get(spec.key, {}).get(region)
            if q is None:
                q = syscost_twh(baseline_output, shock_output, spec, year=year,
                                areas=c_aggs[region], id2Pt=id2Pt, id2EA=id2EA)
            if has_nuc and spec.key == "nuc_syscost":
                q += float(flex["dTWh"])
            used_twh[spec.key][region] = float(q)
            s[spec.key] = syscost_mio(q, used_rate[spec.key]) * (scale / 1000.0)

        s[NET_ROW[0]] = sum(s[k] for k, _ in WELFARE_ROWS) + sum(s[sp.key] for sp in syscosts)
        out[region] = s

    index = ([k for k, _ in WELFARE_ROWS] + [sp.key for sp in syscosts]
             + [NET_ROW[0]] + ["ann_" + i for i in items])
    block = pd.DataFrame(out).reindex(index=index)
    block.index.name, block.columns.name = "component", "region"
    block.attrs.update(year=year, unit=unit, markets=tuple(markets), techs=tuple(items),
                       col=col, syscosts=tuple(syscosts), rates=used_rate, twh=used_twh,
                       nuc_u_min=nuc_u_min)
    return block


def welfare_block_by_rate(baseline_output, shock_output, schedules, **kw):
    """{navn: welfare_block} for hver rentestruktur. `schedules`: {navn: r}."""
    return {name: welfare_block(baseline_output, shock_output, r=sched, **kw)
            for name, sched in schedules.items()}


# =========================================================================== #
# 8. printer
# =========================================================================== #
def _dk(v, dec=2, signed=True):
    """Dansk talformat med eksplicit fortegn (. tusind, , decimal)."""
    txt = f"{abs(v):,.{dec}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return (("-" if v < 0 else "+") if signed else ("-" if v < 0 else "")) + txt


def print_welfare_block(block, *, width=52):
    """Printer output fra :func:`welfare_block` -- én blok pr. region."""
    year = block.attrs.get("year")
    unit = UNIT_LABEL.get(block.attrs.get("unit", "mio"), "")
    techs = block.attrs.get("techs", ())
    specs = [sp for sp in block.attrs.get("syscosts", ()) if sp.key in block.index]
    rates, twh = block.attrs.get("rates", {}), block.attrs.get("twh", {})
    rule = "─" * width

    def _row(label, v, indent=2):
        return " " * indent + label.ljust(width - 14 - indent) + _dk(v).rjust(12)

    rows = list(WELFARE_ROWS) + [(sp.key, sp.label) for sp in specs]
    pad = max([len(sp.label.split(",")[-1].strip()) for sp in specs], default=0)

    for region in block.columns:
        s = block[region]
        out = [f"Velfærdseffekt, {region}" + (f" — {year}" if year else ""),
               f"({unit})", rule]
        out += [_row(lbl, float(s[k])) for k, lbl in rows]
        out += [rule, _row(NET_ROW[1], float(s[NET_ROW[0]])), rule]

        if techs:
            out.append("  heraf annuitet (+ = omkostning, − = sparet):")
            out += [_row(TECH_LABELS.get(t, t), float(s["ann_" + t]), indent=4)
                    for t in techs]
            out.append(_row("netto omkostning",
                            sum(float(s["ann_" + t]) for t in techs), indent=4))
            out.append(rule)

        if specs:
            out.append("  systemomkostning, grundlag:")
            for sp in specs:
                name = sp.label.split(",")[-1].strip()
                q = twh.get(sp.key, {}).get(region)
                basis = "niveau" if sp.basis == "level" else "ændring"
                out.append(f"    {name.ljust(pad)}  {_dk(q, 3)} TWh ({basis}) "
                           f"x {_dk(rates.get(sp.key, sp.rate), 0, signed=False)} {sp.unit_label}")
            out.append(rule)
        print("\n".join(out))


def rate_sweep_table(blocks, *, region="DK"):
    """Komponenter x rentescenarier for én region."""
    df = pd.DataFrame({name: b[region] for name, b in blocks.items()})
    df.index.name, df.columns.name = "component", "rente"
    return df.round(2)


# =========================================================================== #
# 8b. multi-scenarie printer (fx rentestrukturer side om side)
# =========================================================================== #
def _blocks_attr(blocks, key, default=None):
    """Fælles attrs-værdi på tværs af blokke; rejser hvis de er uenige."""
    vals = [b.attrs.get(key, default) for b in blocks.values()]
    if any(v != vals[0] for v in vals[1:]):
        raise ValueError(f"blokke er uenige om attrs[{key!r}]: {vals}")
    return vals[0]


def print_welfare_blocks(blocks, *, regions=None, gap=3, dec=2):
    """Printer én eller flere `welfare_block`e side om side, én blok pr. region.
    `blocks`: {kolonnenavn: block} (fx rentescenarier) eller en enkelt block."""
    if hasattr(blocks, "columns"):                   # enkelt block -> én kolonne
        blocks = {"": blocks}
    blocks = dict(blocks)
    if not blocks:
        raise ValueError("ingen blokke at printe")
    names = list(blocks)

    year = _blocks_attr(blocks, "year")
    unit = UNIT_LABEL.get(_blocks_attr(blocks, "unit", "mio"), "")
    techs = _blocks_attr(blocks, "techs", ())
    specs = [sp for sp in _blocks_attr(blocks, "syscosts", ())
             if all(sp.key in b.index for b in blocks.values())]
    rates = {n: blocks[n].attrs.get("rates", {}) for n in names}
    twh = {n: blocks[n].attrs.get("twh", {}) for n in names}

    if regions is None:
        regions = [r for r in blocks[names[0]].columns
                   if all(r in b.columns for b in blocks.values())]
    elif isinstance(regions, str):
        regions = [regions]

    rows = list(WELFARE_ROWS) + [(sp.key, sp.label) for sp in specs]
    vals = lambda key, reg: [float(blocks[n].loc[key, reg]) for n in names]

    for region in regions:
        # ("row", indent, label, values) | ("rule",) | ("text", linje)
        entries = [("row", 2, lbl, vals(key, region)) for key, lbl in rows]
        entries += [("rule",), ("row", 2, NET_ROW[1], vals(NET_ROW[0], region)), ("rule",)]

        if techs:
            entries.append(("text", "  heraf annuitet (+ = omkostning, − = sparet):"))
            entries += [("row", 4, TECH_LABELS.get(t, t), vals("ann_" + t, region))
                        for t in techs]
            entries.append(("row", 4, "netto omkostning",
                            [sum(float(blocks[n].loc["ann_" + t, region]) for t in techs)
                             for n in names]))
            entries.append(("rule",))

        if specs:                                    # mængder er rente-uafhængige
            entries.append(("text", "  systemomkostning, grundlag:"))
            pad = max(len(sp.label.split(",")[-1].strip()) for sp in specs)
            for sp in specs:
                nm = sp.label.split(",")[-1].strip().ljust(pad)
                basis = "niveau" if sp.basis == "level" else "ændring"
                qs = [twh[n].get(sp.key, {}).get(region) for n in names]
                rs = [float(rates[n].get(sp.key, sp.rate)) for n in names]
                q_txt = lambda q: "n/a" if q is None else _dk(q, 3) + " TWh"
                same = (all(q is not None for q in qs)
                        and max(qs) - min(qs) < 1e-9 and len(set(rs)) == 1)
                if same:
                    entries.append(("text", f"    {nm}  {q_txt(qs[0])} ({basis}) "
                                            f"x {_dk(rs[0], 0, signed=False)} {sp.unit_label}"))
                else:
                    for n, q, rr in zip(names, qs, rs):
                        entries.append(("text", f"    {nm}  {n}: {q_txt(q)} ({basis}) "
                                                f"x {_dk(rr, 0, signed=False)} {sp.unit_label}"))
            entries.append(("rule",))

        rws = [e for e in entries if e[0] == "row"]
        label_w = max(e[1] + len(e[2]) for e in rws) + gap
        col_w = max(max(len(_dk(v, dec)) for e in rws for v in e[3]),
                    max(len(n) for n in names)) + gap
        rule = "─" * (label_w + col_w * len(names))

        out = [f"Velfærdseffekt, {region}" + (f" — {year}" if year else ""),
               f"({unit})", rule]
        if any(names):                               # tom navn => gammelt enkelt-layout
            out += [" " * label_w + "".join(n.rjust(col_w) for n in names), rule]

        for e in entries:
            if e[0] == "rule":
                out.append(rule)
            elif e[0] == "text":
                out.append(e[1])
            else:
                _, ind, lbl, vs = e
                out.append(" " * ind + lbl.ljust(label_w - ind)
                           + "".join(_dk(v, dec).rjust(col_w) for v in vs))
        print("\n".join(out))


# =========================================================================== #
# 9. LCOE -- GR's egne drifts- og produktionstal + annuitet fra plant_cost
# =========================================================================== #
LCOE_TECHS = {                       # label -> nøgle i LCOE_UNITS (VE-labels = GR's Pt-koder)
    "KK": "Nuclear", "WS": "WindOffshore", "WL": "WindOnshore", "PV": "PV",
}
LCOE_VRE = ("WS", "WL", "PV")        # indgår i den produktionsvægtede VE-række
LCOE_VRE_LABEL = "VE (vægtet)"

LCOE_UNITS = {                       # (CAPEX MEUR/MW, levetid år, FOM %); FOM-% ignoreres -- FOM tages fra GR
    **PLANT_UNITS,
    "WindOnshore": (9.7 / EUR_DKK, 30, 0.0),   # ET
    "PV":          (4.4 / EUR_DKK, 40, 0.0),   # ET
}
LCOE_IDC = {**IDC_YEARS, "WindOnshore": 0.0, "PV": 0.0}


def _lcoe_yr(db, sym, year):
    """Symbolværdier for ét år med t droppet. Rejser hvis året ikke findes."""
    v = db.symbols[sym].vals
    out = v[v.index.get_level_values("t").astype(str) == str(year)].droplevel("t")
    if out.index.nlevels == 1:
        out.index = out.index.astype(str)
    if out.empty:
        raise KeyError(f"{sym}: ingen værdier for t={year}")
    return out


def _lcoe_map(db, sym):
    """id -> kategori fra en 2-niveau mapping (id2Pt, id2EA)."""
    m = db.symbols[sym].vals
    return pd.Series(m.get_level_values(-1), index=m.get_level_values(0))


def lcoe_inputs(db, *, year, areas=c_aggs["DK"], nuc_ids=None):
    """GR-tal pr. id for KK/WS/WL/PV i `areas`: kapacitet, FOM, VOM, MC, produktion.
    ECap_el = ECap*(1+Cv/Cm) for EX (indtastet nameplate); MC_EX dækker EX's manglende MC.
    KK-id'erne er NUCLEAR_IDS fra deres startår. `db` skal bære stødets inputs
    (ShockedDB), ellers er kernekraftens ECap/FOM/VOM basisforløbets."""
    nuc_ids = NUCLEAR_IDS if nuc_ids is None else nuc_ids
    nuc = {i for i, start in nuc_ids.items() if int(year) >= start}
    pt_of, ea_of = _lcoe_map(db, "id2Pt"), _lcoe_map(db, "id2EA")
    pt_of.index, ea_of.index = pt_of.index.astype(str), ea_of.index.astype(str)

    tech_of = pt_of.where(pt_of.isin(LCOE_VRE)).astype(object)
    tech_of[tech_of.index.isin(nuc)] = "KK"
    tech_of = tech_of.dropna()

    ea = tech_of.index.map(ea_of)
    if pd.isna(ea).any():
        raise KeyError(f"id'er uden EA: {tech_of.index[pd.isna(ea)].tolist()}")
    tech_of = tech_of[ea.isin(areas)]
    if "KK" not in tech_of.values:
        raise KeyError(f"ingen kernekraft-id'er ({sorted(nuc)}) i {areas} i {year} -- er db kernekraftstødet?")
    ids = tech_of.index

    mc = _lcoe_yr(db, "MC", year)
    if "MC_EX" in db.symbols:                                  # EX har ingen MC
        mc = mc.combine_first(_lcoe_yr(db, "MC_EX", year))

    ecap = _lcoe_yr(db, "ECap", year)
    el_fac = (1 + _lcoe_yr(db, "Cv", year).div(_lcoe_yr(db, "Cm", year))).reindex(ids).fillna(1.0)

    seh = _lcoe_yr(db, "SEh", year)                            # årstotaler over cluster-timer
    gen = seh[seh.index.get_level_values("id").isin(ids)].groupby(level="id").sum()

    df = pd.DataFrame({
        "tech":        tech_of,
        "EA":          ea_of.reindex(ids),
        "ECap_GW":     ecap.reindex(ids),
        "ECap_el_GW":  ecap.reindex(ids) * el_fac,
        "FOM_mDKK_GW": _lcoe_yr(db, "FOM", year).reindex(ids),
        "VOM_DKK_MWh": _lcoe_yr(db, "VOM", year).reindex(ids),
        "MC_DKK_MWh":  mc.reindex(ids),
        "E_GWh":       gen.reindex(ids),
    })
    df = df[df["ECap_GW"] > 0]

    bad = df.isna().any(axis=1)
    if bad.any():
        raise ValueError(f"manglende værdier for: {df.index[bad].tolist()}")
    return df


def lcoe_table(db, *, year, areas=c_aggs["DK"], r=R, units=None, idc=None,
               techs=LCOE_TECHS, vre=LCOE_VRE, vre_label=LCOE_VRE_LABEL):
    """LCOE pr. teknologi + produktionsvægtet VE-række (øre/kWh).
    Annuitet (plant_cost) på ECap_el; FOM på ECap (modelbasis); variabelt led = MC."""
    units = LCOE_UNITS if units is None else units
    idc = LCOE_IDC if idc is None else idc
    df = lcoe_inputs(db, year=year, areas=areas)

    # annuitet pr. GW nameplate (mio. kr./GW/år) -- 1 GW-delta pr. teknologi
    unit_cost = plant_cost(pd.Series(1.0, index=list(techs.values())), r=r, units=units, idc=idc)
    ann_per_gw = (unit_cost["annuitet"].astype(float)
                  .rename(index={v: k for k, v in techs.items()}))

    # årlige omkostninger (mio. kr./år) og produktion (GWh) pr. teknologi
    s = df.groupby("tech")[["ECap_GW", "ECap_el_GW", "E_GWh"]].sum()
    s["ann_mio"] = ann_per_gw.reindex(s.index) * s["ECap_el_GW"]                 # CAPEX på nameplate
    s["fom_mio"] = (df["FOM_mDKK_GW"] * df["ECap_GW"]).groupby(df["tech"]).sum() # FOM på modelbasis
    s["mc_x_e"]  = (df["MC_DKK_MWh"] * df["E_GWh"]).groupby(df["tech"]).sum()

    missing = [t for t in techs if t not in s.index]
    if missing:
        raise KeyError(f"ingen kapacitet for {missing} i {areas}, {year}")
    s = s.loc[list(techs)]
    s.loc[vre_label] = s.loc[list(vre)].sum()                  # sum -> produktionsvægtet ved division

    # mio. kr./GWh = 1000 kr./MWh; /10 -> øre/kWh
    out = pd.DataFrame({
        "CF":       s["E_GWh"] / (s["ECap_el_GW"] * 8760),
        "annuitet": s["ann_mio"] / s["E_GWh"] * 100,
        "FOM":      s["fom_mio"] / s["E_GWh"] * 100,
        "MC":       s["mc_x_e"] / s["E_GWh"] / 10,
    })
    out["LCOE"] = out[["annuitet", "FOM", "MC"]].sum(axis=1)
    out.index.name = "tech"
    out.attrs.update(year=year, areas=tuple(areas), r=r, unit="øre/kWh")
    return out