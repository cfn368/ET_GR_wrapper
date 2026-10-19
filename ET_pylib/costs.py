"""
Costs — capacity swap, annuitised investment, system costs and LCOE
====================================================================

Assumptions and cost functions for the nuclear-for-offshore-wind swap. Reads the
swap off the solved databases (`nuc_wind_swap`), annuitises CAPEX under flat or
stepped discount rates with interest during construction, and builds LCOE from
GR's own operating and production figures. Welfare aggregation lives in `welfare`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .groups import NUC_ID_BY_EA, NUCLEAR_IDS, c_aggs
from .style import dk_num, col_names
from ._util import sym_year, id_map


# ==================== ==================== ==================== ====================
# 1. assumptions

R, T    = 0.04, 40                   # real discount rate, technical lifetime (yr)
EUR_DKK = 7.46                       # fixed-rate peg, mio. kr. per MEUR

PLANT_UNITS = {                      # CAPEX MEUR/MW, lifetime yr, annual FOM as % of CAPEX
    "Nuclear":      (8,     60, 164/100),   # ET / FOA / OECD-IEA
    "WindOffshore": (25/10, 30, 192/100),   # ET / FOA / ET
}
PLANT_OWNERS = {                     # technology -> countries carrying the cost
    "Nuclear":      ("DK",),
    "WindOffshore": ("DK",),
}
IDC_YEARS = {                        # effective years of interest during construction
    "Nuclear":      2.0,             # ~4-year build, even disbursement
    "WindOffshore": 0.0,
}

# System costs (grid, balancing, reserves) -- not in UP / ΔPS.
ORE_KWH_PR_TWH      = 10.0           # 1 øre/kWh x 1 TWh = 10 mio. kr.
VRE_PTS             = ("PV", "WL", "WS")
NUC_PTS             = ("Nuclear",)
VRE_SYSCOST_ORE_KWH = 14.5
NUC_SYSCOST_ORE_KWH = 3.0

FM_STEPS = ((35, 0.035), (70, 0.025), (None, 0.015))   # Finansministeriet 2021
RATE_SCHEDULES = {
    "flat4":  0.04,
    "flat35": 0.035,
    "flat8":  0.08,
    "fm":     FM_STEPS,
}


# ==================== ==================== ==================== ====================
# 2. capacity swap read off the solved databases

def nuc_wind_swap(baseline, shock, yr, dk=("DK1", "DK2"), wind=("WS",), verbose=True,
                  nuc_ids=None):
    """Nuclear capacity added vs. offshore wind displaced in `yr`.
    Returns (nuc, cut): per-id frames of GW."""
    nuc_ids = list(NUC_ID_BY_EA.values()) if nuc_ids is None else list(nuc_ids)
    xs = lambda db, sym: sym_year(db, sym, yr).astype("float64")
    b, s, hcap = xs(baseline, "ECap"), xs(shock, "ECap"), xs(shock, "HCap")
    pt, ea = id_map(shock("id2Pt")), id_map(shock("id2EA"))

    nuc = pd.DataFrame({"EA": ea.reindex(nuc_ids).values, "Pt": pt.reindex(nuc_ids).values,
                        "GW_base": b.reindex(nuc_ids, fill_value=0.0).values,
                        "GW_added": s.reindex(nuc_ids, fill_value=0.0).values,
                        "GW_heat": hcap.reindex(nuc_ids, fill_value=0.0).values}, index=nuc_ids)
    nuc = nuc[nuc["EA"].isin(dk)]

    w = [i for i in pt.index[pt.isin(wind)] if i not in nuc_ids]
    base_w, shock_w = b.reindex(w, fill_value=0.0), s.reindex(w, fill_value=0.0)
    cut = pd.DataFrame({"EA": ea.reindex(w).values, "Pt": pt.reindex(w).values,
                        "GW_base": base_w.values, "GW_cut": (base_w - shock_w).values},
                       index=w)
    cut = cut[cut["EA"].isin(dk)]

    if verbose:
        print_swap(nuc, cut, yr, wind)
    return nuc, cut


def print_swap(nuc, cut, yr, wind=("WS",)):
    """Capacity-swap table (GW): nuclear added, wind removed, ratio."""
    f = lambda v, signed=False: dk_num(v, 2, signed)
    nuc_label = col_names.get("Nuclear", "Kernekraft")
    rows = [(f"  {nuc_label} {r.EA}", r.GW_base, r.GW_added)
            for _, r in nuc.sort_values("EA").iterrows()]
    agg = cut.groupby(["EA", "Pt"])[["GW_base", "GW_cut"]].sum()
    rows += [(f"  {col_names.get(p, p)} {a}", r.GW_base, r.GW_base - r.GW_cut)
             for (a, p), r in agg.iterrows()]

    lw = max([len(r[0]) for r in rows] + [28]) + 2
    rule = "─" * (lw + 30)
    print(f"Kapacitetsbytte, {yr} (GW)\n{rule}")
    print(f"{'':<{lw}}{'basis':>10}{'stød':>10}{'Δ':>10}\n{rule}")

    if nuc.GW_added.abs().sum() == 0:
        print(f"  Ingen kernekraft i {yr}.")
    for label, base, shock in rows:
        print(f"{label:<{lw}}{f(base):>10}{f(shock):>10}{f(shock - base, True):>10}")
    print(rule)

    d_nuc = float((nuc.GW_added - nuc.GW_base).sum())
    d_wind = float(-cut.GW_cut.sum())
    wind_label = col_names.get(wind[0], "Vind") if len(wind) == 1 else "Vind"
    print(f"{f'  {nuc_label} i alt':<{lw + 20}}{f(d_nuc, True):>10}")
    print(f"{f'  {wind_label} i alt':<{lw + 20}}{f(d_wind, True):>10}")
    if d_nuc > 0:
        print(f"{f'  {wind_label} fjernet pr. GW {nuc_label.lower()}':<{lw + 20}}"
              f"{f(-d_wind / d_nuc):>10}")
    if nuc.GW_heat.abs().sum() > 0:
        print(f"{f'  {nuc_label}, varme (GW-th)':<{lw + 20}}{f(nuc.GW_heat.sum(), True):>10}")
    print(rule)


def swap_deltas(nuc, cut, *, include_heat=True):
    """Signed GW deltas {Nuclear, WindOffshore} for `plant_cost`. Nuclear capacity
    splits across ECap (el) and HCap (heat); the entered nameplate is their sum."""
    gw = nuc["GW_added"] + (nuc["GW_heat"] if include_heat else 0.0)
    return pd.Series({"Nuclear": float(gw.sum()),
                      "WindOffshore": -float(cut["GW_cut"].sum())})


# ==================== ==================== ==================== ====================
# 3. discounting

def _steps_to_vector(steps, n, offset):
    """Step schedule ((edge_year, rate), ...) -> annual rates for years 1..n."""
    out = []
    for k in range(offset, offset + n):              # k = 0-indexed calendar year
        for edge, rate in steps:
            if edge is None or k < edge:
                out.append(float(rate))
                break
        else:
            raise ValueError(f"rate schedule does not cover year {k}: {steps}")
    return out


def rate_vector(r, n, *, offset=0):
    """Annual rates for years 1..n from a scalar, schedule name, step schedule,
    callable or vector."""
    n = int(n)
    if isinstance(r, str):
        r = RATE_SCHEDULES[r]
    if isinstance(r, (int, float)):
        return [float(r)] * n
    if callable(r):
        return [float(r(k)) for k in range(offset, offset + n)]
    seq = list(r)
    if seq and isinstance(seq[0], (tuple, list)):
        return _steps_to_vector(seq, n, offset)
    if len(seq) < offset + n:
        raise ValueError(f"rate vector has {len(seq)} years, needs {offset + n}")
    return [float(x) for x in seq[offset:offset + n]]


def discount_factors(rates):
    """Cumulative discount factors d_s for a list of annual rates."""
    out, d = [], 1.0
    for r_k in rates:
        d /= (1.0 + r_k)
        out.append(d)
    return out


def annuity_factor(r=R, T=T, *, offset=0):
    """Capital recovery factor: closed form at a constant rate, else 1 / sum(d_s)."""
    rates = rate_vector(r, T, offset=offset)
    if len(set(rates)) == 1:
        r0 = rates[0]
        return r0 / (1 - (1 + r0) ** -int(T))
    return 1.0 / sum(discount_factors(rates))


def idc_factor(years, r=R, *, offset=0):
    """Interest during construction, prod (1 + r_u)^dt: overnight cost -> CAPEX."""
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


# ==================== ==================== ==================== ====================
# 4. investment cost

def plant_cost(deltas, *, r=R, units=None, eur_dkk=EUR_DKK, idc=None):
    """Annual cost per technology (mio. kr./yr) from signed GW deltas. `capex` is
    overnight, `capex_idc` includes construction interest and carries the annuity;
    FOM is charged on overnight cost."""
    units = PLANT_UNITS if units is None else units
    idc = IDC_YEARS if idc is None else idc
    rows = {}
    for tech, gw in deltas.items():
        capex_meur_pr_mw, lifetime, fom_pct = units[tech]
        idc_years = idc[tech]
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


def annuity_by_tech(deltas, *, r=R, units=None, owners=None, scale=1e-3, col="annuitet"):
    """Annuity per (country, technology); index country, columns tech + 'total'.
    col="annuitet" is CAPEX only (FOM is already netted out of UP / ΔPS);
    col="total" adds FOM for a standalone resource-cost table."""
    units = PLANT_UNITS if units is None else units
    owners = PLANT_OWNERS if owners is None else owners
    cost = plant_cost(deltas, r=r, units=units)

    out = {}
    for tech, row in cost.iterrows():
        parties = owners[tech]
        share = float(row[col]) / len(parties) * scale
        for c in parties:
            bucket = out.setdefault(c, {})
            bucket[tech] = bucket.get(tech, 0.0) + share

    df = pd.DataFrame(out).T.fillna(0.0)
    df = df.reindex(columns=[t for t in units if t in df.columns])
    df["total"] = df.sum(axis=1)
    return df


# ==================== ==================== ==================== ====================
# 5. system costs

@dataclass(frozen=True)
class SysCost:
    """One system-cost item: a rate in øre/kWh times a quantity in TWh."""
    key: str
    label: str
    rate: float                   # øre/kWh
    pts: tuple                    # Pt codes whose generation carries the cost
    basis: str = "delta"          # "delta" (shock - baseline) or "level" (shock)
    split_nuclear: bool = False   # needs Nuclear split out of CD
    unit_label: str = "øre/kWh"


SYSCOST_SPECS = (
    SysCost("vre_syscost", "Systemomkostning, VE", VRE_SYSCOST_ORE_KWH,
            pts=VRE_PTS, basis="delta"),
    SysCost("nuc_syscost", "Systemomkostning, kernekraft", NUC_SYSCOST_ORE_KWH,
            pts=NUC_PTS, basis="delta", split_nuclear=True),
)


def syscost_mio(twh, rate):
    """Welfare contribution (mio. kr./yr): negative = cost, positive = saved."""
    return -float(twh) * float(rate) * ORE_KWH_PR_TWH


# ==================== ==================== ==================== ====================
# 6. LCOE from GR's own figures

LCOE_TECHS     = {"KK": "Nuclear", "WS": "WindOffshore", "WL": "WindOnshore", "PV": "PV"}
LCOE_VRE       = ("WS", "WL", "PV")
LCOE_VRE_LABEL = "VE (vægtet)"
LCOE_UNITS = {                       # FOM % unused for LCOE: FOM is taken from GR
    **PLANT_UNITS,
    "WindOnshore": (9.7 / EUR_DKK, 30, 0.0),   # ET
    "PV":          (4.4 / EUR_DKK, 40, 0.0),   # ET
}
LCOE_IDC = {**IDC_YEARS, "WindOnshore": 0.0, "PV": 0.0}


def lcoe_inputs(db, *, year, areas=c_aggs["DK"], nuc_ids=None):
    """GR figures per id for KK/WS/WL/PV in `areas`: capacity, FOM, VOM, MC,
    generation. ECap_el = ECap * (1 + Cv/Cm) for EX; MC_EX covers EX's missing MC.
    Nuclear ids are NUCLEAR_IDS from their start year, so `db` must be the shock."""
    nuc_ids = NUCLEAR_IDS if nuc_ids is None else nuc_ids
    nuc = {i for i, start in nuc_ids.items() if int(year) >= start}
    pt_of, ea_of = id_map(db("id2Pt")), id_map(db("id2EA"))

    tech_of = pt_of.where(pt_of.isin(LCOE_VRE)).astype(object)
    tech_of[tech_of.index.isin(nuc)] = "KK"
    tech_of = tech_of.dropna()

    ea = tech_of.index.map(ea_of)
    if pd.isna(ea).any():
        raise KeyError(f"ids without EA: {tech_of.index[pd.isna(ea)].tolist()}")
    tech_of = tech_of[ea.isin(areas)]
    if "KK" not in tech_of.values:
        raise KeyError(f"no nuclear ids ({sorted(nuc)}) in {areas} in {year} -- is db the nuclear shock?")
    ids = tech_of.index

    mc = sym_year(db, "MC", year)
    if "MC_EX" in db.symbols:
        mc = mc.combine_first(sym_year(db, "MC_EX", year))

    ecap = sym_year(db, "ECap", year)
    el_fac = (1 + sym_year(db, "Cv", year).div(sym_year(db, "Cm", year))).reindex(ids).fillna(1.0)

    seh = sym_year(db, "SEh", year)                  # annual totals over cluster hours
    gen = seh[seh.index.get_level_values("id").isin(ids)].groupby(level="id").sum()

    df = pd.DataFrame({
        "tech":        tech_of,
        "EA":          ea_of.reindex(ids),
        "ECap_GW":     ecap.reindex(ids),
        "ECap_el_GW":  ecap.reindex(ids) * el_fac,
        "FOM_mDKK_GW": sym_year(db, "FOM", year).reindex(ids),
        "VOM_DKK_MWh": sym_year(db, "VOM", year).reindex(ids),
        "MC_DKK_MWh":  mc.reindex(ids),
        "E_GWh":       gen.reindex(ids),
    })
    df = df[df["ECap_GW"] > 0]

    bad = df.isna().any(axis=1)
    if bad.any():
        raise ValueError(f"missing values for: {df.index[bad].tolist()}")
    return df


def lcoe_table(db, *, year, areas=c_aggs["DK"], r=R, units=None, idc=None,
               techs=LCOE_TECHS, vre=LCOE_VRE, vre_label=LCOE_VRE_LABEL):
    """LCOE per technology plus a generation-weighted VRE row (øre/kWh). Annuity
    on ECap_el (nameplate), FOM on ECap (model basis), variable part = MC."""
    units = LCOE_UNITS if units is None else units
    idc = LCOE_IDC if idc is None else idc
    df = lcoe_inputs(db, year=year, areas=areas)

    # annuity per GW nameplate (mio. kr./GW/yr): a 1 GW delta per technology
    unit_cost = plant_cost(pd.Series(1.0, index=list(techs.values())), r=r, units=units, idc=idc)
    ann_per_gw = (unit_cost["annuitet"].astype(float)
                  .rename(index={v: k for k, v in techs.items()}))

    s = df.groupby("tech")[["ECap_GW", "ECap_el_GW", "E_GWh"]].sum()
    s["ann_mio"] = ann_per_gw.reindex(s.index) * s["ECap_el_GW"]
    s["fom_mio"] = (df["FOM_mDKK_GW"] * df["ECap_GW"]).groupby(df["tech"]).sum()
    s["mc_x_e"]  = (df["MC_DKK_MWh"] * df["E_GWh"]).groupby(df["tech"]).sum()

    missing = [t for t in techs if t not in s.index]
    if missing:
        raise KeyError(f"no capacity for {missing} in {areas}, {year}")
    s = s.loc[list(techs)]
    s.loc[vre_label] = s.loc[list(vre)].sum()        # a sum -> generation-weighted on division

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
