"""Producer- and consumer-side welfare proxies for GreenREFORM shock scenarios.

profit_i = UP_i * cap_i   (million DKK/GW * GW = million DKK)"""
import pandas as pd

from .groups import c_aggs, NUCLEAR_IDS, PT_RENAME, ANALYSIS_YEAR
from .energy_balance_build import relabel_nuclear

HEAT_PTS = {"BH", "EPH", "SH", "HS"}   # UP per heat capacity, no ECap
# NB: HS still in — settled convention excludes it pending HCap dispatch-vs-reservoir.

DK = c_aggs["DK"]
NO = c_aggs["NO"]
SE = c_aggs["SE"]


# --- shared helpers ============================================
def _id_map(m):
    """id->category Series from a 2-level GpyDB mapping (accepts symbol or index)."""
    mi = m.vals if hasattr(m, "vals") else m
    return pd.Series(mi.get_level_values(-1), index=mi.get_level_values(0).astype(str))


def _scope(vals, ea, areas, *, what):
    """Keep (t, id) rows whose id sits in `areas`. Raises on any unmapped id."""
    if areas is None:
        return vals
    if ea is None:
        raise ValueError(f"areas given but no id2EA map passed to {what}.")
    ids = vals.index.get_level_values("id").astype(str)
    mapped = ids.map(ea)
    if mapped.isna().any():
        miss = sorted(set(ids[mapped.isna()]))[:5]
        raise KeyError(f"{what}: {int(mapped.isna().sum())} ids have no EA: {miss}")
    return vals[mapped.isin(areas)]


def _by_tech(vals, name, pt, split_nuclear=False):
    """Sum a (t, id) Series to (t, Pt); unmapped ids surface as 'UNMAPPED'."""
    df = vals.astype("float64").rename(name).reset_index()
    df["Pt"] = df["id"].astype(str).map(pt).replace(PT_RENAME)
    if split_nuclear:
        df = relabel_nuclear(df, NUCLEAR_IDS)
    df["Pt"] = df["Pt"].fillna("UNMAPPED")
    return df.groupby(["t", "Pt"])[name].sum()


def _compare(base, shock, *, cols, delta, pt, split_nuclear, year, scale=1.0):
    """Base/shock (t, id) Series -> (t, Pt) frame with a live delta column."""
    if year is not None:
        base = base.xs(year, level="t", drop_level=False)
        shock = shock.xs(year, level="t", drop_level=False)
    out = pd.concat(
        {cols[0]: _by_tech(base,  "v", pt, split_nuclear),
         cols[1]: _by_tech(shock, "v", pt, split_nuclear)}, axis=1,
    ).fillna(0.0) * scale
    out[delta] = out[cols[1]] - out[cols[0]]
    return out.sort_values(delta)


def _trapezoid(q0, q1, dp, *, sign, scale):
    """First-order transfer and second-order adjustment. sign=-1 consumers
    (price rise hurts), +1 producers (envelope: dPi/dp = net supply)."""
    return (sign * q0 * dp * scale,
            sign * 0.5 * (q1 - q0) * dp * scale)


def _print_table(title, df, cols, heads, fmt, *, w=12, label_w=6):
    """Shared table printer: header, rows, rule, summed total."""
    print(title)
    hdr = f'{"":<8}' + "".join(f"{h:>{w}}" for h in heads)
    print(hdr)
    rule = "  " + "─" * (len(hdr) - 2)
    print(rule)
    for key, row in df.iterrows():
        print(f"  {key:<{label_w}}" + "".join(f"{row[c]:>{w},{fmt}}" for c in cols))
    print(rule)
    tot = df[cols].sum()
    print(f'  {"Total":<{label_w}}' + "".join(f"{tot[c]:>{w},{fmt}}" for c in cols))
    print()


def annuity(capex_meur_per_mw, lifetime, r=0.04, EUR_DKK=7.458):
    return capex_meur_per_mw * 1000 * EUR_DKK * (r / (1 - (1 + r)**(-lifetime)))


# --- producer levels ===========================================
def _profit(output, pt, *, ea=None, areas=None, heat_pts=HEAT_PTS):
    """UP x capacity per (t, id). Elec on ECap; heat-only (HEAT_PTS) on HCap,
    Pt-restricted so CHP heat isn't double-counted (already inside UP)."""
    up = output["UP"].vals
    heat_ids = {i for i in pt.index if pt[i] in heat_pts}
    heat = (up * output["HCap"].vals).dropna()
    heat = heat[heat.index.get_level_values("id").astype(str).isin(heat_ids)]
    prof = pd.concat([(up * output["ECap"].vals).dropna(), heat])  # disjoint
    prof.name = "profit"
    return _scope(prof, ea, areas, what="_profit")


def _quantity(output, *, ea=None, areas=None):
    """Annual electricity generation per (t, id) (GWh) from SEh (pre-summed
    over cluster hours -> plain sum)."""
    q = output["SEh"].vals.groupby(level=["t", "id"]).sum().rename("q")
    return _scope(q, ea, areas, what="_quantity")


def producer_profit(baseline_output, shock_output, *, id2Pt, id2EA=None,
                    areas=DK, year=ANALYSIS_YEAR, split_nuclear=False):
    """Producer profit (UP x capacity) by (t, Pt), scoped to `areas`. UP covers
    DK1/DK2 + SE1-4 + NOS/NOM/NON, so the scope must be explicit."""
    pt = _id_map(id2Pt)
    ea = _id_map(id2EA) if id2EA is not None else None
    return _compare(
        _profit(baseline_output, pt, ea=ea, areas=areas),
        _profit(shock_output,    pt, ea=ea, areas=areas),
        cols=("profit_base", "profit_shock"), delta="dProfit",
        pt=pt, split_nuclear=split_nuclear, year=year,
    )


def producer_generation(baseline_output, shock_output, *, id2Pt, id2EA=None,
                        areas=None, year=ANALYSIS_YEAR, split_nuclear=False):
    """Electricity generation (TWh) by (t, Pt), scoped to `areas`."""
    pt = _id_map(id2Pt)
    ea = _id_map(id2EA) if id2EA is not None else None
    return _compare(
        _quantity(baseline_output, ea=ea, areas=areas),
        _quantity(shock_output,    ea=ea, areas=areas),
        cols=("q_base", "q_shock"), delta="dQ_TWh",
        pt=pt, split_nuclear=split_nuclear, year=year, scale=1e-3,
    )


# --- producer decomposition ====================================
def _net_flow(output, *, sym_S, sym_D, id2c):
    """Net energy flow per (t, h, id): output minus own consumption (storage)."""
    s = output[sym_S].vals.rename("q").reset_index()
    d = output[sym_D].vals.rename("d").reset_index()
    inv = {}
    for k, v in id2c.items():                       # cIdx -> id, must be injective
        if v in inv:
            raise ValueError(f"cIdx {v!r} maps to both {inv[v]!r} and {k!r}")
        inv[v] = k
    d["id"] = d["cIdx"].astype(str).map(inv)
    d = d.dropna(subset=["id"]).groupby(["t", "h", "id"], as_index=False)["d"].sum()
    m = s.merge(d, on=["t", "h", "id"], how="outer").fillna(0.0)
    return m.set_index(["t", "h", "id"]).eval("q - d").rename("flow")


def _effects_one(base, shock, *, sym_S, sym_D, sym_p, area_col, id2area, id2c, scale):
    """Price and quantity effect per (t, id) for one market (elec or heat)."""
    f = (_net_flow(base,  sym_S=sym_S, sym_D=sym_D, id2c=id2c).reset_index()
         .merge(_net_flow(shock, sym_S=sym_S, sym_D=sym_D, id2c=id2c).reset_index(),
                on=["t", "h", "id"], how="outer", suffixes=("0", "1")).fillna(0.0))
    f["area"] = f["id"].astype(str).map(id2area)
    f = f.dropna(subset=["area"])                   # ids absent from this market
    for tag, db in (("0", base), ("1", shock)):
        p = db[sym_p].vals.rename("p" + tag).reset_index().rename(columns={area_col: "area"})
        f = f.merge(p, on=["t", "h", "area"], how="left")
    if f[["p0", "p1"]].isna().any(axis=None):
        n = int(f[["p0", "p1"]].isna().any(axis=1).sum())
        raise KeyError(f"missing {sym_p} for {n} rows")

    f["price_effect"], f["qty_effect"] = _trapezoid(
        f["flow0"], f["flow1"], f["p1"] - f["p0"], sign=1, scale=scale)
    return f.groupby(["t", "id"])[["price_effect", "qty_effect"]].sum()


def profit_decomposition(baseline_output, shock_output, *, id2Pt, id2EA, id2HA,
                         id2cIdxE, id2cIdxH, areas=DK, year=ANALYSIS_YEAR,
                         split_nuclear=False, scale=1e-3):
    """Price/quantity split of dProfit per (t, Pt). Assumes dCap = 0 and
    unchanged marginal costs -- otherwise the residual absorbs the difference."""
    pt, ea, ha = _id_map(id2Pt), _id_map(id2EA), _id_map(id2HA)
    markets = (
        dict(sym_S="SEh", sym_D="DEh", sym_p="pE", area_col="EA",
             id2area=ea, id2c=_id_map(id2cIdxE).to_dict()),
        dict(sym_S="SHh", sym_D="DHh", sym_p="pH", area_col="HA",
             id2area=ha, id2c=_id_map(id2cIdxH).to_dict()),
    )
    eff = None
    for kw in markets:                              # CHP picks up both channels
        one = _effects_one(baseline_output, shock_output, scale=scale, **kw)
        eff = one if eff is None else eff.add(one, fill_value=0.0)

    eff = _scope(eff, ea, areas, what="profit_decomposition")
    if year is not None:
        eff = eff.xs(year, level="t", drop_level=False)
    return pd.concat(
        {c: _by_tech(eff[c], c, pt, split_nuclear) for c in eff.columns}, axis=1
    ).fillna(0.0)


# --- producer printers =========================================
def print_producer_profit(by_tech, *, year, areas=DK, decomp=None):
    """Producer profit by Pt. With `decomp`, adds price/quantity/residual;
    residual = dProfit - price - qty, ~0 when capacity and MC are unchanged."""
    cols = ["profit_base", "profit_shock", "dProfit"]
    heads = ["base", "shock", "Δprofit"]
    df = by_tech.xs(year, level="t")[cols]
    if decomp is not None:
        df = df.join(decomp.xs(year, level="t").reindex(df.index).fillna(0.0))
        df["rest"] = df["dProfit"] - df["price_effect"] - df["qty_effect"]
        cols += ["price_effect", "qty_effect", "rest"]
        heads += ["pris", "mængde", "rest"]
    _print_table(f"Producer profit — {year}, {'+'.join(areas)}  (million DKK)",
                 df, cols, heads, ".1f")


def print_producer_qty(by_qty, *, year, areas):
    """Generation by Pt (TWh) for one area scope."""
    _print_table(f"Generation — {year}, {'+'.join(areas)}  (TWh)",
                 by_qty.xs(year, level="t"),
                 ["q_base", "q_shock", "dQ_TWh"], ["base", "shock", "ΔTWh"], ".3f")


# --- consumer ==================================================
def _cs_core(qD0, qD1, p0, p1, area_col, *, year, areas, scale, label):
    """Trapezoid ΔCS per (h, area) for a single year."""
    _f = lambda s, n: s.rename(n).reset_index()
    keys = ["t", "h", area_col]
    d = (_f(qD0, "D0").merge(_f(qD1, "D1"), on=keys, how="outer")
                      .merge(_f(p0, "p0"),  on=keys, how="outer")
                      .merge(_f(p1, "p1"),  on=keys, how="outer"))
    d = d[(d["t"] == year) & (d[area_col].isin(set(areas)))].copy()
    if d.empty:
        raise ValueError(f"no {label} rows for year {year} in areas {areas}")
    d[["D0", "D1"]] = d[["D0", "D1"]].fillna(0.0)
    d["p0"] = d["p0"].fillna(d["p1"]); d["p1"] = d["p1"].fillna(d["p0"])
    d["price_effect"], d["quantity_effect"] = _trapezoid(
        d["D0"], d["D1"], d["p1"] - d["p0"], sign=-1, scale=scale)
    d["dCS"] = d["price_effect"] + d["quantity_effect"]
    d["market"] = label
    return d.rename(columns={area_col: "area"})


def consumer_surplus(baseline, shock, *, year, areas=DK, markets=("E", "H"), scale=1e-3):
    """ΔCS by market/area. `markets` picks which of ('E','H') to compute --
    pass ('E',) for areas with no district-heat market (e.g. foreign EAs)."""
    if isinstance(areas, str):
        areas = (areas,)
    spec = {"E": ("qDemandE_DCIIA", "pE", "EA"), "H": ("qDemandH_DCIIA", "pH", "HA")}
    parts = [
        _cs_core(baseline[q].vals, shock[q].vals, baseline[p].vals, shock[p].vals,
                 a, year=year, areas=areas, scale=scale, label=m)
        for m in markets for q, p, a in [spec[m]]
    ]
    d = pd.concat(parts, ignore_index=True)
    cols = ["dCS", "price_effect", "quantity_effect"]
    by_market = d.groupby("market")[cols].sum()
    total = by_market.copy()
    total.loc["+".join(markets)] = by_market.sum()
    return d.groupby(["market", "area"])[cols].sum(), by_market, total, d


def print_cs_table(baseline, shock, *, year, regions=None, scale=1.0):
    """ΔCS table (electricity only) with per-country subtotals + grand total."""
    if regions is None:
        regions = {"Denmark": DK, "Norway": NO, "Sweden": SE}
    cols = ["dCS", "price_effect", "quantity_effect"]
    grand = pd.Series(0.0, index=cols)
    line = lambda lab, v: f'{"  " + lab:<16}' + "".join(f"{x:>12,.1f}" for x in v)

    print(f"ΔCS {year} — million DKK (electricity)\n")
    print(f'{"":<16}{"ΔCS":>12}{"price":>12}{"quantity":>12}')
    print("─" * 52)
    for country, areas in regions.items():
        bma, _, _, _ = consumer_surplus(baseline, shock, year=year, areas=areas,
                                        scale=scale, markets=("E"))
        # An area without consumers (DK3) has no demand rows: it contributes zero.
        sub = bma.xs("E", level="market")[cols].reindex(list(areas), fill_value=0.0)
        print(country)
        for zone in areas:
            print(line(zone, sub.loc[zone]))
        print(line(f"{country} total", sub.sum()))
        grand += sub.sum()
    print("═" * 52)
    print(f'{"Total":<16}' + "".join(f"{v:>12,.1f}" for v in grand))