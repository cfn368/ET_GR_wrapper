"""
Welfare — consumer and producer surplus, congestion rent and the welfare block
===============================================================================

Partial-equilibrium welfare of a shock read off the solved databases: ΔCS
(trapezoid on DC-IIA demand), ΔPS (UP x capacity, with a price/quantity split),
ΔCI (congestion rent to the TSOs), the nuclear flexibility correction, system
costs, and `welfare_block`, which nets all of it against the annuitised
investment from `costs`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .groups import (
    c_aggs, EA_TO_COUNTRY, SCAND_ZONES, NUCLEAR_IDS, PT_RENAME, HEAT_PTS,
    ANALYSIS_YEAR, REGION_ORDER, NUC_ID_BY_EA,
)
from .style import (
    STATE_SHORT, TOTAL_LABEL, dk_num, price_bin_labels, print_table, label_of, area_label,
)
from ._util import id_map, filter_years, relabel_nuclear, active_line_export, equal_hour_bins
from .costs import (
    R, SYSCOST_SPECS, NUC_SYSCOST_ORE_KWH, ORE_KWH_PR_TWH,
    annuity_by_tech, swap_deltas, syscost_mio, nuc_wind_swap,
)

DK = c_aggs["DK"]


# ==================== ==================== ==================== ====================
# 0. helpers

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
    """Base/shock (t, id) Series -> (t, Pt) frame with a delta column."""
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
    """First-order transfer and second-order adjustment. sign=-1 for consumers
    (a price rise hurts), +1 for producers (envelope: dPi/dp = net supply)."""
    return (sign * q0 * dp * scale,
            sign * 0.5 * (q1 - q0) * dp * scale)


def _countries_in(region):
    """Countries whose zones make up `region`, deduplicated, order-stable."""
    seen = []
    for ea in c_aggs[region]:
        c = EA_TO_COUNTRY.get(ea)
        if c is None:
            raise KeyError(f"{ea} has no country in EA_TO_COUNTRY")
        if c not in seen:
            seen.append(c)
    return seen


# ==================== ==================== ==================== ====================
# 1. producer surplus

def _profit(output, pt, *, ea=None, areas=None, heat_pts=HEAT_PTS):
    """UP x capacity per (t, id): electricity on ECap, heat-only types on HCap
    (restricted by Pt so CHP heat, already inside UP, is not counted twice)."""
    up = output["UP"].vals
    heat_ids = {i for i in pt.index if pt[i] in heat_pts}
    heat = (up * output["HCap"].vals).dropna()
    heat = heat[heat.index.get_level_values("id").astype(str).isin(heat_ids)]
    prof = pd.concat([(up * output["ECap"].vals).dropna(), heat])
    prof.name = "profit"
    return _scope(prof, ea, areas, what="_profit")


def _quantity(output, *, ea=None, areas=None):
    """Annual generation per (t, id) (GWh); SEh is pre-summed over cluster hours."""
    q = output["SEh"].vals.groupby(level=["t", "id"]).sum().rename("q")
    return _scope(q, ea, areas, what="_quantity")


def producer_profit(baseline_output, shock_output, *, id2Pt, id2EA=None,
                    areas=DK, year=ANALYSIS_YEAR, split_nuclear=False):
    """Producer profit (UP x capacity, mio. kr.) by (t, Pt) in `areas`. UP covers
    DK, SE and NO, so the scope must be explicit."""
    pt = id_map(id2Pt)
    ea = id_map(id2EA) if id2EA is not None else None
    return _compare(
        _profit(baseline_output, pt, ea=ea, areas=areas),
        _profit(shock_output,    pt, ea=ea, areas=areas),
        cols=("profit_base", "profit_shock"), delta="dProfit",
        pt=pt, split_nuclear=split_nuclear, year=year,
    )


def producer_generation(baseline_output, shock_output, *, id2Pt, id2EA=None,
                        areas=None, year=ANALYSIS_YEAR, split_nuclear=False):
    """Electricity generation (TWh) by (t, Pt) in `areas`."""
    pt = id_map(id2Pt)
    ea = id_map(id2EA) if id2EA is not None else None
    return _compare(
        _quantity(baseline_output, ea=ea, areas=areas),
        _quantity(shock_output,    ea=ea, areas=areas),
        cols=("q_base", "q_shock"), delta="dQ_TWh",
        pt=pt, split_nuclear=split_nuclear, year=year, scale=1e-3,
    )


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
    """Price and quantity effect per (t, id) for one market (electricity or heat)."""
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
    """Price/quantity split of ΔProfit per (t, Pt). Assumes ΔCap = 0 and unchanged
    marginal costs; otherwise the residual absorbs the difference."""
    pt, ea, ha = id_map(id2Pt), id_map(id2EA), id_map(id2HA)
    markets = (
        dict(sym_S="SEh", sym_D="DEh", sym_p="pE", area_col="EA",
             id2area=ea, id2c=id_map(id2cIdxE).to_dict()),
        dict(sym_S="SHh", sym_D="DHh", sym_p="pH", area_col="HA",
             id2area=ha, id2c=id_map(id2cIdxH).to_dict()),
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


def print_producer_profit(by_tech, *, year, areas=DK, decomp=None):
    """Producer profit by Pt; with `decomp`, the price/quantity/residual split
    (residual ~0 when capacity and marginal cost are unchanged)."""
    cols = ["profit_base", "profit_shock", "dProfit"]
    heads = [STATE_SHORT["baseline"], STATE_SHORT["shock"], "Δ"]
    df = by_tech.xs(year, level="t")[cols]
    if decomp is not None:
        df = df.join(decomp.xs(year, level="t").reindex(df.index).fillna(0.0))
        df["rest"] = df["dProfit"] - df["price_effect"] - df["qty_effect"]
        cols += ["price_effect", "qty_effect", "rest"]
        heads += ["pris", "mængde", "rest"]
    df = df[cols]
    df.index = [label_of(p) for p in df.index]
    print_table(df, [",.1f", ",.1f"] + ["+,.1f"] * (len(cols) - 2), heads=heads, total="all",
                groups=["Producentoverskud"] * 3 + ["Δ, heraf"] * (len(cols) - 3),
                title=f"Producentoverskud — {area_label(areas)}, {year}", unit="mio. kr.")


def print_producer_qty(by_qty, *, year, areas):
    """Generation by plant type (TWh) for one area scope."""
    df = by_qty.xs(year, level="t")[["q_base", "q_shock", "dQ_TWh"]]
    df.index = [label_of(p) for p in df.index]
    print_table(df, [",.2f", ",.2f", "+,.2f"], total="all",
                heads=[STATE_SHORT["baseline"], STATE_SHORT["shock"], "Δ"],
                title=f"Elproduktion — {area_label(areas)}, {year}", unit="TWh")


# ==================== ==================== ==================== ====================
# 2. consumer surplus

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
    d["p0"] = d["p0"].fillna(d["p1"])
    d["p1"] = d["p1"].fillna(d["p0"])
    d["price_effect"], d["quantity_effect"] = _trapezoid(
        d["D0"], d["D1"], d["p1"] - d["p0"], sign=-1, scale=scale)
    d["dCS"] = d["price_effect"] + d["quantity_effect"]
    d["market"] = label
    return d.rename(columns={area_col: "area"})


def consumer_surplus(baseline, shock, *, year, areas=DK, markets=("E", "H"), scale=1e-3):
    """ΔCS by market and area. markets=("E",) for areas without a district-heat
    market. Returns (by market x area, by market, with total row, per-(h, area) detail)."""
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
    """ΔCS table (electricity) by zone, with country subtotals and a grand total.
    `regions` is {country code: zones}; default DK, NO and SE."""
    if regions is None:
        regions = {c: c_aggs[c] for c in ("DK", "NO", "SE")}
    cols = ["dCS", "price_effect", "quantity_effect"]
    grand = pd.Series(0.0, index=cols)
    rows, rules = {}, []
    for country, areas in regions.items():
        bma, _, _, _ = consumer_surplus(baseline, shock, year=year, areas=areas,
                                        scale=scale, markets=("E",))
        # An area without consumers (DK3) has no demand rows: it contributes zero.
        sub = bma.xs("E", level="market")[cols].reindex(list(areas), fill_value=0.0)
        if rows:
            rules.append(len(rows))
        for zone in areas:
            rows[label_of(zone)] = sub.loc[zone].tolist()
        rows[f"{label_of(country)} i alt"] = sub.sum().tolist()
        grand += sub.sum()
    rules.append(len(rows))
    rows[TOTAL_LABEL] = grand.tolist()
    print_table(pd.DataFrame.from_dict(rows, orient="index"), "+,.1f",
                heads=["ΔCS", "pris", "mængde"], groups=["", "heraf", "heraf"], rules=rules,
                title=f"Forbrugeroverskud, el — {year}", unit="mio. kr.")


def cs_by_price_bin(baseline_output, shock_output, regions=None, *,
                    year=ANALYSIS_YEAR, n_bins=5, cs_scale=1.0):
    """ΔCS per region x equal-hour bin of the baseline price. Each zone is binned
    on its own price (hWeight hours); bin k sums the zones' k-th bins."""
    regions = list(REGION_ORDER) if regions is None else list(regions)
    hw = baseline_output("hWeight")
    p0 = filter_years(baseline_output("pE"), year, what="pE").droplevel("t")

    rows = {}
    for region in regions:
        detail = consumer_surplus(baseline_output, shock_output, year=year,
                                  areas=c_aggs[region], markets=("E",), scale=cs_scale)[3]
        acc = np.zeros(n_bins)
        for ea, g in detail.groupby("area"):
            dcs = g.groupby("h")["dCS"].sum()
            price = p0.xs(ea, level="EA").reindex(dcs.index)
            w = hw.reindex(dcs.index)
            if price.isna().any() or w.isna().any():
                raise ValueError(f"pE/hWeight missing for some states in {ea}, {year}")
            b = equal_hour_bins(price, w, n_bins)
            acc += dcs.groupby(b).sum().reindex(range(n_bins), fill_value=0.0).to_numpy()
        rows[region] = acc
    return pd.DataFrame(rows, index=price_bin_labels(n_bins)).T


def cs_by_agg_price_bin(baseline_output, shock_output, d, region, members, *,
                        year, n_bins=5, cs_scale=1.0):
    """ΔCS per equal-hour bin of the aggregate baseline price in `d` (state prices
    of the region). Every hour h lands in one bin for the whole region, so the
    bars line up with the binned means of the price scatter."""
    detail = consumer_surplus(baseline_output, shock_output, year=year,
                              areas=list(members), markets=("E",), scale=cs_scale)[3]
    dcs = detail.groupby("h")["dCS"].sum()

    bins = equal_hour_bins(d["baseline"], d["hours"], n_bins)
    bins.index = bins.index.astype(str)             # h may be int in one source, str in the other
    b = bins.reindex(dcs.index.astype(str))
    if b.isna().any():
        missing = list(dcs.index[b.isna().to_numpy()][:5])
        raise ValueError(f"No aggregate price for some states in {region}, {year}: {missing}")

    per_bin = dcs.groupby(b.to_numpy()).sum().reindex(range(n_bins), fill_value=0.0)
    if not np.isclose(per_bin.sum(), dcs.sum()):
        raise ValueError("Binned ΔCS does not sum to total ΔCS")
    return pd.DataFrame([per_bin.to_numpy()], index=[region], columns=price_bin_labels(n_bins))


# ==================== ==================== ==================== ====================
# 3. congestion rent

def _ci_by_direction(db, year):
    """CI (mio. DKK) and flow (TWh) per line-direction: CI = (eta*p_EAEA - p_EA) *
    lineExport. lineExport is TWh pre-summed over cluster hours: plain sum, no hWeight."""
    le = active_line_export(db, years=[year])
    eta = db("trEff").rename("trEff").reset_index().rename(columns={"trEff": "eta"})
    p = filter_years(db("pE").rename("pE").reset_index(), year, what="pE")[["h", "EA", "pE"]]

    df = (le.merge(eta, on=["t", "line"], how="left")
            .merge(p, on=["h", "EA"], how="left")
            .merge(p.rename(columns={"EA": "EAEA", "pE": "pE_EAEA"}),
                   on=["h", "EAEA"], how="left"))
    if df[["eta", "pE", "pE_EAEA"]].isna().any().any():
        raise ValueError("merge left NaNs in eta/pE -- check keys")

    df["CI"] = (df["eta"] * df["pE_EAEA"] - df["pE"]) * df["lineExport"]
    return df.groupby(["line", "EA", "EAEA"]).agg(
        CI=("CI", "sum"), TWh=("lineExport", "sum")
    ).reset_index()


def ci_by_country(db, year, *, tso_share=0.5):
    """CI accruing to each Scandinavian country's TSOs (mio. DKK), plus a total.
    Each direction pays tso_share to the zone at each end, so an internal line
    (e.g. Storebælt) returns 100 pct. to its country."""
    d = _ci_by_direction(db, year)
    half = d.assign(half=d["CI"] * tso_share)

    by_zone = (half.groupby("EA")["half"].sum()
                 .add(half.groupby("EAEA")["half"].sum(), fill_value=0))
    by_zone = by_zone[by_zone.index.isin(SCAND_ZONES)]

    zones = by_zone.rename("CI_mioDKK").to_frame()
    zones["land"] = zones.index.map(EA_TO_COUNTRY)

    by_country = zones.groupby("land")["CI_mioDKK"].sum().sort_values(ascending=False)
    by_country.loc["Scandi i alt"] = by_country.sum()
    return zones.round(1), by_country.round(1).to_frame()


def print_ci(baseline_output, shock_output, *, year, country="DK", tso_share=0.5):
    """Congestion rent to one country's TSOs by zone, baseline vs. shock."""
    zb, _ = ci_by_country(baseline_output, year, tso_share=tso_share)
    zs, _ = ci_by_country(shock_output, year, tso_share=tso_share)
    zones = [z for z in zb.index if zb.loc[z, "land"] == country]
    df = pd.DataFrame({"b": zb.loc[zones, "CI_mioDKK"],
                       "s": zs["CI_mioDKK"].reindex(zones, fill_value=0.0)})
    df["d"] = df["s"] - df["b"]
    df.index = [label_of(z) for z in zones]
    print_table(df, [",.1f", ",.1f", "+,.1f"], total="all",
                heads=[STATE_SHORT["baseline"], STATE_SHORT["shock"], "Δ"],
                title=f"Flaskehalsindtægt til TSO — {area_label(country)}, {year}", unit="mio. kr.")


def _ci_delta_by_country(baseline_output, shock_output, year):
    """Δ congestion rent by country (mio. kr.), shock - baseline."""
    _, c_base = ci_by_country(baseline_output, year)
    _, c_shock = ci_by_country(shock_output, year)
    b = c_base["CI_mioDKK"].groupby(level=0).sum()
    s = c_shock["CI_mioDKK"].groupby(level=0).sum()
    return s.subtract(b, fill_value=0.0)


# ==================== ==================== ==================== ====================
# 4. nuclear flexibility correction

def nuc_flex_value(db, *, year, u_min=1.0, syscost=NUC_SYSCOST_ORE_KWH):
    """Per nuclear id + total: output, extra TWh when forced to run at u_min of
    capacity in every hour, average price of the extra output, and the welfare
    change dW (mio. kr./yr). dW < 0; the value of GR's flexibility is -dW."""
    ids = [str(i) for i, start in NUCLEAR_IDS.items() if int(year) >= start]
    m = db("id2EA").to_frame(index=False).astype(str)
    m.columns = ["id", "EA"]

    def _long(sym):
        s = db(sym).rename(sym)
        if "t" in s.index.names:
            s = filter_years(s, year, what=sym).droplevel("t")
        df = s.reset_index()
        return df.astype({c: str for c in df.columns if c != sym})

    df = _long("SEh")
    df = df[df["id"].isin(ids)].merge(m, on="id")
    if set(df["id"]) != set(ids):
        raise KeyError(f"nuclear ids missing from SEh in {year}: {sorted(set(ids) - set(df['id']))}")
    for sym, on in (("pE", ["h", "EA"]), ("hWeight", ["h"]), ("ECap", ["id"]), ("MC", ["id"])):
        df = df.merge(_long(sym), on=on, how="left")
    if df[["pE", "hWeight", "ECap", "MC"]].isna().any().any():
        raise ValueError("missing pE, hWeight, ECap or MC after merge")

    q = df["SEh"] / 1000.0                                           # GWh -> TWh
    df["dq"] = (u_min * df["hWeight"] * df["ECap"] / 1000.0 - q).clip(lower=0.0)
    df["dW"] = (df["pE"] - df["MC"] - syscost * ORE_KWH_PR_TWH) * df["dq"]
    df["q"], df["p_dq"] = q, df["pE"] * df["dq"]

    s = df.groupby("id")[["q", "dq", "p_dq", "dW"]].sum()
    s.loc["total"] = s.sum()
    s["p_extra"] = s["p_dq"] / s["dq"]
    return s.rename(columns={"q": "TWh", "dq": "dTWh"})[["TWh", "dTWh", "p_extra", "dW"]]


def print_flex(db, *, year, u_mins=(1.0,), syscost=NUC_SYSCOST_ORE_KWH):
    """`nuc_flex_value` per minimum load: one table per u_min, plants then total."""
    ea_of = {i: ea for ea, i in NUC_ID_BY_EA.items()}
    for u in u_mins:
        f = nuc_flex_value(db, year=year, u_min=u, syscost=syscost)
        f.index = [TOTAL_LABEL if i == "total" else f"{label_of('Nuclear')} {ea_of.get(i, i)}"
                   for i in f.index]
        print_table(f, [",.2f", ",.2f", ",.1f", "+,.1f"],
                    heads=["produktion", "ekstra", "pris, ekstra", "ΔW"],
                    groups=["TWh", "TWh", "kr./MWh", "mio. kr."], rules=[len(f) - 1],
                    title=f"Fleksibilitetsværdi, kernekraft — {year}, "
                          f"minimumslast {dk_num(100 * u)} pct.")


# ==================== ==================== ==================== ====================
# 5. welfare by region

def welfare_levels(baseline_output, shock_output, *, id2Pt=None, id2EA=None, regions=None,
                   year=ANALYSIS_YEAR, markets=("E",), cs_scale=1.0,
                   include_ci=True, annuity=None):
    """ΔCS / ΔPS / ΔCI / annuity / net per region (mia. kr.). `annuity` is
    {country: mio. kr./yr}; a region without its own key sums its countries."""
    regions = list(REGION_ORDER) if regions is None else list(regions)
    ci = _ci_delta_by_country(baseline_output, shock_output, year) if include_ci else None
    ann = {} if annuity is None else dict(annuity)

    rows = []
    for region in regions:
        areas = c_aggs[region]
        countries = _countries_in(region)

        _, _, total, _ = consumer_surplus(baseline_output, shock_output, year=year,
                                          areas=areas, markets=markets, scale=cs_scale)
        dcs = float(total.loc["+".join(markets), "dCS"]) / 1000.0

        prof = producer_profit(baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
                               areas=areas, year=year)
        dps = float(prof["dProfit"].sum()) / 1000.0

        dci = 0.0 if ci is None else sum(float(ci.get(c, 0.0)) for c in countries) / 1000.0
        cap = (ann[region] if region in ann
               else sum(ann.get(c, 0.0) for c in countries)) / 1000.0

        rows.append({"region": region, "dCS": dcs, "dPS": dps, "dCI": dci,
                     "annuity": -cap, "net": dcs + dps + dci - cap})
    return pd.DataFrame(rows).set_index("region")


# ==================== ==================== ==================== ====================
# 6. welfare block: raw model output -> printable table in one call

WELFARE_ROWS = (                     # model components; their sum is net
    ("dCS",     "Δ Forbrugeroverskud"),
    ("dPS",     "Δ Producentoverskud"),
    ("dCI",     "Δ Flaskehalsindtægter"),
    ("annuity", "Annuitet, netto"),
)
NET_ROW      = ("net", "Netto velfærdseffekt")
TECH_LABELS  = {"Nuclear": "Kernekraft (ny)", "WindOffshore": "Havvind (nedlagt)"}
REGION_COUNTRIES = {"DK": ("DK",), "NO": ("NO",), "SE": ("SE",), "Scandi": ("DK", "NO", "SE")}
UNIT_SCALE = {"mio": 1000.0, "mia": 1.0}      # welfare_levels returns mia. kr.
UNIT_LABEL = {"mio": "mio. kr./år", "mia": "mia. kr./år"}


def syscost_twh(baseline_output, shock_output, spec, *, year, areas, id2Pt, id2EA,
                base_tol=1e-6):
    """The quantity behind `spec` (TWh): "delta" = shock - baseline generation of
    spec.pts, "level" = shock generation (requires a zero baseline)."""
    gen = producer_generation(baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
                              areas=areas, year=year, split_nuclear=spec.split_nuclear)
    sel = [str(k) in spec.pts for k in gen.index.get_level_values("Pt")]
    if not any(sel):
        raise KeyError(f"{spec.key}: none of {spec.pts} in the generation table")

    if spec.basis == "delta":
        return float(gen["dQ_TWh"][sel].sum())
    if spec.basis == "level":
        q_base = float(gen["q_base"][sel].sum())
        if abs(q_base) > base_tol:
            raise ValueError(f"{spec.key}: baseline is {q_base:.6f} TWh, not zero -- use basis='delta'")
        return float(gen["q_shock"][sel].sum())
    raise ValueError(f"basis must be 'delta' or 'level', got {spec.basis!r}")


def welfare_block(baseline_output, shock_output, *, year, deltas=None, annuity=None,
                  regions=("DK",), markets=("E",), include_ci=True, include_heat=True,
                  id2Pt=None, id2EA=None, r=R, units=None, owners=None, col="annuitet",
                  unit="mio", syscosts=SYSCOST_SPECS, rates=None, twh=None, nuc_u_min=0.3):
    """Welfare decomposition + annuity + system costs, components x regions.

    The annuity is built from the capacity swap read off the databases, unless
    `deltas` (signed GW per technology) or `annuity` ({country: mio. kr./yr} or an
    `annuity_by_tech` frame) is given. With `nuc_u_min`, ΔPS and the nuclear system
    cost carry the flexibility correction of `nuc_flex_value`; None switches it off.
    """
    if unit not in UNIT_SCALE:
        raise ValueError(f"unit must be one of {tuple(UNIT_SCALE)}, got {unit!r}")
    if annuity is not None and r != R:
        raise ValueError("r= is ignored when annuity= is given; build the annuity "
                         "with the intended rate (annuity_by_tech(..., r=...))")

    scale = UNIT_SCALE[unit]
    regions = list(regions)
    id2Pt = shock_output["id2Pt"].vals if id2Pt is None else id2Pt
    id2EA = shock_output["id2EA"].vals if id2EA is None else id2EA
    rates, twh = dict(rates or {}), dict(twh or {})

    flex = None
    if nuc_u_min is not None:
        flex = nuc_flex_value(shock_output, year=year, u_min=nuc_u_min, syscost=0).loc["total"]

    if annuity is None:
        if deltas is None:
            nuc, cut = nuc_wind_swap(baseline_output, shock_output, yr=year, verbose=False)
            deltas = swap_deltas(nuc, cut, include_heat=include_heat)
        ann_mio = annuity_by_tech(deltas, r=r, units=units, owners=owners, scale=1.0, col=col)
    elif hasattr(annuity, "columns"):
        ann_mio = annuity.copy()
        if "total" not in ann_mio.columns:
            ann_mio["total"] = ann_mio.sum(axis=1)
    else:
        ann_mio = pd.DataFrame({"total": pd.Series(annuity, dtype=float)})

    ann = ann_mio * (scale / 1000.0)
    items = [c for c in ann.columns if c != "total"]

    w = welfare_levels(
        baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
        year=year, markets=markets, regions=regions, include_ci=include_ci,
        annuity=ann_mio["total"].to_dict(),
    ) * scale

    used_rate = {s.key: float(rates.get(s.key, s.rate)) for s in syscosts}
    used_twh = {s.key: {} for s in syscosts}

    out = {}
    for region in regions:
        s = {k: float(w.loc[region, k]) for k, _ in WELFARE_ROWS}
        net_model = float(w.loc[region, NET_ROW[0]])
        net_chk = sum(s[k] for k, _ in WELFARE_ROWS)
        if abs(net_chk - net_model) >= 1e-6:
            raise AssertionError(f"{region}: net {net_model:.4f} != sum of components {net_chk:.4f}")

        lands = REGION_COUNTRIES.get(region, (region,))
        has_nuc = flex is not None and "DK" in lands      # the nuclear plants are in DK
        if has_nuc:
            s["dPS"] += float(flex["dW"]) * (scale / 1000.0)

        for item in items:
            s["ann_" + item] = sum(float(ann.loc[c, item]) for c in lands if c in ann.index)
        if items:
            ann_sum = sum(s["ann_" + i] for i in items)
            if abs(ann_sum + s["annuity"]) >= 1e-6:
                raise AssertionError(f"{region}: annuity {s['annuity']:.4f} != -sum ann_* {-ann_sum:.4f}")

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
    """{name: welfare_block} per discount-rate schedule; `schedules` is {name: r}."""
    return {name: welfare_block(baseline_output, shock_output, r=sched, **kw)
            for name, sched in schedules.items()}


def _blocks_attr(blocks, key, default=None):
    """A block attribute shared by all blocks; raises if they disagree."""
    vals = [b.attrs.get(key, default) for b in blocks.values()]
    if any(v != vals[0] for v in vals[1:]):
        raise ValueError(f"blocks disagree on attrs[{key!r}]: {vals}")
    return vals[0]


def print_welfare_blocks(blocks, *, regions=None, dec=2):
    """One or more welfare blocks side by side (e.g. rate schedules), one table per
    region. `blocks` is {column name: block} or a single block."""
    if hasattr(blocks, "columns"):
        blocks = {"effekt": blocks}
    blocks = dict(blocks)
    if not blocks:
        raise ValueError("no blocks to print")
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

    fmt = f"+,.{dec}f"
    for region in regions:
        vals = lambda key: [float(blocks[n].loc[key, region]) for n in names]
        rows = {lbl: vals(key) for key, lbl in list(WELFARE_ROWS) + [(sp.key, sp.label) for sp in specs]}
        rules = [len(rows)]
        rows[NET_ROW[1]] = vals(NET_ROW[0])
        if techs:
            rules.append(len(rows))
            for t in techs:
                rows[f"Annuitet, {TECH_LABELS.get(t, t).lower()}"] = vals("ann_" + t)

        notes = []
        if techs:
            notes.append("Annuitet: + = omkostning, - = sparet.")
        for sp in specs:                             # quantities do not depend on the rate
            name = sp.label.split(",")[-1].strip()
            basis = "niveau" if sp.basis == "level" else "ændring"
            qs = [twh[n].get(sp.key, {}).get(region) for n in names]
            rs = [float(rates[n].get(sp.key, sp.rate)) for n in names]
            q_txt = lambda q: "n/a" if q is None else dk_num(q, 3, True) + " TWh"
            if (all(q is not None for q in qs) and max(qs) - min(qs) < 1e-9
                    and len(set(rs)) == 1):
                notes.append(f"Systemomkostning, {name}: {q_txt(qs[0])} ({basis}) "
                             f"x {dk_num(rs[0], 0)} {sp.unit_label}")
            else:
                notes += [f"Systemomkostning, {name}, {n}: {q_txt(q)} ({basis}) "
                          f"x {dk_num(rr, 0)} {sp.unit_label}" for n, q, rr in zip(names, qs, rs)]

        print_table(pd.DataFrame.from_dict(rows, orient="index", columns=names), fmt,
                    rules=rules, notes=notes, unit=unit,
                    title=f"Velfærdseffekt — {area_label(region)}"
                          + (f", {year}" if year else ""))
