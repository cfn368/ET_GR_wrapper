"""
Intra-yearly energy balance — the hFull-expanded frames every analysis reads
============================================================================

Builds, per state (baseline, shock), `supply_dict` / `demand_dict` (one row per
t, area, hFull; supply >= 0, demand <= 0, GWh per real hour) and `NX_dict`
(index t, hFull, EA, EAEA; net export in TWh per real hour, positive = export
from EA). Electricity, a Denmark-wide collapse of it, and district heat.

Conventions: explicit merges on the id2*/cIdx2* maps, KF26 renames (V2G plant,
transport demand), PV-battery clusters as one net injection (PVES), distribution
loss D*(1/distrEff - 1) on the DC-IIA classes, transmission loss on the receiving
area, nuclear by explicit ids and start year, and a hard error if a non-zero
series falls outside the column order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .groups import (
    expand_areas,
    _META_BUILD, _FULL_ORDER, _META_BUILD_HA, _FULL_ORDER_HA,
    CLAMP_SUPPLY, CLAMP_DEMAND, SPLIT_COLS, CLAMP_SUPPLY_HA, CLAMP_DEMAND_HA,
    PT_RENAME, CTYPE_RENAME, DCIIA_TYPES, NUCLEAR_IDS, STATE_BASE, STATE_SHOCK, c_aggs,
)
from ._util import (
    has_sym, sym_frame, set_frame, mapping, require_mapped, filter_years,
    relabel_nuclear, cluster_members, active_line_export,
)


# ==================== ==================== ==================== ====================
# 0. container

@dataclass
class ElBalanceContext:
    """The built frames per state, plus the source databases (for capacities)."""

    supply_dict: Mapping[str, pd.DataFrame]
    demand_dict: Mapping[str, pd.DataFrame]
    NX_dict: Mapping[str, pd.DataFrame] | None
    db_dict: Mapping[str, object]


def _db_dict(baseline_output, shock_output):
    """{"baseline": db[, "shock": db]}. No copy: the build only reads."""
    db_dict = {STATE_BASE: baseline_output}
    if shock_output is not None:
        db_dict[STATE_SHOCK] = shock_output
    return db_dict


def _hours(db):
    """hWeight and the h -> hFull expansion, read once per state."""
    return sym_frame(db, "hWeight"), set_frame(db, "h2hFull")


# ==================== ==================== ==================== ====================
# 1. electricity build

def _cluster_net(db, years=None):
    """Net cluster injection by (t, EA, h) in TWh over the cluster state (the same
    unit as SEh / 1000). Positive is an export from the cluster into the area."""
    missing = {"qClusterNet", "cluster2EA"} - set(map(str, db.symbols.keys()))
    if missing:
        raise KeyError(f"PV-battery clusters exist (id2Cluster) but {sorted(missing)} is missing")
    net = filter_years(db("qClusterNet").rename("PVES").reset_index(), years, what="qClusterNet")
    net = net.merge(mapping(db, "cluster2EA", "cluster"), on="cluster", how="left")
    require_mapped(net, ["EA"], "qClusterNet")
    return net.groupby(["t", "EA", "h"], as_index=False)["PVES"].sum()


def _supply_seh(db, nuclear_ids, hw, h2hf, years=None):
    """Per-hour supply by production type Pt (TWh per real hour), nuclear split out."""
    df = sym_frame(db, "SEh", years)
    df = (df.merge(mapping(db, "id2Pt", "id"), on="id", how="left")
            .merge(mapping(db, "id2EA", "id"), on="id", how="left"))
    require_mapped(df, ["Pt", "EA"], "SEh")
    df["Pt"] = df["Pt"].replace(PT_RENAME)
    df = relabel_nuclear(df, nuclear_ids)

    members = cluster_members(db)
    if members:
        df = df[~df["id"].astype(str).isin(members)]

    df["SEh_pos"] = df["SEh"].clip(lower=0)
    wide = df.groupby(["t", "EA", "h", "Pt"])["SEh_pos"].sum().unstack("Pt") / 1000
    front = [c for c in ("ES", "HYS") if c in wide.columns]
    wide = wide[front + [c for c in wide.columns if c not in front]]
    wide.columns.name = None
    wide = wide.reset_index()

    if members:
        wide = wide.merge(_cluster_net(db, years), on=["t", "EA", "h"], how="left")

    wide = wide.merge(sym_frame(db, "pE", years), on=["t", "EA", "h"], how="left")
    wide = wide.merge(hw, on="h", how="left")
    wide = wide.merge(h2hf, on="h", how="left")

    keep = {"t", "EA", "h", "pE", "hWeight", "hFull"}
    for col in [c for c in wide.columns if c not in keep]:
        wide[col] = wide[col] / wide["hWeight"]

    wide = wide.sort_values(["t", "EA", "pE"])
    return wide.fillna(0)


def _transmission_loss(db, line, hw, years=None):
    """Energy lost on the interconnectors, booked on the receiving area
    (index t, h, EA; TWh per real hour, the same scale as the demand series)."""
    if not has_sym(db, "trEff"):
        raise KeyError("trEff is missing: cannot compute the transmission loss")
    eff = sym_frame(db, "trEff", years)
    keys = [c for c in eff.columns if c != "trEff"]
    if not set(keys) <= set(line.columns):
        raise ValueError(f"trEff is indexed by {keys}, not all of which are in lineExport")
    line = line.merge(eff, on=keys, how="left")
    require_mapped(line, ["trEff"], "lineExport x trEff")
    line = line.merge(hw, on="h", how="left")
    line["trLoss"] = (1 - line["trEff"]) * line["lineExport"] / line["hWeight"]
    return line.groupby(["t", "h", "EAEA"])["trLoss"].sum().rename_axis(["t", "h", "EA"])


def _demand_deh(db, line, hw, years=None):
    """Per-hour demand by consumption type cType (TWh per real hour), with
    distribution and transmission losses. Index (t, h, EA), cluster hours."""
    df = sym_frame(db, "DEh", years)
    # Demand behind a PV-battery connection is already netted in qClusterNet (PVES).
    if has_sym(db, "cIdx2Cluster"):
        cl = set(set_frame(db, "cIdx2Cluster")["cIdx"].astype(str))
        df = df[~df["cIdx"].astype(str).isin(cl)]
    df = (df.merge(mapping(db, "cIdx2cType", "cIdx"), on="cIdx", how="left")
            .merge(mapping(db, "cIdx2EA", "cIdx"), on="cIdx", how="left"))
    require_mapped(df, ["cType", "EA"], "DEh")
    df["cType"] = df["cType"].replace(CTYPE_RENAME)
    df = df.merge(hw, on="h", how="left")
    df["DEh"] = df["DEh"] / df["hWeight"] / 1000

    wide = df.groupby(["t", "h", "EA", "cType"])["DEh"].sum().unstack("cType").fillna(0)
    # Every priced area gets demand rows, also one without consumers (DK3), so its
    # transmission loss and CCS load are not lost.
    areas = sym_frame(db, "pE", years).set_index(["t", "h", "EA"]).index
    wide = wide.reindex(wide.index.union(areas), fill_value=0)
    wide.index.names = ["t", "h", "EA"]
    for col in CLAMP_DEMAND:
        if col in wide.columns:
            wide[col] = wide[col].clip(lower=0)

    # DEh is end-user demand; clearing grosses the DC-IIA classes up by 1/distrEff.
    dciia = [c for c in DCIIA_TYPES if c in wide.columns]
    if not dciia:
        raise ValueError(f"none of the DC-IIA classes {DCIIA_TYPES} found in DEh")
    eff = sym_frame(db, "distrEff", years).set_index(["t", "EA"])["distrEff"]
    eff = eff.reindex(wide.index.droplevel("h"))
    d = wide[dciia].sum(axis=1)
    if eff[d.values > 0].isna().any():
        raise ValueError("distrEff is missing for some (t, EA) with DC-IIA demand")
    wide["distrLoss"] = d * (1 / eff.values - 1)
    wide["distrLoss"] = wide["distrLoss"].fillna(0)

    wide["trLoss"] = _transmission_loss(db, line, hw, years).reindex(wide.index).fillna(0)

    # CCS auxiliary load (qDemandE_CCS, TWh over the cluster state) enters the
    # market balance next to plant and DC-IIA demand.
    if has_sym(db, "qDemandE_CCS"):
        ccs = sym_frame(db, "qDemandE_CCS", years).merge(hw, on="h", how="left")
        ccs["CCS"] = ccs["qDemandE_CCS"] / ccs["hWeight"]
        wide["CCS"] = ccs.set_index(["t", "h", "EA"])["CCS"].reindex(wide.index).fillna(0)
    wide.columns.name = None
    return wide


def _net_export(db, line, hw, h2hf, years=None):
    """Net export EA -> EAEA per real hour (TWh), with both prices attached."""
    line = line.groupby(["t", "h", "EA", "EAEA"])["lineExport"].sum()
    # Net = flow EA -> EAEA minus flow EAEA -> EA, both directions present (antisymmetric).
    rev = line.copy()
    rev.index = rev.index.rename(["t", "h", "EAEA", "EA"])
    rev = rev.reorder_levels(["t", "h", "EA", "EAEA"])
    nx = line.sub(rev, fill_value=0).rename("lineExport")

    pe = sym_frame(db, "pE", years)
    nx = nx.reset_index().merge(hw, on="h", how="left")
    nx = nx.merge(h2hf, on="h", how="left")
    nx["lineExport"] = nx["lineExport"] / nx["hWeight"]
    nx = nx.merge(pe.rename(columns={"pE": "pE_EA"}), on=["t", "h", "EA"], how="left")
    nx = nx.merge(pe.rename(columns={"EA": "EAEA", "pE": "pE_EAEA"}),
                  on=["t", "h", "EAEA"], how="left")
    return nx.set_index(["t", "hFull", "EA", "EAEA"]).sort_index().drop(columns=["h", "hWeight"])


def _assemble(df_sup, df_dem, meta, full_order, clamp_supply, split_cols=(), nx=None):
    """Mirrored supply (>= 0) / demand (<= 0) frames for one state. Raises if a
    non-zero series falls outside `full_order`: add it to groups.py, never drop it."""
    area = meta[1]
    dem_neg = -df_dem
    dem_zeros = df_dem * 0

    value_sup = [c for c in df_sup.columns if c not in meta]
    sup_pos = df_sup.copy()
    for col in tuple(clamp_supply) + tuple(split_cols):
        if col in sup_pos.columns:
            sup_pos[col] = sup_pos[col].clip(lower=0)
    sup_neg = df_sup.copy()
    sup_neg[value_sup] = 0
    for col in split_cols:                       # net injections: negative part is demand
        if col in df_sup.columns:
            sup_neg[col] = df_sup[col].clip(upper=0)

    # A storage type is both a plant (discharge, supply) and a consumption type
    # (charging, demand). Each side keeps its own value; the other side's zero
    # column is left out so the merge does not suffix it.
    keys = ["t", area, "h"]
    base = df_sup[meta]
    dem_zeros = dem_zeros.drop(columns=[c for c in dem_zeros.columns if c in sup_pos.columns])
    dem_cols = set(df_dem.columns)
    sup_neg = sup_neg[[c for c in sup_neg.columns if c in meta or c not in dem_cols]]
    supply = (base.merge(sup_pos, on=meta, how="left")
                  .merge(dem_zeros.reset_index(), on=keys, how="left"))
    demand = (base.merge(dem_neg.reset_index(), on=keys, how="left")
                  .merge(sup_neg, on=meta, how="left"))

    if nx is not None:
        nx_neg = nx.copy()
        nx_pos = nx.copy()
        nx_neg.loc[nx_neg["lineExport"] > 0, "lineExport"] = 0  # imports -> supply
        nx_pos.loc[nx_pos["lineExport"] < 0, "lineExport"] = 0  # exports -> demand
        nx_keys = ["t", "hFull", area]
        supply = supply.merge((-nx_neg["lineExport"].unstack("EAEA")).reset_index(),
                              on=nx_keys, how="left")
        demand = demand.merge((-nx_pos["lineExport"].unstack("EAEA")).reset_index(),
                              on=nx_keys, how="left")

    for frame in (supply, demand):
        value_cols = [c for c in frame.columns if c not in meta]
        frame[value_cols] = frame[value_cols].mul(1000).fillna(0)

    for name, frame in (("supply", supply), ("demand", demand)):
        extra = [c for c in frame.columns if c not in full_order]
        lost = {c: float(frame[c].abs().sum()) for c in extra if frame[c].abs().sum() > 0}
        if lost:
            raise ValueError(f"{name}: non-zero series not in the column order {lost}. "
                             "Add them to groups.py.")

    return (
        supply.reindex(columns=full_order, fill_value=0),
        demand.reindex(columns=full_order, fill_value=0),
    )


def _attach_baseline_price(state_dict, area_key, price):
    """Add the baseline hourly price to every state, for a shared hour ordering."""
    keys = ["t", area_key, "h", "hFull"]
    base = state_dict[STATE_BASE][keys + [price]].rename(columns={price: price + "_baseline"})
    for state in state_dict:
        state_dict[state] = state_dict[state].merge(base, on=keys, how="left")


def _attach_nx_baseline_price(NX_dict):
    """Attach the baseline area price (pE_EA_baseline) to every state's net-export frame."""
    base = (
        NX_dict[STATE_BASE].reset_index()[["t", "hFull", "EA", "pE_EA"]]
        .drop_duplicates(subset=["t", "hFull", "EA"])
        .rename(columns={"pE_EA": "pE_EA_baseline"})
    )
    for state in NX_dict:
        df = NX_dict[state].reset_index().merge(base, how="left", on=["t", "hFull", "EA"])
        NX_dict[state] = df.set_index(["t", "hFull", "EA", "EAEA"]).sort_index()


def build_el_balance_data(baseline_output, shock_output=None, *,
                          nuclear_ids=NUCLEAR_IDS, years=None) -> ElBalanceContext:
    """Electricity supply / demand / net-export frames per state, by area.
    `years` (int or iterable) restricts the build; None keeps every model year."""
    db_dict = _db_dict(baseline_output, shock_output)
    supply_dict, demand_dict, NX_dict = {}, {}, {}

    for state, db in db_dict.items():
        hw, h2hf = _hours(db)
        line = active_line_export(db, years)
        nx = _net_export(db, line, hw, h2hf, years)
        NX_dict[state] = nx
        supply_dict[state], demand_dict[state] = _assemble(
            _supply_seh(db, nuclear_ids, hw, h2hf, years), _demand_deh(db, line, hw, years),
            _META_BUILD, _FULL_ORDER, CLAMP_SUPPLY, SPLIT_COLS, nx=nx,
        )

    _attach_baseline_price(supply_dict, "EA", "pE")
    _attach_baseline_price(demand_dict, "EA", "pE")
    _attach_nx_baseline_price(NX_dict)
    return ElBalanceContext(supply_dict, demand_dict, NX_dict, db_dict)


# ==================== ==================== ==================== ====================
# 2. country aggregation: the Danish EAs collapsed into one "DK" area

def _country_price(db, members, years=None):
    """Per-(t, h) consumer-demand-weighted price across `members` (DKK/MWh), weighted
    on qDemandE_DCIIA so the annual mean equals pEt_C_DK. An area without consumers
    in a year (DK3) has no qDemandE_DCIIA rows and weight 0."""
    w = "qDemandE_DCIIA"
    p = sym_frame(db, "pE", years).merge(sym_frame(db, w, years), on=["t", "h", "EA"], how="left")
    p = p[p["EA"].isin(members)].copy()

    has_demand = p.groupby(["t", "EA"])[w].transform(lambda s: s.notna().any())
    if (has_demand & p[w].isna()).any():
        raise ValueError(f"{w} missing for some hours of an area that has demand that year")
    p[w] = p[w].fillna(0)

    p["wp"] = p["pE"] * p[w]
    g = p.groupby(["t", "h"])[["wp", w]].sum()             # same hWeight across zones
    if (g[w] <= 0).any():
        raise ValueError(f"no {w} in any of {members} for some (t, h)")
    return (g["wp"] / g[w]).rename("pE").reset_index()


def _collapse_supply(df_sup, members, price, meta, label):
    """Sum supply over `members` into one `label` row per (t, h, hFull); country price."""
    series = [c for c in df_sup.columns if c not in meta]
    sub = df_sup[df_sup["EA"].isin(members)]
    agg = sub.groupby(["t", "h", "hFull"], as_index=False).agg(
        {**{s: "sum" for s in series}, "hWeight": "first"}
    )
    agg["EA"] = label
    agg = agg.merge(price, on=["t", "h"], how="left")
    return agg.sort_values(["t", "pE"]).reset_index(drop=True)


def _collapse_demand(df_dem, members, label):
    """Sum demand over `members` into one `label` row (index t, h, EA)."""
    d = df_dem.reset_index()
    d = d[d["EA"].isin(members)]
    series = [c for c in d.columns if c not in ("t", "h", "EA")]
    agg = d.groupby(["t", "h"], as_index=False)[series].sum()
    agg["EA"] = label
    return agg.set_index(["t", "h", "EA"])


def _collapse_nx(nx, members, price_hf, label):
    """`members` -> `label` on the source side, internal trade dropped, summed per
    foreign partner, source price swapped for the country price."""
    df = nx.reset_index()
    df = df[df["EA"].isin(members) & ~df["EAEA"].isin(members)].copy()
    df["EA"] = label
    agg = df.groupby(["t", "hFull", "EA", "EAEA"], as_index=False).agg(
        {"lineExport": "sum", "pE_EAEA": "first"}
    )
    agg = agg.merge(price_hf[["t", "hFull", "pE"]].rename(columns={"pE": "pE_EA"}),
                    on=["t", "hFull"], how="left")
    return agg.set_index(["t", "hFull", "EA", "EAEA"]).sort_index()


def build_el_balance_data_country(baseline_output, shock_output=None, *,
                                  nuclear_ids=NUCLEAR_IDS, country_areas=c_aggs["DK"],
                                  country_label="DK", years=None) -> ElBalanceContext:
    """Denmark-wide counterpart of `build_el_balance_data` (DK1 + DK2 + DK3)."""
    country_areas = expand_areas(country_areas)
    db_dict = _db_dict(baseline_output, shock_output)
    supply_dict, demand_dict, NX_dict = {}, {}, {}

    for state, db in db_dict.items():
        hw, h2hf = _hours(db)
        line = active_line_export(db, years)
        price = _country_price(db, country_areas, years)
        price_hf = price.merge(h2hf, on="h", how="left")

        df_sup = _collapse_supply(_supply_seh(db, nuclear_ids, hw, h2hf, years),
                                  country_areas, price, _META_BUILD, country_label)
        df_dem = _collapse_demand(_demand_deh(db, line, hw, years), country_areas, country_label)
        nx = _collapse_nx(_net_export(db, line, hw, h2hf, years),
                          country_areas, price_hf, country_label)

        NX_dict[state] = nx
        sup, dem = _assemble(df_sup, df_dem, _META_BUILD, _FULL_ORDER,
                             CLAMP_SUPPLY, SPLIT_COLS, nx=nx)
        # _FULL_ORDER still carries the member areas as (now empty) partner columns.
        drop = [c for c in country_areas if c in sup.columns]
        supply_dict[state], demand_dict[state] = sup.drop(columns=drop), dem.drop(columns=drop)

    _attach_baseline_price(supply_dict, "EA", "pE")
    _attach_baseline_price(demand_dict, "EA", "pE")
    _attach_nx_baseline_price(NX_dict)
    return ElBalanceContext(supply_dict, demand_dict, NX_dict, db_dict)


# ==================== ==================== ==================== ====================
# 3. district-heat build (HA market): mirrors electricity, no inter-area trade

def _supply_shh(db, nuclear_ids, hw, h2hf, years=None):
    """Per-hour heat supply by production type Pt (TWh per real hour)."""
    df = sym_frame(db, "SHh", years)
    df = (df.merge(mapping(db, "id2Pt", "id"), on="id", how="left")
            .merge(mapping(db, "id2HA", "id"), on="id", how="left"))
    require_mapped(df, ["Pt", "HA"], "SHh")
    df["Pt"] = df["Pt"].replace(PT_RENAME)
    df = relabel_nuclear(df, nuclear_ids)

    wide = df.groupby(["t", "HA", "h", "Pt"])["SHh"].sum().unstack("Pt") / 1000  # not floored
    front = [c for c in ("HS",) if c in wide.columns]
    wide = wide[front + [c for c in wide.columns if c not in front]]
    wide.columns.name = None
    wide = wide.reset_index()

    wide = wide.merge(sym_frame(db, "pH", years), on=["t", "HA", "h"], how="left")
    wide = wide.merge(hw, on="h", how="left")
    wide = wide.merge(h2hf, on="h", how="left")

    keep = {"t", "HA", "h", "pH", "hWeight", "hFull"}
    for col in [c for c in wide.columns if c not in keep]:
        wide[col] = wide[col] / wide["hWeight"]

    wide = wide.sort_values(["t", "HA", "pH"])
    return wide.fillna(0)


def _demand_dhh(db, hw, years=None):
    """Per-hour heat demand by consumption type cType (TWh per real hour)."""
    df = sym_frame(db, "DHh", years)
    df = (df.merge(mapping(db, "cIdx2cType", "cIdx"), on="cIdx", how="left")
            .merge(mapping(db, "cIdx2HA", "cIdx"), on="cIdx", how="left"))
    require_mapped(df, ["cType", "HA"], "DHh")
    df = df.merge(hw, on="h", how="left")
    df["DHh"] = df["DHh"] / df["hWeight"] / 1000

    wide = df.groupby(["t", "h", "HA", "cType"])["DHh"].sum().unstack("cType").fillna(0)
    for col in CLAMP_DEMAND_HA:
        if col in wide.columns:
            wide[col] = wide[col].clip(lower=0)

    # CCS heat load (qDemandH_CCS, TWh over the cluster state) enters the heat balance.
    if has_sym(db, "qDemandH_CCS"):
        ccs = sym_frame(db, "qDemandH_CCS", years).merge(hw, on="h", how="left")
        ccs["CCS"] = ccs["qDemandH_CCS"] / ccs["hWeight"]
        ccs = ccs.set_index(["t", "h", "HA"])["CCS"]
        wide = wide.reindex(wide.index.union(ccs.index), fill_value=0)
        wide.index.names = ["t", "h", "HA"]
        wide["CCS"] = ccs.reindex(wide.index).fillna(0)
    wide.columns.name = None
    return wide


def build_dh_balance_data(baseline_output, shock_output=None, *,
                          nuclear_ids=NUCLEAR_IDS, years=None) -> ElBalanceContext:
    """District-heat counterpart of `build_el_balance_data` (NX_dict is None)."""
    db_dict = _db_dict(baseline_output, shock_output)
    supply_dict, demand_dict = {}, {}

    for state, db in db_dict.items():
        hw, h2hf = _hours(db)
        supply_dict[state], demand_dict[state] = _assemble(
            _supply_shh(db, nuclear_ids, hw, h2hf, years), _demand_dhh(db, hw, years),
            _META_BUILD_HA, _FULL_ORDER_HA, CLAMP_SUPPLY_HA, nx=None,
        )

    _attach_baseline_price(supply_dict, "HA", "pH")
    _attach_baseline_price(demand_dict, "HA", "pH")
    return ElBalanceContext(supply_dict, demand_dict, None, db_dict)
