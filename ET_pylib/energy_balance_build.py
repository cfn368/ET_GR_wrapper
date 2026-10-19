"""Intra-yearly electricity and district-heat balance: data build (KF26).

Produces the intermediate format the plot and summary layers read:
``supply_dict`` / ``demand_dict`` (one row per t, area, hFull; supply >= 0,
demand <= 0, GWh per real hour) and ``NX_dict`` (index t, hFull, EA, EAEA;
net export in TWh per real hour, positive = export from EA).

Changes from KF25, all in the build: explicit merges instead of wildcard
``id2*`` / ``cIdx2*`` joins; DEh/DHh carry the area through ``cIdx2EA`` /
``cIdx2HA``; KF26 renames (V2G plant, transport demand); PV-battery clusters
as one net injection (PVES); distribution loss D*(1/distrEff - 1) on the DC-IIA
classes; transmission loss on the receiving area; nuclear by explicit ids and
start year; and a hard error if a non-zero series would fall outside the column
order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .groups import (
    _META_BUILD, _FULL_ORDER, _META_BUILD_HA, _FULL_ORDER_HA,
    CLAMP_SUPPLY, CLAMP_DEMAND, SPLIT_COLS, CLAMP_SUPPLY_HA, CLAMP_DEMAND_HA,
    PT_RENAME, CTYPE_RENAME, DCIIA_TYPES, NUCLEAR_PT, NUCLEAR_IDS, c_aggs,
)


# =========================================================================== #
# 0. year filtering
# =========================================================================== #
def _year_set(years):
    """Normalise the `years` argument to a set of ints, or None for 'all'."""
    if years is None:
        return None
    if isinstance(years, (int, str)):
        years = [years]
    return {int(y) for y in years}


def _fy(obj, years, *, what="frame"):
    """Keep only `years`. Handles a `t` column or a `t` index level; no-op when
    `years is None` or the object carries no `t`. Raises loudly on an empty cut,
    because `t` is int64 and a string year would otherwise match nothing."""
    yrs = _year_set(years)
    if yrs is None:
        return obj
    cols = getattr(obj, "columns", [])
    if "t" in list(cols):
        have = obj["t"].astype("int64")
    elif "t" in list(obj.index.names or []):
        have = obj.index.get_level_values("t").astype("int64")
    else:
        return obj
    out = obj[have.isin(yrs)]
    if len(out) == 0:
        raise ValueError(
            f"years={sorted(yrs)} matched no rows in {what}; t holds {sorted(set(have))}"
        )
    return out


# =========================================================================== #
# 1. small helpers
# =========================================================================== #
def _symbols(db):
    return set(map(str, db.symbols.keys()))


def _series(db, sym, years=None):
    """A parameter/variable as a flat frame with its value column named `sym`."""
    return _fy(db(sym).rename(sym).reset_index(), years, what=sym)


def _map(db, sym, key):
    """A mapping set as a frame, required to be one-to-one on `key` so a merge
    on it can never duplicate rows."""
    df = db(sym).to_frame()[[]].reset_index()
    dup = df[key].duplicated()
    if dup.any():
        raise ValueError(f"{sym} maps some {key} to more than one value: "
                         f"{sorted(df.loc[dup, key].astype(str).unique())[:10]}")
    return df


def _require_mapped(df, cols, what):
    """Raise if a merge left some rows without their mapping."""
    for c in cols:
        if df[c].isna().any():
            raise ValueError(f"{what}: {int(df[c].isna().sum())} rows without {c} after the merge")


def _relabel_nuclear(df, nuclear_ids):
    """Pt = Nuclear on the reused ids from their first nuclear year on."""
    if not nuclear_ids:
        return df
    start = df["id"].astype(str).map(nuclear_ids)
    mask = start.notna() & (df["t"].astype("int64") >= start.fillna(0).astype("int64"))
    df.loc[mask, "Pt"] = NUCLEAR_PT
    return df


relabel_nuclear = _relabel_nuclear      # public: the one nuclear rule for every module


def cluster_members(db):
    """Plant ids behind a shared PV-battery connection (cleared as one net injection)."""
    if "id2Cluster" not in _symbols(db):
        return set()
    return set(db("id2Cluster").to_frame()[[]].reset_index()["id"].astype(str))


def _cluster_net(db, years=None):
    """Net cluster injection by (t, EA, h) in TWh over the cluster state (the same
    unit as SEh / 1000). Positive is an export from the cluster into the area."""
    need = {"qClusterNet", "cluster2EA"}
    missing = need - _symbols(db)
    if missing:
        raise KeyError(f"PV-battery clusters exist (id2Cluster) but {sorted(missing)} is missing")
    net = _fy(db("qClusterNet").rename("PVES").reset_index(), years, what="qClusterNet")
    net = net.merge(_map(db, "cluster2EA", "cluster"), on="cluster", how="left")
    _require_mapped(net, ["EA"], "qClusterNet")
    return net.groupby(["t", "EA", "h"], as_index=False)["PVES"].sum()


# =========================================================================== #
# 2. electricity build
# =========================================================================== #
def _supply_seh(db, nuclear_ids, years=None):
    """Per-hour supply by production type ``Pt`` (TWh per real hour), nuclear split out."""
    df = _series(db, "SEh", years)
    df = (df.merge(_map(db, "id2Pt", "id"), on="id", how="left")
            .merge(_map(db, "id2EA", "id"), on="id", how="left"))
    _require_mapped(df, ["Pt", "EA"], "SEh")
    df["Pt"] = df["Pt"].replace(PT_RENAME)
    df = _relabel_nuclear(df, nuclear_ids)

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

    wide = wide.merge(_series(db, "pE", years), on=["t", "EA", "h"], how="left")
    wide = wide.merge(_series(db, "hWeight"), on="h", how="left")
    wide = wide.merge(db("h2hFull").to_frame()[[]].reset_index(), on="h", how="left")

    keep = {"t", "EA", "h", "pE", "hWeight", "hFull"}
    for col in [c for c in wide.columns if c not in keep]:
        wide[col] = wide[col] / wide["hWeight"]

    wide = wide.sort_values(["t", "EA", "pE"])
    return wide.fillna(0)


def _active_line_export(db, years=None):
    """lineExport restricted to the rows the market clears on: activeLineEA[t,line,EA,EAEA]
    with both areas active (F_ExportEA_h / F_ImportEA_h in the model). lineExport also
    carries levels on inactive combinations, e.g. DK2<->DELU on the Bornholm line DELU__DK3,
    which no balance counts."""
    missing = {"activeLineEA", "activeEA"} - _symbols(db)
    if missing:
        raise KeyError(f"{sorted(missing)} is missing: cannot tell active lines from inactive")
    line = _series(db, "lineExport", years)
    act = _fy(db("activeLineEA").to_frame()[[]].reset_index(), years, what="activeLineEA")
    ea = _fy(db("activeEA").to_frame()[[]].reset_index(), years, what="activeEA")
    act = (act.merge(ea, on=["t", "EA"])
              .merge(ea.rename(columns={"EA": "EAEA"}), on=["t", "EAEA"]))
    return line.merge(act, on=["t", "line", "EA", "EAEA"], how="inner")


active_line_export = _active_line_export   # public: congestion income uses the same rows


def _transmission_loss(db, years=None):
    """Energy lost on the interconnectors, booked on the receiving area
    (index t, h, EA; TWh per real hour, the same scale as the demand series)."""
    if "trEff" not in _symbols(db):
        raise KeyError("trEff is missing: cannot compute the transmission loss")
    line = _active_line_export(db, years)
    eff = _series(db, "trEff", years)
    keys = [c for c in eff.columns if c != "trEff"]
    if not set(keys) <= set(line.columns):
        raise ValueError(f"trEff is indexed by {keys}, not all of which are in lineExport")
    line = line.merge(eff, on=keys, how="left")
    _require_mapped(line, ["trEff"], "lineExport x trEff")
    line = line.merge(_series(db, "hWeight"), on="h", how="left")
    line["trLoss"] = (1 - line["trEff"]) * line["lineExport"] / line["hWeight"]
    return line.groupby(["t", "h", "EAEA"])["trLoss"].sum().rename_axis(["t", "h", "EA"])


def _demand_deh(db, years=None):
    """Per-hour demand by consumption type ``cType`` (TWh per real hour), with
    distribution and transmission losses. Index (t, h, EA), cluster hours."""
    df = _series(db, "DEh", years)
    # Demand behind a PV-battery connection is already netted in qClusterNet (PVES).
    if "cIdx2Cluster" in _symbols(db):
        cl = set(db("cIdx2Cluster").to_frame()[[]].reset_index()["cIdx"].astype(str))
        df = df[~df["cIdx"].astype(str).isin(cl)]
    df = (df.merge(_map(db, "cIdx2cType", "cIdx"), on="cIdx", how="left")
            .merge(_map(db, "cIdx2EA", "cIdx"), on="cIdx", how="left"))
    _require_mapped(df, ["cType", "EA"], "DEh")
    df["cType"] = df["cType"].replace(CTYPE_RENAME)
    df = df.merge(_series(db, "hWeight"), on="h", how="left")
    df["DEh"] = df["DEh"] / df["hWeight"] / 1000

    wide = df.groupby(["t", "h", "EA", "cType"])["DEh"].sum().unstack("cType").fillna(0)
    # Every priced area gets demand rows, also one without consumers (DK3), so its
    # transmission loss and CCS load are not lost.
    areas = _series(db, "pE", years).set_index(["t", "h", "EA"]).index
    wide = wide.reindex(wide.index.union(areas), fill_value=0)
    wide.index.names = ["t", "h", "EA"]
    for col in CLAMP_DEMAND:
        if col in wide.columns:
            wide[col] = wide[col].clip(lower=0)

    # DEh is end-user demand; clearing grosses the DC-IIA classes up by 1/distrEff.
    dciia = [c for c in DCIIA_TYPES if c in wide.columns]
    if not dciia:
        raise ValueError(f"none of the DC-IIA classes {DCIIA_TYPES} found in DEh")
    eff = _series(db, "distrEff", years).set_index(["t", "EA"])["distrEff"]
    eff = eff.reindex(wide.index.droplevel("h"))
    d = wide[dciia].sum(axis=1)
    if eff[d.values > 0].isna().any():
        raise ValueError("distrEff is missing for some (t, EA) with DC-IIA demand")
    wide["distrLoss"] = d * (1 / eff.values - 1)
    wide["distrLoss"] = wide["distrLoss"].fillna(0)

    wide["trLoss"] = _transmission_loss(db, years).reindex(wide.index).fillna(0)

    # CCS auxiliary load (qDemandE_CCS, TWh over the cluster state) enters the
    # market balance next to plant and DC-IIA demand.
    if "qDemandE_CCS" in _symbols(db):
        ccs = _series(db, "qDemandE_CCS", years).merge(_series(db, "hWeight"), on="h", how="left")
        ccs["CCS"] = ccs["qDemandE_CCS"] / ccs["hWeight"]
        wide["CCS"] = ccs.set_index(["t", "h", "EA"])["CCS"].reindex(wide.index).fillna(0)
    wide.columns.name = None
    return wide


def _net_export(db, years=None):
    """Net export EA -> EAEA per real hour (TWh), with both prices attached."""
    line = _active_line_export(db, years).groupby(["t", "h", "EA", "EAEA"])["lineExport"].sum()
    # Net = flow EA -> EAEA minus flow EAEA -> EA, both directions present (antisymmetric).
    rev = line.copy()
    rev.index = rev.index.rename(["t", "h", "EAEA", "EA"])
    rev = rev.reorder_levels(["t", "h", "EA", "EAEA"])
    nx = line.sub(rev, fill_value=0).rename("lineExport")

    pe = _series(db, "pE", years)
    nx = nx.reset_index().merge(_series(db, "hWeight"), on="h", how="left")
    nx = nx.merge(db("h2hFull").to_frame()[[]].reset_index(), on="h", how="left")
    nx["lineExport"] = nx["lineExport"] / nx["hWeight"]
    nx = nx.merge(pe.rename(columns={"pE": "pE_EA"}), on=["t", "h", "EA"], how="left")
    nx = nx.merge(pe.rename(columns={"EA": "EAEA", "pE": "pE_EAEA"}),
                  on=["t", "h", "EAEA"], how="left")
    nx = nx.set_index(["t", "hFull", "EA", "EAEA"]).sort_index().drop(columns=["h", "hWeight"])
    return nx


def _assemble(df_sup, df_dem, meta, full_order, clamp_supply, split_cols=(), nx=None):
    """Build the mirrored supply (>=0) / demand (<=0) frames for one state.

    Raises if a non-zero series would fall outside `full_order`: a new plant or
    consumption type, or an area, must be added to groups.py, not dropped."""
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

    # A storage type (ES, HYS) is both a plant (discharge, supply) and a consumption
    # type (charging, demand). On each side the side's own value wins; the other
    # side's zero column is left out so the merge does not suffix it.
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
    """Add the baseline hourly price to every state (for comparable sorting)."""
    keys = ["t", area_key, "h", "hFull"]
    base = state_dict["baseline"][keys + [price]].rename(columns={price: price + "_baseline"})
    for state in state_dict:
        state_dict[state] = state_dict[state].merge(base, on=keys, how="left")


def _attach_nx_baseline_price(NX_dict):
    """Attach the baseline area price (`pE_EA_baseline`) to every state's net-export
    frame, so net-export panels can share one hour ordering across states."""
    base = (
        NX_dict["baseline"].reset_index()[["t", "hFull", "EA", "pE_EA"]]
        .drop_duplicates(subset=["t", "hFull", "EA"])
        .rename(columns={"pE_EA": "pE_EA_baseline"})
    )
    for state in NX_dict:
        df = NX_dict[state].reset_index().merge(base, how="left", on=["t", "hFull", "EA"])
        NX_dict[state] = df.set_index(["t", "hFull", "EA", "EAEA"]).sort_index()


@dataclass
class ElBalanceData:
    """The dicts produced by :func:`build_el_balance_data`."""

    db_dict: Mapping[str, object]
    supply_dict: Mapping[str, pd.DataFrame]
    demand_dict: Mapping[str, pd.DataFrame]
    NX_dict: Mapping[str, pd.DataFrame]

    def save(self, path) -> None:
        """Pickle the three DataFrame dicts (not db_dict -- GpyDB doesn't pickle)."""
        save_data(self.supply_dict, self.demand_dict, self.NX_dict, path)


def save_data(supply_dict, demand_dict, NX_dict, path) -> None:
    """Persist the plottable dicts so a kernel restart skips the slow build."""
    import pickle

    payload = {"supply_dict": supply_dict, "demand_dict": demand_dict, "NX_dict": NX_dict}
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def load_data(path) -> dict:
    """Reload dicts saved by :func:`save_data`. Returns a dict with keys
    ``supply_dict`` / ``demand_dict`` / ``NX_dict``."""
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)


def _db_dict(baseline_output, shock_dict, shock_step):
    """{"baseline": db} or {"baseline": db, "shock": db}. No .copy(): the build only
    reads, and an unpickled GpyDB has no live GAMS workspace."""
    db_dict = {"baseline": baseline_output}
    if shock_dict is not None:
        db_dict["shock"] = shock_dict[shock_step] if shock_step is not None else shock_dict
    return db_dict


def build_el_balance_data(
    baseline_output,
    shock_dict=None,
    shock_step=None,
    *,
    nuclear_ids=NUCLEAR_IDS,
    years=None,
) -> ElBalanceData:
    """Build supply/demand/net-export frames from raw model output.

    `shock_dict` may be a dict of shock steps (with `shock_step`), a single
    database (with `shock_step=None`), or None for the baseline alone.
    `years` (int or iterable) restricts the build; None keeps every model year."""
    db_dict = _db_dict(baseline_output, shock_dict, shock_step)
    supply_dict, demand_dict, NX_dict = {}, {}, {}

    for state, db in db_dict.items():
        df_SEh = _supply_seh(db, nuclear_ids, years=years)
        df_DEh = _demand_deh(db, years=years)
        nx = _net_export(db, years=years)

        NX_dict[state] = nx
        supply_dict[state], demand_dict[state] = _assemble(
            df_SEh, df_DEh, _META_BUILD, _FULL_ORDER, CLAMP_SUPPLY, SPLIT_COLS, nx=nx
        )

    _attach_baseline_price(supply_dict, "EA", "pE")
    _attach_baseline_price(demand_dict, "EA", "pE")
    _attach_nx_baseline_price(NX_dict)

    return ElBalanceData(db_dict, supply_dict, demand_dict, NX_dict)


# --------------------------------------------------------------------------- #
# 3. heat build (HA market) -- mirrors electricity, no inter-area trade
# --------------------------------------------------------------------------- #
def _supply_shh(db, nuclear_ids, years=None):
    """Per-hour heat supply by production type ``Pt`` (TWh per real hour)."""
    df = _series(db, "SHh", years)
    df = (df.merge(_map(db, "id2Pt", "id"), on="id", how="left")
            .merge(_map(db, "id2HA", "id"), on="id", how="left"))
    _require_mapped(df, ["Pt", "HA"], "SHh")
    df["Pt"] = df["Pt"].replace(PT_RENAME)
    df = _relabel_nuclear(df, nuclear_ids)

    wide = df.groupby(["t", "HA", "h", "Pt"])["SHh"].sum().unstack("Pt") / 1000  # NB: not floored
    front = [c for c in ("HS",) if c in wide.columns]
    wide = wide[front + [c for c in wide.columns if c not in front]]
    wide.columns.name = None
    wide = wide.reset_index()

    wide = wide.merge(_series(db, "pH", years), on=["t", "HA", "h"], how="left")
    wide = wide.merge(_series(db, "hWeight"), on="h", how="left")
    wide = wide.merge(db("h2hFull").to_frame()[[]].reset_index(), on="h", how="left")

    keep = {"t", "HA", "h", "pH", "hWeight", "hFull"}
    for col in [c for c in wide.columns if c not in keep]:
        wide[col] = wide[col] / wide["hWeight"]

    wide = wide.sort_values(["t", "HA", "pH"])  # merit-order x-axis (heat price)
    return wide.fillna(0)


def _demand_dhh(db, years=None):
    """Per-hour heat demand by consumption type ``cType`` (TWh per real hour)."""
    df = _series(db, "DHh", years)
    df = (df.merge(_map(db, "cIdx2cType", "cIdx"), on="cIdx", how="left")
            .merge(_map(db, "cIdx2HA", "cIdx"), on="cIdx", how="left"))
    _require_mapped(df, ["cType", "HA"], "DHh")
    df = df.merge(_series(db, "hWeight"), on="h", how="left")
    df["DHh"] = df["DHh"] / df["hWeight"] / 1000

    wide = df.groupby(["t", "h", "HA", "cType"])["DHh"].sum().unstack("cType").fillna(0)
    for col in CLAMP_DEMAND_HA:
        if col in wide.columns:
            wide[col] = wide[col].clip(lower=0)

    # CCS heat load (qDemandH_CCS, TWh over the cluster state) enters the heat balance.
    if "qDemandH_CCS" in _symbols(db):
        ccs = _series(db, "qDemandH_CCS", years).merge(_series(db, "hWeight"), on="h", how="left")
        ccs["CCS"] = ccs["qDemandH_CCS"] / ccs["hWeight"]
        ccs = ccs.set_index(["t", "h", "HA"])["CCS"]
        wide = wide.reindex(wide.index.union(ccs.index), fill_value=0)
        wide.index.names = ["t", "h", "HA"]
        wide["CCS"] = ccs.reindex(wide.index).fillna(0)
    wide.columns.name = None
    return wide


def build_dh_balance_data(
    baseline_output, shock_dict=None, shock_step=None, *,
    nuclear_ids=NUCLEAR_IDS, years=None,
):
    """District-heating counterpart of :func:`build_el_balance_data`"""
    db_dict = _db_dict(baseline_output, shock_dict, shock_step)
    supply_dict, demand_dict = {}, {}

    for state, db in db_dict.items():
        df_SHh = _supply_shh(db, nuclear_ids, years=years)
        df_DHh = _demand_dhh(db, years=years)
        supply_dict[state], demand_dict[state] = _assemble(
            df_SHh, df_DHh, _META_BUILD_HA, _FULL_ORDER_HA, CLAMP_SUPPLY_HA, nx=None
        )

    _attach_baseline_price(supply_dict, "HA", "pH")
    _attach_baseline_price(demand_dict, "HA", "pH")

    return ElBalanceData(db_dict, supply_dict, demand_dict, None)


# =========================================================================== #
# 4. country aggregation -- collapse the Danish EAs into one "DK" area
# =========================================================================== #
def _country_price(db, members, years=None):
    """Per-(t,h) consumer-demand-weighted electricity price across `members` (DKK/MWh).
    Weights on qDemandE_DCIIA so the annual DCIIA-weighted mean equals pEt_C_DK.
    An area with no consumers in a year (DK3) has no qDemandE_DCIIA rows and weight 0."""
    w = "qDemandE_DCIIA"
    p = _series(db, "pE", years).merge(_series(db, w, years), on=["t", "h", "EA"], how="left")
    p = p[p["EA"].isin(members)].copy()

    has_demand = p.groupby(["t", "EA"])[w].transform(lambda s: s.notna().any())
    partial = has_demand & p[w].isna()
    if partial.any():                                       # fail loudly, no silent drop
        raise ValueError(f"{w} missing for some hours of an area that has demand that year")
    p[w] = p[w].fillna(0)

    p["wp"] = p["pE"] * p[w]
    g = p.groupby(["t", "h"])[["wp", w]].sum()             # same hWeight across zones
    if (g[w] <= 0).any():
        raise ValueError(f"no {w} in any of {members} for some (t, h)")
    return (g["wp"] / g[w]).rename("pE").reset_index()     # cols: t, h, pE


def _collapse_supply(df_sup, members, price, meta, label):
    """Sum supply series over `members` -> one `label` row; swap in the country price.
    Groups on (t,h,hFull) because `_supply_seh` is expanded to hFull."""
    series = [c for c in df_sup.columns if c not in meta]
    sub = df_sup[df_sup["EA"].isin(members)]
    agg = sub.groupby(["t", "h", "hFull"], as_index=False).agg(
        {**{s: "sum" for s in series}, "hWeight": "first"}
    )
    agg["EA"] = label
    agg = agg.merge(price, on=["t", "h"], how="left")   # price unique per (t,h) -> 1:1
    return agg.sort_values(["t", "pE"]).reset_index(drop=True)


def _collapse_demand(df_dem, members, label):
    """Sum demand series over `members` -> one `label` row (index t,h,EA)."""
    d = df_dem.reset_index()
    d = d[d["EA"].isin(members)]
    series = [c for c in d.columns if c not in ("t", "h", "EA")]
    agg = d.groupby(["t", "h"], as_index=False)[series].sum()
    agg["EA"] = label
    return agg.set_index(["t", "h", "EA"])


def _collapse_nx(nx, members, price_hf, label):
    """Collapse `members` -> `label` on the source side, drop internal trade,
    sum to each foreign partner, swap the source price for the country price"""
    df = nx.reset_index()
    df = df[df["EA"].isin(members) & ~df["EAEA"].isin(members)].copy()
    df["EA"] = label
    agg = df.groupby(["t", "hFull", "EA", "EAEA"], as_index=False).agg(
        {"lineExport": "sum", "pE_EAEA": "first"}
    )
    agg = agg.merge(
        price_hf[["t", "hFull", "pE"]].rename(columns={"pE": "pE_EA"}),
        on=["t", "hFull"], how="left",          # unique per (t,hFull) -> 1:1
    )
    return agg.set_index(["t", "hFull", "EA", "EAEA"]).sort_index()


def build_el_balance_data_country(
    baseline_output, shock_dict=None, shock_step=None, *,
    nuclear_ids=NUCLEAR_IDS,
    country_areas: tuple = c_aggs["DK"],
    country_label: str = "DK",
    years=None,
) -> ElBalanceData:
    """Denmark-wide counterpart of :func:`build_el_balance_data` (DK1 + DK2 + DK3)."""
    db_dict = _db_dict(baseline_output, shock_dict, shock_step)
    supply_dict, demand_dict, NX_dict = {}, {}, {}

    for state, db in db_dict.items():
        price = _country_price(db, country_areas, years=years)          # t,h,pE
        price_hf = price.merge(                                         # t,hFull,pE (unique)
            db("h2hFull").to_frame()[[]].reset_index(), on="h", how="left"
        )

        df_SEh = _collapse_supply(
            _supply_seh(db, nuclear_ids, years=years),
            country_areas, price, _META_BUILD, country_label,
        )
        df_DEh = _collapse_demand(_demand_deh(db, years=years), country_areas, country_label)
        nx = _collapse_nx(
            _net_export(db, years=years), country_areas, price_hf, country_label
        )

        NX_dict[state] = nx
        sup, dem = _assemble(df_SEh, df_DEh, _META_BUILD, _FULL_ORDER,
                             CLAMP_SUPPLY, SPLIT_COLS, nx=nx)
        # _FULL_ORDER still carries the member areas as (now-empty) partner columns; drop them.
        drop = [c for c in country_areas if c in sup.columns]
        supply_dict[state], demand_dict[state] = sup.drop(columns=drop), dem.drop(columns=drop)

    _attach_baseline_price(supply_dict, "EA", "pE")
    _attach_baseline_price(demand_dict, "EA", "pE")
    _attach_nx_baseline_price(NX_dict)
    return ElBalanceData(db_dict, supply_dict, demand_dict, NX_dict)