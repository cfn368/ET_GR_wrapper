"""
Balance figures — hour-sorted energy balance and net export, baseline vs. shock
================================================================================

Reads the hFull-expanded frames of an `ElBalanceContext` and draws 1x2 figures
(baseline | shock): the electricity or district-heat balance as stacked areas
with the price on a twin axis, net export by partner, and net export attributed
to supply components (plant types plus transit bands).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .groups import (
    as_list,
    META_COLS, META_COLS_HA, COL_ORDER, PT_COLS, EA_ORDER, EA_cols, NUCLEAR_PT,
    STATE_BASE, STATES, c_aggs_countries, LEGEND_GROUPS_EL, LEGEND_GROUPS_NX,
)
from .style import (
    FIG_TEXT, STATE_LABELS, LEGEND_MAX_COLS, color_of, label_of, col_names, print_table,
)
from .plot_base import (
    new_figure, frame_panels, bottom_legend, grouped_legend, layout_pair, save_show,
    area_panel, price_twin,
)

TITLE_X = 0.124


# ==================== ==================== ==================== ====================
# 0. market definitions and helpers

@dataclass(frozen=True)
class _Market:
    """What separates the electricity balance from the district-heat balance."""
    area_key: str        # "EA" / "HA"
    price: str           # "pE" / "pH"
    price_base: str      # "pE_baseline" / "pH_baseline"
    price_label: str
    meta_cols: tuple     # columns that are not plottable series
    has_trade: bool
    title_key: str
    legend_groups: tuple | None    # (group key, members) per legend block; None = one block


EL_MARKET = _Market("EA", "pE", "pE_baseline", FIG_TEXT["ylabel_price_el"],
                    tuple(META_COLS), True, "title_balance_el",
                    LEGEND_GROUPS_EL)
DH_MARKET = _Market("HA", "pH", "pH_baseline", FIG_TEXT["ylabel_price_dh"],
                    tuple(META_COLS_HA), False, "title_balance_dh",
                    None)


def _years(years, values):
    """`years` as a list; None -> every year in `values`."""
    if years is None:
        return sorted(pd.unique(values))
    return [years] if isinstance(years, int) else list(years)


def _for_area(df, area, years, area_key):
    return df[(df[area_key] == area) & (df["t"].isin(years))]


def _value_cols(df, meta):
    return [c for c in df.columns if c not in meta]


def _connected_eas(NX_dict, ea):
    nx = NX_dict[STATE_BASE].xs(ea, level="EA").groupby("EAEA")[["lineExport"]].sum()
    return sorted(nx.index[nx["lineExport"] != 0].tolist())


def _stacked_extent(data_dict, area, years, agg, market, margin=1.05):
    extremes = []
    for state in STATES:
        sub = _for_area(data_dict[state], area, years, market.area_key)
        totals = sub[_value_cols(sub, market.meta_cols)].sum(axis=1)
        if not totals.empty:
            extremes.append(getattr(totals, agg)())
    if not extremes:
        return 0.0
    return (min(extremes) if agg == "min" else max(extremes)) * margin


def _price_extent(prices, margin=1.05):
    """(min incl. 0, max) of the price series across states, with margin."""
    prices = [p for p in prices if len(p)]
    if not prices:
        return 0.0, 1.0
    allp = pd.concat(prices)
    return min(allp.min(), 0.0) * margin, allp.max() * margin


def _annual_twh(frames, series):
    """Largest annual energy (TWh) per series across the given frames (GWh per hFull)."""
    totals = {s: 0.0 for s in series}
    for sub in frames:
        for s in series:
            if s in sub.columns:
                totals[s] = max(totals[s], abs(sub[s].sum()) / 1000.0)
    return totals


def _legend_groups(groups, entries):
    """(header, handles, labels) per legend group for the (series, handle) `entries`;
    raises on a series that belongs to no group."""
    stray = [s for s, _ in entries if not any(s in members for _, members in groups)]
    if stray:
        raise KeyError(f"series without a legend group: {stray}")
    return [(FIG_TEXT[f"legend_group_{key}"],
             [h for s, h in entries if s in members],
             [label_of(s) for s, _ in entries if s in members])
            for key, members in groups]


def _finish_pair(fig, axes, entries, *, groups, legend_y, legend_max_cols, title, save_path,
                 show):
    """Shared tail of the 1x2 figures: panel frames, layout, title, legend, save, show.
    `entries` are (series, handle); `groups` splits them under headers, None keeps one block
    (which needs an explicit `legend_y`)."""
    frame_panels(axes)
    layout_pair(fig, axes, title=title, x=TITLE_X)
    if entries and groups is not None:
        grouped_legend(fig, _legend_groups(groups, entries), legend_y=legend_y,
                       max_cols=legend_max_cols)
    elif entries:
        bottom_legend(fig, [h for _, h in entries], [label_of(s) for s, _ in entries],
                      legend_y=legend_y, ncol=min(len(entries), 3))
    save_show(fig, save_path=save_path, show=show)


# ==================== ==================== ==================== ====================
# 1. energy balance (electricity and district heat)

def _plot_balance(ctx, area, *, market, hatched, years, sort_by, legend_min_twh, figsize,
                  save_path_fmt, show, verbose, legend_y, legend_max_cols=LEGEND_MAX_COLS,
                  only=None, align_zero=False, state_labels=None):
    """Baseline | shock balance for one area and market."""
    supply_dict, demand_dict = ctx.supply_dict, ctx.demand_dict
    state_labels = state_labels or STATE_LABELS
    years = _years(years, supply_dict[STATE_BASE]["t"])
    price_lims = _price_extent([_for_area(supply_dict[s], area, years, market.area_key)[market.price]
                                for s in STATES])

    if market.has_trade and ctx.NX_dict is not None:
        connected = _connected_eas(ctx.NX_dict, area)
        idx = ctx.NX_dict[STATE_BASE].index
        all_eas = sorted(set(idx.get_level_values("EA")) | set(idx.get_level_values("EAEA")))
        drop = [e for e in all_eas if e not in connected]
        prep = lambda df: (lambda d: d.drop(columns=[c for c in drop if c in d.columns]))(
            _for_area(df, area, years, market.area_key))
    else:
        prep = lambda df: _for_area(df, area, years, market.area_key)

    supply = {s: prep(supply_dict[s]) for s in STATES}
    demand = {s: prep(demand_dict[s]) for s in STATES}

    # Only series with content for the area in at least one state.
    series = [s for s in _value_cols(supply[STATE_BASE], market.meta_cols)
              if any(supply[st][s].abs().sum() + demand[st][s].abs().sum() > 0 for st in STATES)]
    if only is not None:
        series = [s for s in series if s in only]
    if NUCLEAR_PT in series:                        # nuclear at the bottom of the stack
        series = [NUCLEAR_PT, *(s for s in series if s != NUCLEAR_PT)]
    colors = [color_of(c) for c in series]

    if only is not None:
        y_lims = (0.0, max(supply[s][series].sum(axis=1).max() for s in STATES) * 1.05)
    else:
        y_lims = (_stacked_extent(demand_dict, area, years, "min", market),
                  _stacked_extent(supply_dict, area, years, "max", market))

    l0, l1 = y_lims
    if align_zero and l0 < 0 < l1 and price_lims[1] > 0:
        f = (0 - l0) / (l1 - l0)
        price_ylim = (f * price_lims[1] / (f - 1), price_lims[1]) if f < 0.999 else price_lims
    else:
        price_ylim = price_lims

    results = []
    for year in years:
        fig, axes = new_figure(1, 2, figsize=figsize, sharey=True)
        handles = None
        order = ["hFull"] if sort_by == "hFull" else [sort_by, "hFull"]
        for col, (ax, state) in enumerate(zip(axes, STATES)):
            sup_t = supply[state][supply[state]["t"] == year].sort_values(order).reset_index(drop=True)
            dem_t = demand[state][demand[state]["t"] == year].sort_values(order).reset_index(drop=True)
            h = area_panel(ax, sup_t, dem_t, series, colors, hatched, y_lims=y_lims,
                           xlabel=FIG_TEXT["xlabel_hours_price"], show_ylabel=(col == 0))
            price_twin(ax, sup_t[market.price_base], sup_t[market.price], ylim=price_ylim,
                       ylabel=market.price_label, show_axis=(col == 1), positive_ticks=align_zero)
            ax.set_title(state_labels.get(state, state), loc="left")
            if handles is None:                     # both panels share the series
                handles = h

        annual = _annual_twh([_for_area(d[s], area, [year], market.area_key)
                              for d in (supply_dict, demand_dict) for s in STATES], series)
        if verbose:
            print_table(pd.DataFrame({"TWh": annual}).rename(index=label_of), ",.2f",
                        title=f"Årlig energi pr. serie — {label_of(area)}, {year}", unit="TWh")
        keep = [annual[s] >= legend_min_twh for s in series]
        _finish_pair(
            fig, axes, [(s, hd) for s, hd, k in zip(series, handles, keep) if k],
            groups=market.legend_groups, legend_y=legend_y, legend_max_cols=legend_max_cols,
            title=FIG_TEXT[market.title_key].format(area=col_names.get(area, area), year=year),
            save_path=save_path_fmt.format(area=area, year=year) if save_path_fmt else None,
            show=show,
        )
        results.append((fig, tuple(axes)))
    return results


def plot_energy_balance_el(ctx, ea, *, years=None, sort_by="pE_baseline", legend_min_twh=0.0,
                           figsize=(12, 5), save_path_fmt=None, show=True, verbose=False,
                           legend_y=None, legend_max_cols=LEGEND_MAX_COLS, only=None,
                           align_zero=False,
                           state_labels=None):
    """Electricity balance, baseline | shock, for area `ea`; one figure per year, legend
    grouped into supply, demand and trade. Returns [(fig, (ax_base, ax_shock)), ...]."""
    return _plot_balance(ctx, ea, market=EL_MARKET, hatched=EA_cols, years=years,
                         sort_by=sort_by, legend_min_twh=legend_min_twh, figsize=figsize,
                         save_path_fmt=save_path_fmt, show=show, verbose=verbose,
                         legend_y=legend_y, legend_max_cols=legend_max_cols, only=only,
                         align_zero=align_zero, state_labels=state_labels)


def plot_energy_balance_dh(ctx, ha, *, years=None, sort_by="pH_baseline", legend_min_twh=0.0,
                           figsize=(12, 5), save_path_fmt=None, show=True, verbose=False,
                           legend_y=-0.2, state_labels=None):
    """District-heat balance, baseline | shock, for heat area `ha`."""
    return _plot_balance(ctx, ha, market=DH_MARKET, hatched=set(), years=years,
                         sort_by=sort_by, legend_min_twh=legend_min_twh, figsize=figsize,
                         save_path_fmt=save_path_fmt, show=show, verbose=verbose,
                         legend_y=legend_y, state_labels=state_labels)


# ==================== ==================== ==================== ====================
# 2. net export by partner

def _nx_rows(NX_dict, state, ea, years):
    df = NX_dict[state]
    return df[df.index.get_level_values("t").isin(years) & (df.index.get_level_values("EA") == ea)]


def plot_net_export(ctx, ea, *, years=None, sort_by="pE_EA_baseline", figsize=(12, 5),
                    legend_y=-0.1, save_path_fmt=None, show=True, state_labels=None):
    """Net electricity export by partner, baseline | shock, for area `ea`."""
    NX = ctx.NX_dict
    if NX is None:
        raise ValueError("plot_net_export needs an electricity context (NX_dict).")
    state_labels = state_labels or STATE_LABELS
    years = _years(years, NX[STATE_BASE].index.get_level_values("t"))

    connected = _connected_eas(NX, ea)
    partners = [x for x in COL_ORDER if x in connected]
    colors = [color_of(p) for p in partners]
    mins, maxs = [], []
    for state in STATES:
        le = _nx_rows(NX, state, ea, years)["lineExport"]
        if le.empty:
            continue
        maxs.append(le.clip(lower=0).groupby(level=["t", "hFull"]).sum().max())
        mins.append(le.clip(upper=0).groupby(level=["t", "hFull"]).sum().min())
    y_lims = ((min(mins) * 1.05 if mins else 0.0) * 1000,     # GWh, as the balance figure
              (max(maxs) * 1.05 if maxs else 0.0) * 1000)
    price_lims = _price_extent([_nx_rows(NX, s, ea, years)["pE_EA"] for s in STATES])

    results = []
    for year in years:
        fig, axes = new_figure(1, 2, figsize=figsize, sharey=True)
        handles = None
        for col, (ax, state) in enumerate(zip(axes, STATES)):
            sub = NX[state].xs((year, ea), level=["t", "EA"]).sort_values([sort_by, "hFull"])
            le = sub["lineExport"]
            wide = lambda s: (s.unstack("EAEA", sort=False).reset_index(drop=True)
                              .reindex(columns=partners, fill_value=0) * 1000)
            first = lambda c: sub[c].groupby(level="hFull", sort=False).first().reset_index(drop=True)
            h = area_panel(ax, wide(le.clip(lower=0)), wide(le.clip(upper=0)), partners, colors,
                           set(), y_lims=y_lims, xlabel=FIG_TEXT["xlabel_hours_price"],
                           show_ylabel=(col == 0))
            price_twin(ax, first("pE_EA_baseline"), first("pE_EA"), ylim=price_lims,
                       ylabel=FIG_TEXT["ylabel_price_el"], show_axis=(col == 1))
            ax.set_title(state_labels.get(state, state), loc="left")
            if handles is None:                     # both panels share the series
                handles = h

        _finish_pair(
            fig, axes, list(zip(partners, handles)), groups=None, legend_y=legend_y,
            legend_max_cols=None,
            title=FIG_TEXT["title_net_export"].format(area=col_names.get(ea, ea), year=year),
            save_path=save_path_fmt.format(area=ea, year=year) if save_path_fmt else None,
            show=show,
        )
        results.append((fig, tuple(axes)))
    return results


# ==================== ==================== ==================== ====================
# 3. net export attributed to supply components

def build_supply_shares(ctx, *, aggregates=None):
    """Supply mix as shares per (t, EA, h, hFull), imports included, for every area
    plus one aggregate per entry in `aggregates` (label -> member zones)."""
    aggregates = c_aggs_countries if aggregates is None else aggregates
    shares = {}
    for state, df in ctx.supply_dict.items():
        pcols = [c for c in PT_COLS if c in df.columns]          # plant types
        tcols = [c for c in EA_ORDER if c in df.columns]         # trade (imports)
        keep = ["t", "EA", "h", "hFull"]
        base = df[keep + pcols + tcols].copy()

        extra = []
        for label, members in aggregates.items():
            ext_tcols = [c for c in tcols if c not in members]   # internal trade dropped
            agg = base[base["EA"].isin(members)].groupby(
                ["t", "h", "hFull"], as_index=False)[pcols + ext_tcols].sum()
            agg["EA"] = label
            extra.append(agg.reindex(columns=keep + pcols + tcols, fill_value=0))

        full = pd.concat([base] + extra, ignore_index=True)
        comp = pcols + tcols
        tot = full[comp].sum(axis=1)
        full[comp] = full[comp].div(tot.where(tot != 0, 1.0), axis=0)
        shares[state] = full.set_index(["t", "EA", "h", "hFull"])
    return shares


def _aggregate_partner(df, label):
    """Partner zones in `df` collapsed into one `label` partner (flow-weighted price)."""
    w = df["lineExport"].abs()
    g = df.assign(_w=w, _wp=w * df["pE_EAEA"])
    agg = g.groupby(["t", "hFull", "EA"], as_index=False).agg(
        lineExport=("lineExport", "sum"),
        pE_EA=("pE_EA", "first"),
        pE_EA_baseline=("pE_EA_baseline", "first"),
        _wp=("_wp", "sum"), _w=("_w", "sum"), _pmean=("pE_EAEA", "mean"),
    )
    agg["pE_EAEA"] = (agg["_wp"] / agg["_w"]).where(agg["_w"] > 0, agg["_pmean"])
    agg["EAEA"] = label
    return agg.drop(columns=["_wp", "_w", "_pmean"])


def build_nx_by_pt(nx_dict, share_dict, ea, eaea, *, partner_map=None):
    """Net export EA -> EAEA split by supply component: {state: (export, import)}."""
    lvl = share_dict[STATE_BASE].index.get_level_values("EA")
    if ea not in lvl:
        raise KeyError(f"{ea} not in share_dict (build it from the per-EA supply_dict).")
    if eaea not in lvl:
        raise KeyError(f"{eaea} not in share_dict -- add it via aggregates= in build_supply_shares.")
    partner_zones = list((partner_map or {}).get(eaea, [eaea]))

    comp = list(share_dict[STATE_BASE].columns)
    out = {}
    for state in nx_dict:
        nx = nx_dict[state].reset_index()
        nx = nx[(nx["EA"] == ea) & (nx["EAEA"].isin(partner_zones))].copy()
        if len(partner_zones) > 1:
            nx = _aggregate_partner(nx, eaea)

        sh_ea = share_dict[state].xs(ea, level="EA").reset_index()[["t", "hFull"] + comp]
        sh_pa = share_dict[state].xs(eaea, level="EA").reset_index()[["t", "hFull"] + comp]
        exp = nx.merge(sh_ea, on=["t", "hFull"], how="left")
        imp = nx.merge(sh_pa, on=["t", "hFull"], how="left")
        for c in comp:
            exp[c] = (exp[c].clip(lower=0) * exp["lineExport"]).clip(lower=0)
            imp[c] = (imp[c].clip(lower=0) * imp["lineExport"]).clip(upper=0)
        out[state] = (exp, imp)
    return out


def build_nx_by_pt_agg(nx_dict, share_dict, ea, partners, *, partner_map=None,
                       partner_label="ALL"):
    """Net export EA -> all `partners`, summed per hour and component; each
    partner's import flow is split with its own mix before the sum."""
    partners = as_list(partners)
    comp = list(share_dict[STATE_BASE].columns)
    per_partner = {p: build_nx_by_pt(nx_dict, share_dict, ea, p, partner_map=partner_map)
                   for p in partners}
    out = {}
    for state in nx_dict:
        frames = []
        for side, clip in ((0, "lower"), (1, "upper")):     # 0 = export, 1 = import
            df = pd.concat([per_partner[p][state][side] for p in partners], ignore_index=True)
            df["lineExport"] = df["lineExport"].clip(**{clip: 0})
            agg = df.groupby(["t", "hFull"], as_index=False).agg(
                {**{c: "sum" for c in comp}, "lineExport": "sum",
                 "pE_EA": "first", "pE_EA_baseline": "first"})
            agg["EA"], agg["EAEA"] = ea, partner_label
            frames.append(agg)
        out[state] = tuple(frames)
    return out


def _legend_comps(annual, comps, *, min_twh=0.0, min_share=0.0, max_n=None):
    """Components that earn a legend entry (bands are drawn regardless)."""
    total = sum(annual.values())
    keep = [c for c in comps
            if annual[c] >= min_twh and (total <= 0 or annual[c] / total >= min_share)]
    if max_n is not None and len(keep) > max_n:
        top = sorted(keep, key=lambda c: annual[c], reverse=True)[:max_n]
        keep = [c for c in keep if c in set(top)]
    return keep


def _plot_nx_pt(ctx, out, comp, *, ea, partner, years, sort_by, legend_min_twh,
                legend_min_share, legend_max_n, figsize, legend_y, legend_max_cols, save_path_fmt,
                show, state_labels):
    """Engine of the two net-export-by-component figures."""
    state_labels = state_labels or STATE_LABELS
    years = _years(years, ctx.NX_dict[STATE_BASE].index.get_level_values("t"))

    tot = pd.Series(0.0, index=comp)                 # components with flow, in COL_ORDER
    for state in out:
        for df in out[state]:
            tot = tot.add(df[comp].abs().sum(), fill_value=0)
    present = [c for c in comp if tot.get(c, 0) > 1e-7]
    comps = [c for c in COL_ORDER if c in present] + [c for c in present if c not in COL_ORDER]
    if NUCLEAR_PT in comps:                          # nuclear at the bottom of the stack
        comps = [NUCLEAR_PT, *(c for c in comps if c != NUCLEAR_PT)]
    colors = [color_of(c) for c in comps]

    out = {s: tuple(df.assign(**{c: df[c].mul(1000) for c in comps}) for df in out[s])
           for s in out}                             # flows to GWh
    annual = {c: sum(df[c].abs().sum() / 1000.0 for s in out for df in out[s]) / (len(out) or 1)
              for c in comps}
    legend_comps = _legend_comps(annual, comps, min_twh=legend_min_twh,
                                 min_share=legend_min_share, max_n=legend_max_n)
    y_lims = (min(out[s][1][comps].sum(axis=1).min() for s in out) * 1.05,
              max(out[s][0][comps].sum(axis=1).max() for s in out) * 1.05)

    results = []
    for year in years:
        fig, axes = new_figure(1, 2, figsize=figsize, sharey=True)
        handles = None
        for col, (ax, state) in enumerate(zip(axes, STATES)):
            exp, imp = (df[df["t"] == year].sort_values([sort_by, "hFull"]).reset_index(drop=True)
                        for df in out[state])
            h = area_panel(ax, exp, imp, comps, colors, EA_cols, y_lims=y_lims,
                           xlabel=FIG_TEXT["xlabel_hours_src_price"], show_ylabel=(col == 0))
            ax.set_title(state_labels.get(state, state), loc="left")
            if handles is None:                     # both panels share the series
                handles = h

        _finish_pair(
            fig, axes, [(c, handles[comps.index(c)]) for c in legend_comps],
            groups=LEGEND_GROUPS_NX, legend_y=legend_y, legend_max_cols=legend_max_cols,
            title=FIG_TEXT["title_net_export_pt"].format(
                area=col_names.get(ea, ea), partner=col_names.get(partner, partner), year=year),
            save_path=(save_path_fmt.format(area=ea, eaea=partner, year=year)
                       if save_path_fmt else None),
            show=show,
        )
        results.append((fig, tuple(axes)))
    return results


def plot_net_export_by_pt(ctx, ea, eaea, *, share_dict, partner_map=None, years=None,
                          sort_by="pE_EA_baseline", legend_min_twh=0.0, legend_min_share=0.0,
                          legend_max_n=None, figsize=(12, 5), legend_y=None,
                          legend_max_cols=LEGEND_MAX_COLS,
                          save_path_fmt=None, show=True, state_labels=None):
    """Net export EA -> EAEA split by supply component, baseline | shock."""
    if ctx.NX_dict is None:
        raise ValueError("plot_net_export_by_pt needs an electricity context (NX_dict).")
    out = build_nx_by_pt(ctx.NX_dict, share_dict, ea, eaea, partner_map=partner_map)
    return _plot_nx_pt(ctx, out, list(share_dict[STATE_BASE].columns), ea=ea, partner=eaea,
                       years=years, sort_by=sort_by, legend_min_twh=legend_min_twh,
                       legend_min_share=legend_min_share, legend_max_n=legend_max_n,
                       figsize=figsize, legend_y=legend_y, legend_max_cols=legend_max_cols,
                       save_path_fmt=save_path_fmt, show=show, state_labels=state_labels)


def plot_net_export_by_pt_agg(ctx, ea, partners, *, share_dict, partner_map=None,
                              partner_label="ALL", years=None, sort_by="pE_EA_baseline",
                              legend_min_twh=0.0, legend_min_share=0.0, legend_max_n=None,
                              figsize=(12, 5), legend_y=None, legend_max_cols=LEGEND_MAX_COLS,
                              save_path_fmt=None, show=True, state_labels=None):
    """Net export EA -> all `partners` split by supply component, baseline | shock."""
    if ctx.NX_dict is None:
        raise ValueError("plot_net_export_by_pt_agg needs an electricity context (NX_dict).")
    out = build_nx_by_pt_agg(ctx.NX_dict, share_dict, ea, partners,
                             partner_map=partner_map, partner_label=partner_label)
    return _plot_nx_pt(ctx, out, list(share_dict[STATE_BASE].columns), ea=ea,
                       partner=partner_label, years=years, sort_by=sort_by,
                       legend_min_twh=legend_min_twh, legend_min_share=legend_min_share,
                       legend_max_n=legend_max_n, figsize=figsize, legend_y=legend_y,
                       legend_max_cols=legend_max_cols, save_path_fmt=save_path_fmt, show=show,
                       state_labels=state_labels)
