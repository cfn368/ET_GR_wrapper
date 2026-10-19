"""
Price figures — price change per hour against the baseline price, and ΔCS by bin
=================================================================================

Scatter of the shock - baseline price per state hour against the baseline price
(marker size = hours represented) with hour-weighted means in equal-hour bins,
optionally next to the ΔCS in the same bins.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .groups import (
    DCIIA_TYPES, AREAS_WITHOUT_DCIIA, ANALYSIS_YEAR, STATES, as_list, areas_of, expand_areas,
)
from .style import (
    FIG_TEXT, SCATTER_COLOR, SCATTER_NEUTRAL, ZERO_LINE, MISSING_COLOR,
    area_colors, col_names, color_of, label_of,
)
from ._util import equal_hour_bins
from .market import _regional_price
from .welfare import cs_by_agg_price_bin
from .plot_base import new_figure, style_axes, draw_grouped, bottom_legend, finish_fixed

TITLE_X = 0.060


# ==================== ==================== ==================== ====================
# 1. state prices

def _state_prices(sup, ea, year):
    """The hFull-expanded frames collapsed to one row per state h; rows per h equal
    hWeight, so `hours` needs no lookup in the db."""
    cols = {}
    for state in STATES:
        df = sup[state]
        g = df[(df["EA"] == ea) & (df["t"] == year)].groupby("h")["pE"]
        if (g.nunique() > 1).any():
            raise ValueError(f"pE varies within a state h for {ea}, {year} ({state})")
        cols[state] = g.first()
        if state == "baseline":
            cols["hours"] = g.size()
    out = pd.DataFrame(cols)
    if out.isna().any().any():
        raise ValueError(f"State sets differ between baseline and shock for {ea}, {year}")
    out["dp"] = out["shock"] - out["baseline"]
    return out


def _agg_state_prices(sup, members, year, weight_col=None):
    """State prices averaged over several areas per state h: simple mean, or weights
    from `weight_col` in the baseline frame (fixed, so dp is a pure price effect)."""
    members = list(expand_areas(members))
    parts = {ea: _state_prices(sup, ea, year) for ea in members}

    hours = parts[members[0]]["hours"]
    for ea, d in parts.items():
        if not d["hours"].sort_index().equals(hours.sort_index()):
            raise ValueError(f"State h / hours differ between {members[0]} and {ea}, {year}")

    if weight_col is None:
        W = pd.DataFrame({ea: 1.0 for ea in members}, index=hours.index)
    else:
        base = sup["baseline"]
        W = pd.DataFrame({
            ea: base[(base["EA"] == ea) & (base["t"] == year)].groupby("h")[weight_col].mean()
            for ea in members
        }).reindex(hours.index)
        if W.isna().any().any():
            raise ValueError(f"Missing '{weight_col}' for some state h in {members}, {year}")
    W = W.div(W.sum(axis=1), axis=0)

    out = pd.DataFrame({state: sum(W[ea] * parts[ea][state] for ea in members)
                        for state in STATES})
    out["hours"] = hours
    out["dp"] = out["shock"] - out["baseline"]
    return out


def _agg_state_prices_demand(ctx, members, year):
    """Aggregate price per state h with the table weights (`_regional_price` on the
    DC-IIA demand of each run), so dp includes any shift in the area demand mix.
    Areas without final demand are dropped."""
    members = [ea for ea in members if ea not in AREAS_WITHOUT_DCIIA]
    yr = int(year)

    # hFull -> h from the supply frame (str keys: dtypes can differ between frames)
    sup_b = ctx.supply_dict["baseline"]
    hmap = (sup_b.loc[sup_b["EA"].isin(members) & (sup_b["t"].astype(int) == yr), ["hFull", "h"]]
                 .drop_duplicates())
    hmap["hFull"] = hmap["hFull"].astype(str)
    if hmap["hFull"].duplicated().any():
        raise ValueError(f"hFull maps to several h in {members}, {year}")
    hmap = hmap.set_index("hFull")["h"]

    cols = {}
    for state in STATES:
        df = ctx.demand_dict[state]
        df = df[df["EA"].isin(members) & (df["t"].astype(int) == yr)]
        missing = set(members) - set(df["EA"].unique())
        if missing:
            raise ValueError(f"No demand rows for {sorted(missing)}, {year} ({state})")

        w_cols = [c for c in DCIIA_TYPES if c in df.columns]
        if not w_cols:
            raise KeyError(f"None of DCIIA_TYPES {list(DCIIA_TYPES)} in the demand frame")
        reg = _regional_price(df[["t", "EA", "hFull", "pE", *w_cols]],
                              price="pE", weight_cols=tuple(w_cols))
        reg["h"] = reg["hFull"].astype(str).map(hmap)
        if reg["h"].isna().any():
            raise ValueError(f"hFull without h mapping in {members}, {year} ({state})")

        g = reg.groupby("h")["pE_region"]
        if not np.allclose(g.max(), g.min(), rtol=1e-9, atol=1e-6):
            raise ValueError(f"Regional pE varies within a state h for {members}, {year} ({state})")
        cols[state] = g.first()
        if state == "baseline":
            cols["hours"] = g.size()

    out = pd.DataFrame(cols)
    if out.isna().any().any():
        raise ValueError(f"State sets differ between baseline and shock for {members}, {year}")
    out["dp"] = out["shock"] - out["baseline"]
    return out


def _agg_dispatch(ctx, members, year, weight_col):
    """weight_col="demand" -> table-consistent run-specific weights; otherwise
    `_agg_state_prices` (None = simple mean)."""
    if weight_col == "demand":
        return _agg_state_prices_demand(ctx, members, year)
    return _agg_state_prices(ctx.supply_dict, members, year, weight_col)


# ==================== ==================== ==================== ====================
# 2. drawing

def _binned_mean(x, y, w, n_bins):
    """Hour-weighted means of x and y in n_bins equal-hour bins sorted by x."""
    b = equal_hour_bins(x, w, n_bins)
    g = pd.DataFrame({"xw": x * w, "yw": y * w, "w": w, "b": b}).groupby("b").sum()
    return g["xw"] / g["w"], g["yw"] / g["w"]


def _mean_style(link_color=None):
    """Binned-mean line; with link_color, coloured squares matching the ΔCS bars."""
    if link_color is None:
        return {"color": ZERO_LINE, "marker": "o", "linewidth": 1.6, "markersize": 5}
    return {"color": link_color, "marker": "s", "linewidth": 2.0, "markersize": 6}


def _draw_price_change_panel(ax, d, color, n_bins, marker_size, link_color=None):
    """Scatter of dp against the baseline price (øre/kWh) plus binned means."""
    ax.scatter(d["baseline"] / 10, d["dp"] / 10,
               s=marker_size * d["hours"] / d["hours"].mean() * 2,
               color=color, alpha=0.7, linewidths=0)
    bx, by = _binned_mean(d["baseline"] / 10, d["dp"] / 10, d["hours"], n_bins)
    ax.axhline(0, color=ZERO_LINE, lw=1, ls="--")
    ax.plot(bx, by, zorder=5, **_mean_style(link_color))
    style_axes(ax)


def _scatter_legend(link_color=None):
    handles = [Line2D([0], [0], color="0.4", marker="o", linestyle="", markersize=6),
               Line2D([0], [0], **_mean_style(link_color))]
    return handles, [FIG_TEXT["legend_state_point"], FIG_TEXT["legend_binned_mean"]]


# ==================== ==================== ==================== ====================
# 3. figures

def plot_price_change_scatter(ctx, areas, *, years=None, aggregates=None, weight_col="demand",
                              n_bins=5, marker_size=55, ncols=2, figsize=None, row_height=5.5,
                              legend_y=0.0, title=FIG_TEXT["title_price_scatter"],
                              save_path_fmt=None, show=True, scatter_color=SCATTER_COLOR):
    """Price change per state hour against the baseline price, one panel per area.
    `aggregates` maps a panel name to member areas, e.g. {"DK": ["DK1", "DK2"]}."""
    sup = ctx.supply_dict
    aggregates = aggregates or {}
    if years is None:
        years = sorted(sup["baseline"]["t"].unique())
    elif isinstance(years, int):
        years = [years]
    areas = as_list(areas)

    n = len(areas)
    ncols = 1 if n == 1 else ncols
    nrows = math.ceil(n / ncols)
    figsize = figsize or (12, row_height * nrows)
    bottom_row = {}
    for idx in range(n):
        r, c = divmod(idx, ncols)
        bottom_row[c] = max(bottom_row.get(c, -1), r)

    results = []
    for year in years:
        fig, axes = new_figure(nrows, ncols, figsize=figsize, sharey=True, sharex=True,
                               squeeze=False)
        for idx, ea in enumerate(areas):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            d = (_agg_dispatch(ctx, aggregates[ea], year, weight_col) if ea in aggregates
                 else _state_prices(sup, ea, year))
            _draw_price_change_panel(ax, d, scatter_color or color_of(ea), n_bins, marker_size)
            ax.set_title(label_of(ea), loc="left")
            if c == 0:
                ax.set_ylabel(FIG_TEXT["ylabel_dp"], loc="bottom")
            if r == bottom_row[c]:
                ax.set_xlabel(FIG_TEXT["xlabel_base_price"], loc="left")
                ax.tick_params(labelbottom=True)

        handles, labels = _scatter_legend()
        empty = [axes[idx // ncols, idx % ncols] for idx in range(n, nrows * ncols)]
        if empty:
            for ax in empty:
                ax.axis("off")
            empty[0].legend(handles, labels, loc="center", frameon=False, fontsize=11)
            bottom = 0.0
        else:
            bottom_legend(fig, handles, labels, legend_y=legend_y, ncol=2)
            bottom = 0.06

        finish_fixed(fig, title=title, x=TITLE_X, rect=(0, bottom, 1, 1.0),
                     save_path=save_path_fmt.format(year=year) if save_path_fmt else None,
                     show=show)
        results.append((fig, axes))
    return results


def plot_price_change_and_cs(ctx, baseline_output, shock_output, region="DK", *, year=None,
                             members=None, weight_col="demand", n_bins=5, cs_scale=1.0,
                             marker_size=55, figsize=(12, 5.5), bar_width=0.6, decimals=0,
                             legend_y=0.0, left_title=FIG_TEXT["panel_price_change"],
                             right_title=FIG_TEXT["panel_cs_change"],
                             title=FIG_TEXT["title_price_scatter"],
                             scatter_color=SCATTER_NEUTRAL, link_color=None,
                             save_path=None, show=True):
    """Price change per state hour (left) next to ΔCS per baseline-price bin (right),
    both on the same equal-hour bins of the region's aggregate price. `members`
    defaults to the region's areas with final demand."""
    year = ANALYSIS_YEAR if year is None else year
    members = (list(expand_areas(members)) if members is not None
               else [ea for ea in areas_of(region) if ea not in AREAS_WITHOUT_DCIIA])

    t_vals = ctx.supply_dict["baseline"]["t"]
    t_year = t_vals[t_vals.astype(int) == int(year)].iloc[0]   # dtype of t in supply_dict
    d = _agg_dispatch(ctx, members, t_year, weight_col)
    df = cs_by_agg_price_bin(baseline_output, shock_output, d, region, members,
                             year=year, n_bins=n_bins, cs_scale=cs_scale)

    link_color = link_color or area_colors.get(region, MISSING_COLOR)
    fig, (ax_l, ax_r) = new_figure(1, 2, figsize=figsize)

    _draw_price_change_panel(ax_l, d, scatter_color or color_of(region), n_bins, marker_size,
                             link_color=link_color)
    ax_l.set_title(left_title, loc="left")
    ax_l.set_ylabel(FIG_TEXT["ylabel_dp"], loc="bottom")
    ax_l.set_xlabel(FIG_TEXT["xlabel_base_price"], loc="left")

    draw_grouped(ax_r, list(df.columns), {region: df.loc[region].to_numpy(dtype=float)},
                 colors={region: link_color}, series_labels=col_names,
                 ylabel=FIG_TEXT["unit_value_mio"], decimals=decimals, signed=True,
                 bar_width=bar_width)
    style_axes(ax_r)
    ax_r.set_title(right_title, loc="left")
    ax_r.set_xlabel(FIG_TEXT["xlabel_price_quintile"], loc="left")

    handles, labels = _scatter_legend(link_color)
    bottom_legend(fig, handles, labels, legend_y=legend_y, ncol=2)
    finish_fixed(fig, title=title, x=TITLE_X, rect=(0, 0.06, 1, 1.0),
                 save_path=save_path, show=show)
    return fig, (ax_l, ax_r)
