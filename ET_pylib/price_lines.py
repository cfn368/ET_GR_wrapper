"""Price-duration curves, price-change scatter and ΔCS by aggregate price bin."""

from __future__ import annotations

import math
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from typing import Mapping, Sequence
from matplotlib.lines import Line2D
from .groups import set_aej, BG_COLOR, DEFAULT_COLORS, MISSING_COLOR, DCIIA_TYPES


_CASE_STYLE = {"baseline": "-", "shock": "--"}
_CASE_LABELS = ["Basisforløb", "Stød"]
_NO_FINAL_DEMAND = ("DK3",)  # no DCIIA demand, skipped in regional price weighting


def _demand_members(members):
    return [ea for ea in members if ea not in _NO_FINAL_DEMAND]


# --- distribution =============================================
def _color(col, area_colors):
    return area_colors[col] if col in area_colors else DEFAULT_COLORS.get(col, MISSING_COLOR)


def _label(col, col_names):
    return col_names.get(col, col) if col_names else col


def plot_price_lines(
    ctx,
    areas: Sequence[str] | str,
    *,
    years: Sequence | int | None = None,
    sort: bool = True,
    ncols: int = 3,
    figsize: tuple | None = None,
    row_height: float = 3.2,
    legend_y: float = -0.05,
    save_path_fmt: str | None = None,
    show: bool = True,
):
    """Area electricity prices, baseline vs. shock, on a 3-column grid."""

    sup = ctx.supply_dict
    if years is None:
        years = sorted(sup["baseline"]["t"].unique())
    elif isinstance(years, int):
        years = [years]
    areas = list(areas)

    n = len(areas)
    nrows = math.ceil(n / ncols)
    figsize = figsize or (12, row_height * nrows)

    # Bottom-most populated row index for each column (so we can place xlabels
    # and force tick labels on the visible bottom, not the empty cell below).
    bottom_row_for_col: dict[int, int] = {}
    for idx in range(n):
        r, c = divmod(idx, ncols)
        bottom_row_for_col[c] = max(bottom_row_for_col.get(c, -1), r)

    results = []
    for year in years:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize,
            sharey=True, sharex=True, squeeze=False,
        )
        fig.patch.set_facecolor(BG_COLOR)

        x_max = 0
        for idx, ea in enumerate(areas):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            ax.set_facecolor(BG_COLOR)

            color = _color(ea, ctx.area_colors)
            for state in ("baseline", "shock"):
                df = sup[state]
                price = df[(df["EA"] == ea) & (df["t"] == year)]["pE"]
                if price.empty:
                    continue
                y = (price.sort_values() if sort else price).reset_index(drop=True)
                ax.plot(
                    range(len(y)), y.to_numpy(),
                    color=color, linestyle=_CASE_STYLE[state], linewidth=1.6,
                )
                x_max = max(x_max, len(y))

            ax.grid(lw=0.6, alpha=0.35)
            ax.set_title(_label(ea, ctx.col_names), loc="left")
            if c == 0:
                ax.set_ylabel("Elpris (kr./MWh)", loc="bottom")
            if r == bottom_row_for_col[c]:
                ax.set_xlabel("Timer, sorteret efter områdepris", loc="left")
                ax.tick_params(labelbottom=True)  # sharex would otherwise hide these
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#C9D2D5")
                spine.set_linewidth(0.8)

        if x_max:
            axes[0, 0].set_xlim(0, x_max - 1)  # sharex propagates

        # Empty cells in the last row: legend goes in the first one (skill rule),
        # the rest are turned off.
        empty_axes = [axes[idx // ncols, idx % ncols] for idx in range(n, nrows * ncols)]
        case_handles = [
            Line2D([0], [0], color="0.3", linestyle=_CASE_STYLE[c_], linewidth=1.6)
            for c_ in ("baseline", "shock")
        ]
        if empty_axes:
            legend_ax = empty_axes[0]
            legend_ax.axis("off")
            legend_ax.legend(
                case_handles, _CASE_LABELS,
                loc="center", frameon=False, fontsize=11,
            )
            for ax in empty_axes[1:]:
                ax.axis("off")
            bottom_margin = 0.0
        else:
            fig.legend(
                case_handles, _CASE_LABELS,
                loc="lower center", bbox_to_anchor=(0.5, legend_y),
                ncol=2, frameon=False, fontsize=11,
            )
            bottom_margin = 0.06

        fig.suptitle(
            f"Elpriser, {year}",
            x=0.075, ha="left", fontweight="bold",
        )

        fig.tight_layout(rect=[0, bottom_margin, 1, 1.0])
        results.append((fig, axes))
        if save_path_fmt:
            fig.savefig(save_path_fmt.format(year=year), dpi=300, facecolor=fig.get_facecolor())
        if show:
            plt.show()

    return results


# --- differences =============================================
def _state_prices(sup, ea, year):
    """Collapse the hFull-expanded frames to one row per state h.
    Row count per h equals hWeight, so `hours` needs no lookup in the db."""
    cols = {}
    for state in ("baseline", "shock"):
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
    """Aggregate state prices over several areas (e.g. DK1 + DK2 -> DK).

    Each area is first collapsed to one row per state h with `_state_prices`,
    so hours are not multiplied by the number of areas. Per h, the aggregate
    price is a weighted mean across areas:
      - weight_col=None: simple average.
      - weight_col="<col>": weights from that column in the BASELINE frame
        (e.g. demand), used for both states. Holding the weights fixed at
        baseline keeps dp a pure price effect, not a shift in area mix.
    """
    members = list(members)
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

    out = pd.DataFrame({
        state: sum(W[ea] * parts[ea][state] for ea in members)
        for state in ("baseline", "shock")
    })
    out["hours"] = hours
    out["dp"] = out["shock"] - out["baseline"]
    return out


def _agg_state_prices_demand(ctx, members, year):
    """Aggregate price per state h with the same cross-area weights as the tables
    (`_regional_price`, weights `DCIIA_TYPES`) from each run's own demand frame.
    dp therefore includes any shift in the area demand mix. Areas without final
    demand (DK3) are dropped. Over time, states are weighted by hours."""
    from .summaries import _regional_price

    members = _demand_members(members)
    yr = int(year)

    # hFull -> h from the supply frame (str keys: dtype can differ between frames)
    sup_b = ctx.supply_dict["baseline"]
    hmap = (sup_b.loc[sup_b["EA"].isin(members) & (sup_b["t"].astype(int) == yr),
                      ["hFull", "h"]]
                 .drop_duplicates())
    hmap["hFull"] = hmap["hFull"].astype(str)
    if hmap["hFull"].duplicated().any():
        raise ValueError(f"hFull maps to several h in {members}, {year}")
    hmap = hmap.set_index("hFull")["h"]

    cols = {}
    for state in ("baseline", "shock"):
        df = ctx.demand_dict[state]
        df = df[df["EA"].isin(members) & (df["t"].astype(int) == yr)]
        missing = set(members) - set(df["EA"].unique())
        if missing:
            raise ValueError(f"No demand rows for {sorted(missing)}, {year} ({state})")

        w_cols = [c for c in DCIIA_TYPES if c in df.columns]
        if not w_cols:
            raise KeyError(f"None of DCIIA_TYPES {list(DCIIA_TYPES)} in demand frame columns")
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
    """weight_col='demand' -> table-consistent run-specific demand weights;
    otherwise the original `_agg_state_prices` (None = simple mean)."""
    if weight_col == "demand":
        return _agg_state_prices_demand(ctx, members, year)
    return _agg_state_prices(ctx.supply_dict, members, year, weight_col)


def _equal_hour_bins(x, w, n_bins):
    """Bin index (0 = cheapest) per state, n_bins bins of equal hours sorted by x.
    A state goes to the bin containing its hour-midpoint. Shared by the scatter's
    binned means and the ΔCS bars, so both panels use identical bins."""
    o = x.sort_values(kind="stable").index
    ws = w[o]
    mid = ws.cumsum() - ws / 2
    b = np.minimum((mid / ws.sum() * n_bins).astype(int), n_bins - 1)
    return pd.Series(b.to_numpy(), index=o).reindex(x.index)


def _binned_mean(x, y, w, n_bins):
    """Hour-weighted means of x and y in n_bins bins of equal hours, sorted by x."""
    b = _equal_hour_bins(x, w, n_bins)
    g = pd.DataFrame({"xw": x * w, "yw": y * w, "w": w, "b": b}).groupby("b").sum()
    return g["xw"] / g["w"], g["yw"] / g["w"]


def cs_by_agg_price_bin(baseline_output, shock_output, d, region, members, *,
                        year, n_bins=5, cs_scale=1.0):
    """ΔCS per equal-hour bin of the AGGREGATE baseline price in `d` (from `_agg_state_prices`).

    Unlike `effect_figs.cs_by_price_bin`, which bins each zone on its own price,
    every hour h is placed in one bin for the whole region, so the bars line up
    one-to-one with the scatter's binned means. The total ΔCS is unchanged.
    """
    from .effect_figs import consumer_surplus, _cs_detail, _price_bin_labels

    out = consumer_surplus(baseline_output, shock_output, year=year,
                           areas=list(members), markets=("E",), scale=cs_scale)
    dcs = _cs_detail(out).groupby("h")["dCS"].sum()

    bins = _equal_hour_bins(d["baseline"], d["hours"], n_bins)
    bins.index = bins.index.astype(str)                 # h may be int in one source, str in the other
    b = bins.reindex(dcs.index.astype(str))
    if b.isna().any():
        missing = list(dcs.index[b.isna().to_numpy()][:5])
        raise ValueError(f"No aggregate price for some states in {region}, {year}: {missing}")

    per_bin = dcs.groupby(b.to_numpy()).sum().reindex(range(n_bins), fill_value=0.0)
    if not np.isclose(per_bin.sum(), dcs.sum()):
        raise ValueError("Binned ΔCS does not sum to total ΔCS")
    return pd.DataFrame([per_bin.to_numpy()], index=[region], columns=_price_bin_labels(n_bins))


def _style_axes(ax):
    ax.set_facecolor(BG_COLOR)
    ax.grid(lw=0.6, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#C9D2D5")
        spine.set_linewidth(0.8)


_MEAN_STYLE = {"color": "0.2", "marker": "o", "linewidth": 1.6, "markersize": 5}


def _mean_style(link_color=None):
    """Style of the binned-mean line. With link_color: coloured squares that match the ΔCS bars."""
    if link_color is None:
        return dict(_MEAN_STYLE)
    return {"color": link_color, "marker": "s", "linewidth": 2.0, "markersize": 6}


def _draw_price_change_panel(ax, d, color, n_bins, marker_size, link_color=None):
    """Scatter of dp against baseline price (øre/kWh) plus binned hour-weighted means."""
    ax.scatter(
        d["baseline"]/10, d["dp"]/10,
        s=marker_size * d["hours"] / d["hours"].mean() *2,
        color=color, alpha=0.7, linewidths=0,
    )
    bx, by = _binned_mean(d["baseline"]/10, d["dp"]/10, d["hours"], n_bins)
    ax.axhline(0, color="0.2", lw=1, ls="--")
    ax.plot(bx, by, zorder=5, **_mean_style(link_color))
    _style_axes(ax)


def _scatter_legend(link_color=None):
    handles = [
        Line2D([0], [0], color="0.4", marker="o", linestyle="", markersize=6),
        Line2D([0], [0], **_mean_style(link_color)),
    ]
    labels = ["Time (skaleret efter klyngestørrelse)", "Time-vægtet gns. per kvintil"]
    return handles, labels


def plot_price_change_scatter(
    ctx,
    areas: Sequence[str] | str,
    *,
    years: Sequence | int | None = None,
    aggregates: Mapping[str, Sequence[str]] | None = None,
    weight_col: str | None = "demand",
    n_bins: int = 5,
    marker_size: float = 55,
    ncols: int = 2,
    figsize: tuple | None = None,
    row_height: float = 5.5,
    legend_y: float = 0.0,
    title: str | None = "Figur 3: Kernekraft sænker priserne i de dyreste timer",
    save_path_fmt: str | None = None,
    show: bool = True,
    scatter_color: str | None = "#C7372A",
):
    """Price change (shock - baseline) per state against baseline price, one panel per area.
    Marker size is proportional to hours represented; the line is hour-weighted means in n_bins equal-hour bins.

    `aggregates` maps a panel name to its member areas, e.g. {"DK": ["DK1", "DK2"]};
    any name in `areas` found there is plotted as an aggregate (see `_agg_state_prices`
    for `weight_col`). Other names are plotted as single areas, as before.
    """

    sup = ctx.supply_dict
    aggregates = aggregates or {}
    if years is None:
        years = sorted(sup["baseline"]["t"].unique())
    elif isinstance(years, int):
        years = [years]
    areas = [areas] if isinstance(areas, str) else list(areas)

    n = len(areas)
    if n == 1:
        ncols = 1  # single panel (e.g. DK aggregate) spans the full width
    nrows = math.ceil(n / ncols)
    figsize = figsize or (12, row_height * nrows)

    bottom_row_for_col: dict[int, int] = {}
    for idx in range(n):
        r, c = divmod(idx, ncols)
        bottom_row_for_col[c] = max(bottom_row_for_col.get(c, -1), r)

    results = []
    for year in years:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize,
            sharey=True, sharex=True, squeeze=False,
        )
        fig.patch.set_facecolor(BG_COLOR)

        for idx, ea in enumerate(areas):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            ax.set_facecolor(BG_COLOR)

            if ea in aggregates:
                d = _agg_dispatch(ctx, aggregates[ea], year, weight_col)
            else:
                d = _state_prices(sup, ea, year)
            color = scatter_color or _color(ea, ctx.area_colors)
            _draw_price_change_panel(ax, d, color, n_bins, marker_size)

            ax.set_title(_label(ea, ctx.col_names), loc="left")
            if c == 0:
                ax.set_ylabel("Stød − basisforløb (øre/kWh)", loc="bottom")
            if r == bottom_row_for_col[c]:
                ax.set_xlabel("Pris, basisforløb (øre/kWh)", loc="left")
                ax.tick_params(labelbottom=True)

        handles, labels = _scatter_legend()

        empty_axes = [axes[idx // ncols, idx % ncols] for idx in range(n, nrows * ncols)]
        if empty_axes:
            legend_ax = empty_axes[0]
            legend_ax.axis("off")
            legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=11)
            for ax in empty_axes[1:]:
                ax.axis("off")
            bottom_margin = 0.0
        else:
            fig.legend(
                handles, labels,
                loc="lower center", bbox_to_anchor=(0.5, legend_y),
                ncol=2, frameon=False, fontsize=11,
            )
            bottom_margin = 0.06

        if title:
            fig.suptitle(title, x=0.060, ha="left", fontweight="bold")

        fig.tight_layout(rect=[0, bottom_margin, 1, 1.0])
        results.append((fig, axes))
        if save_path_fmt:
            fig.savefig(save_path_fmt.format(year=year), dpi=300, facecolor=fig.get_facecolor())
        if show:
            plt.show()

    return results


# --- price change + consumer surplus =========================
def plot_price_change_and_cs(
    ctx,
    baseline_output,
    shock_output,
    region: str = "DK",
    *,
    year=None,
    members: Sequence[str] | None = None,
    weight_col: str | None = "demand",
    n_bins: int = 5,
    cs_scale: float = 1.0,
    marker_size: float = 55,
    figsize: tuple = (12, 5.5),
    bar_width: float = 0.6,
    decimals: int = 0,
    legend_y: float = 0.0,
    left_title: str = "Ændring i pris per time",
    right_title: str = "Ændring i forbrugeroverskud",
    title: str | None = "Figur 3: Kernekraft sænker priserne i de dyreste timer",
    scatter_color: str | None = "#9AA5AA",  # neutral, so the linked line stands out
    link_color: str | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """Two panels for one (aggregate) region, sharing the equal-hour price bins.
    The binned-mean line is drawn with squares in `link_color` (default: the bar colour),
    so each square on the left visibly corresponds to one bar on the right.

    left  = price change per state against baseline price (as `plot_price_change_scatter`),
    right = ΔCS per baseline-price bin (as `plot_cs_by_price_bin` for a single region).

    `members` defaults to the region's areas in `c_aggs` minus areas without final
    demand (DK3), so both panels cover the same zones.
    """
    # Imported here to avoid a circular import between the figure modules.
    from .effect_figs import (
        DEFAULT_YEAR, FIG_TEXT, area_colors, c_aggs, col_names, _draw_grouped,
    )

    year = DEFAULT_YEAR if year is None else year
    members = list(members) if members is not None else _demand_members(c_aggs[region])

    # Left: scatter on the (weighted) aggregate price.
    sup = ctx.supply_dict
    t_vals = sup["baseline"]["t"]
    t_year = t_vals[t_vals.astype(int) == int(year)].iloc[0]  # match dtype of t in supply_dict
    d = _agg_dispatch(ctx, members, t_year, weight_col)

    # Right: ΔCS per bin of the same aggregate price, so bars match the scatter's quintiles.
    df = cs_by_agg_price_bin(baseline_output, shock_output, d, region, members,
                             year=year, n_bins=n_bins, cs_scale=cs_scale)

    bar_color = area_colors.get(region, MISSING_COLOR)
    link_color = link_color or bar_color

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)

    _draw_price_change_panel(ax_l, d, scatter_color or _color(region, ctx.area_colors),
                             n_bins, marker_size, link_color=link_color)
    ax_l.set_title(left_title, loc="left")
    ax_l.set_ylabel("Stød − basisforløb (øre/kWh)", loc="bottom")
    ax_l.set_xlabel("Pris, basisforløb (øre/kWh)", loc="left")

    _draw_grouped(
        ax_r, list(df.columns), {region: df.loc[region].to_numpy(dtype=float)},
        colors={region: link_color},
        series_labels=col_names, ylabel="mio. kr.", decimals=decimals,
        signed=True, bar_width=bar_width,
    )
    _style_axes(ax_r)
    ax_r.set_title(right_title, loc="left")
    ax_r.set_xlabel('Kvintil pris, basisforløb', loc="left")

    handles, labels = _scatter_legend(link_color)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y),
               ncol=2, frameon=False, fontsize=11)

    if title:
        fig.suptitle(title, x=0.060, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])

    if save_path:
        fig.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig, (ax_l, ax_r)