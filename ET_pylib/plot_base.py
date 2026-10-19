"""
Figure furniture — panels, bars, area panels and the shared finishing steps
============================================================================

Drawing pieces shared by `effect_figs`, `balance_figs` and `price_figs`: house
panel styling, bar labels and limits, grouped and signed-stacked bars, area
panels with a price twin axis, and the title / layout / save / show tail.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import (
    BG_COLOR, TEXT_COLOR, PANEL_EDGE, PANEL_EDGE_LW, FIGSIZE_WIDE, FIG_TEXT,
    MISSING_COLOR, PRICE_LINE_COLOR, HATCH_EDGE, ZERO_LINE, col_names, dk_num,
)


# ==================== ==================== ==================== ====================
# 1. panels

def year_label(years):
    """Year tag for figure titles."""
    if years is None:
        return FIG_TEXT["all_years"]
    if isinstance(years, int):
        return str(years)
    ys = sorted(years)
    return str(ys[0]) if len(ys) == 1 else f"{ys[0]}-{ys[-1]}"


def frame_panels(axes):
    """Subtle house border on each axes."""
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(PANEL_EDGE)
            spine.set_linewidth(PANEL_EDGE_LW)


def style_axes(ax):
    """Filled panel, full grid, house border (scatter-type panels)."""
    ax.set_facecolor(BG_COLOR)
    ax.grid(lw=0.6, alpha=0.35)
    frame_panels([ax])


def style_panel(ax, labels, x, *, ylabel=None):
    """Filled panel, house border, y-grid only, Danish category ticks (bar panels)."""
    ax.set_facecolor(BG_COLOR)
    frame_panels([ax])
    ax.grid(axis="y", lw=0.6, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([col_names.get(r, r) for r in labels])
    if ylabel:
        ax.set_ylabel(ylabel, loc="bottom")


def new_figure(nrows=1, ncols=1, *, figsize=FIGSIZE_WIDE, **kw):
    """Figure on the house background."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor(BG_COLOR)
    return fig, axes


def finish(fig, *, title, save_path, show, rect=(0, 0.06, 1, 0.0), w_pad=None):
    """Bar-figure tail: title aligned with the leftmost axes, layout, save, show."""
    kw = {} if w_pad is None else {"w_pad": w_pad}
    fig.tight_layout(rect=list(rect), **kw)
    x0 = min(ax.get_position().x0 for ax in fig.axes)
    fig.suptitle(title, x=x0, ha="left", fontweight="bold")
    fig.tight_layout(rect=list(rect), **kw)
    if save_path:
        fig.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    if show:
        plt.show()


def finish_fixed(fig, *, title, x, rect, save_path, show):
    """Panel-figure tail: title at a fixed x, layout, save, show."""
    if title:
        fig.suptitle(title, x=x, ha="left", fontweight="bold")
    fig.tight_layout(rect=list(rect))
    if save_path:
        fig.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    if show:
        plt.show()


def bottom_legend(fig, handles, labels, *, legend_y, ncol):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y),
               ncol=ncol, frameon=False, fontsize=11)


# ==================== ==================== ==================== ====================
# 2. bars

def bar_labels(ax, pos, vals, span, *, decimals=0, signed=False):
    """Value just outside each bar end."""
    for x, v in zip(pos, vals):
        up = v >= 0
        ax.annotate(dk_num(v, decimals=decimals, signed=signed),
                    (x, v + (0.02 if up else -0.02) * span),
                    ha="center", va="bottom" if up else "top",
                    fontsize=10, color=TEXT_COLOR, zorder=5)


def limits(values, *, pad_frac=0.14):
    """y-limits with headroom for the floating labels; always includes 0."""
    hi, lo = max(values.max(), 0.0), min(values.min(), 0.0)
    pad = pad_frac * max(abs(hi), abs(lo), 1.0)
    return lo - (pad if lo < 0 else 0.0), hi + pad


def zero_line(ax):
    ax.axhline(0, color=ZERO_LINE, lw=1, ls="--", zorder=2)


def signed_bars(ax, labels, values, colors, *, bar_width, decimals, ylabel):
    """One signed bar per label with per-bar colours, value labels and a zero line."""
    x = np.arange(len(labels), dtype=float)
    ax.bar(x, values, bar_width, color=colors, linewidth=0.0, zorder=3)
    y_lims = limits(values, pad_frac=0.22)
    ax.set_ylim(*y_lims)
    bar_labels(ax, x, values, y_lims[1] - y_lims[0], decimals=decimals, signed=True)
    style_panel(ax, labels, x, ylabel=ylabel)
    zero_line(ax)


def draw_grouped(ax, labels, series, *, colors, series_labels, ylabel, decimals,
                 signed=False, bar_width=0.38, edge_keys=(), y_lims=None):
    """Grouped bars for `series` (key -> values) into `ax`; returns legend handles/labels."""
    keys = list(series)
    x = np.arange(len(labels), dtype=float)
    n = len(keys)
    if y_lims is None:
        y_lims = limits(np.concatenate([series[k] for k in keys]))
    span = y_lims[1] - y_lims[0]
    for i, key in enumerate(keys):
        pos = x + (i - (n - 1) / 2) * bar_width
        ax.bar(pos, series[key], bar_width, color=colors[key],
               label=series_labels.get(key, key), linewidth=0.0, zorder=3,
               edgecolor=TEXT_COLOR if key in edge_keys else "none",
               hatch="//" if key in edge_keys else None)
        bar_labels(ax, pos, series[key], span, decimals=decimals, signed=signed)
    ax.set_ylim(*y_lims)
    style_panel(ax, labels, x, ylabel=ylabel)
    if y_lims[0] < 0:
        zero_line(ax)
    handles, leg_labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return handles, leg_labels


def plot_grouped(labels, series, *, colors, series_labels, title, ylabel, decimals,
                 signed=False, figsize=FIGSIZE_WIDE, bar_width=0.38, legend_y=-0.06,
                 edge_keys=(), save_path=None, show=True):
    """Grouped bars over `labels` as a finished figure."""
    fig, ax = new_figure(figsize=figsize)
    handles, leg_labels = draw_grouped(
        ax, labels, series, colors=colors, series_labels=series_labels,
        ylabel=ylabel, decimals=decimals, signed=signed,
        bar_width=bar_width, edge_keys=edge_keys,
    )
    bottom_legend(fig, handles, leg_labels, legend_y=legend_y, ncol=min(len(series), 3))
    finish(fig, title=title, save_path=save_path, show=show)
    return fig, ax


def stack_bars(ax, pos, df, comps, colors, bar_width, *, alpha=None, hatch_keys=()):
    """Sign-aware stack: positive segments up, negative down. Returns (up, down, handles)."""
    up, dn = np.zeros(len(pos)), np.zeros(len(pos))
    handles = {}
    for c in comps:
        v = df[c].to_numpy(dtype=float)
        p, n = np.clip(v, 0, None), np.clip(v, None, 0)
        style = dict(color=colors.get(c, MISSING_COLOR), linewidth=0.0, zorder=3)
        if alpha is not None:
            style["alpha"] = alpha
        if c in hatch_keys:
            style.update(hatch="//", edgecolor=HATCH_EDGE)
        bars = ax.bar(pos, p, bar_width, bottom=up, **style)
        ax.bar(pos, n, bar_width, bottom=dn, **style)
        handles.setdefault(c, bars[0])
        up += p
        dn += n
    return up, dn, handles


def net_markers(ax, pos, net, span, *, decimals, signed):
    """Diamond at the net of a mixed-sign stack, with its value."""
    ax.scatter(pos, net, marker="D", s=42, color=TEXT_COLOR, zorder=6)
    for xx, tv in zip(pos, net):
        ax.annotate(dk_num(tv, decimals=decimals, signed=signed),
                    (xx, tv + 0.03 * span), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=TEXT_COLOR, zorder=7)


def stack_legend(fig, ax, handles, comps, total_label, *, legend_y):
    """Legend for a stacked figure: one entry per component plus the net marker."""
    labels = [col_names.get(c, c) for c in comps] + [total_label]
    leg = [handles[c] for c in comps] + [ax.scatter([], [], marker="D", s=42, color=TEXT_COLOR)]
    bottom_legend(fig, leg, labels, legend_y=legend_y, ncol=min(len(labels), 4))


# ==================== ==================== ==================== ====================
# 3. area panels (hour-sorted stacks with a price twin axis)

def area_panel(ax, pos, neg, series, colors, hatched, *, y_lims, xlabel, show_ylabel):
    """Stacked positive and negative areas over sorted hours; returns legend handles."""
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)

    pos[series].plot(ax=ax, kind="area", stacked=True, linewidth=0.0, ylim=y_lims, color=colors)
    handles, _ = ax.get_legend_handles_labels()
    neg[series].plot(ax=ax, kind="area", stacked=True, linewidth=0.0, ylim=y_lims,
                     color=colors, legend=False)

    for collection, col in zip(ax.collections, list(series) * 2):
        if col in hatched:
            collection.set_hatch("//")
            collection.set_edgecolor(HATCH_EDGE)

    ax.axhline(0, color=ZERO_LINE, lw=1, ls="--")
    ax.grid(lw=0.6, alpha=0.35)
    if show_ylabel:
        ax.set_ylabel(FIG_TEXT["unit_energy"], loc="bottom")
    ax.set_xlabel(xlabel, loc="left")
    if len(pos):
        ax.set_xlim(0, len(pos) - 1)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return handles


def price_twin(ax, base, shock, *, ylim, ylabel, show_axis, positive_ticks=False):
    """Baseline (solid) and shock (dashed) price on a transparent twin axis."""
    ax2 = ax.twinx()
    ax2.patch.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    base.plot(ax=ax2, ylim=ylim, color=PRICE_LINE_COLOR, legend=False)
    shock.plot(ax=ax2, ylim=ylim, color=PRICE_LINE_COLOR, style="--", legend=False)
    ax2.set_ylim(ylim)
    if show_axis:
        ax2.set_ylabel(ylabel, loc="bottom")
        if positive_ticks:
            ax2.set_yticks([t for t in ax2.get_yticks() if t >= 0])
    else:
        ax2.tick_params(right=False, labelright=False)
    ax2.grid(False)
    return ax2
