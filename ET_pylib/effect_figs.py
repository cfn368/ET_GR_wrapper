"""Figures for scenario effects: prices, trade, production, profit, welfare.

Data helpers return frames; plot helpers draw them. All rendered text and
styling comes from `groups` -- no string literals or colours here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .groups import *
from .summaries import price_summary, full_trade_summary, DEFAULT_YEAR
from .partial_welfare import consumer_surplus, producer_profit, producer_generation
from .scarcity import capture_summary
from .bottle_neck import ci_by_country


# =========================================================================== #
# 1. data -- levels
# =========================================================================== #
def price_levels(ctx, regions=None, *, years=DEFAULT_YEAR):
    """Stack `price_summary` over regions -> one frame indexed by (region, run)."""
    regions = list(REGION_ORDER) if regions is None else list(regions)
    return pd.concat([price_summary(ctx, country=r, years=years) for r in regions])


def trade_levels(ctx, regions=None, *, years=DEFAULT_YEAR, component="net"):
    """All-partner trade totals per region -> frame indexed by (region, run),
    columns TWh / bnDKK. `component`: "net" (default), "export", "import"."""
    regions = list(REGION_ORDER) if regions is None else list(regions)
    rows = []
    for region in regions:
        tot = full_trade_summary(ctx, country=region, years=years).loc["TOTAL"]
        for state in STATES:
            rows.append({
                "region": region, "run": state,
                "TWh": tot.loc[state, f"{component}_TWh"],
                "bnDKK": tot.loc[state, f"{component}_mDKK"] / 1000.0,
            })
    return pd.DataFrame(rows).set_index(["region", "run"])


def _countries_in(region):
    """Countries whose zones constitute `region`, deduped, order-stable."""
    seen = []
    for ea in c_aggs[region]:
        c = EA_TO_COUNTRY.get(ea)
        if c is None:
            raise KeyError(f"{ea} has no country in EA_TO_COUNTRY")
        if c not in seen:
            seen.append(c)
    return seen


def _ci_delta_by_country(baseline_output, shock_output, year):
    """Δ congestion rent by country (mio. kr.), shock − baseline."""
    _, c_base = ci_by_country(baseline_output, year)
    _, c_shock = ci_by_country(shock_output, year)
    b = c_base["CI_mioDKK"].groupby(level=0).sum()
    s = c_shock["CI_mioDKK"].groupby(level=0).sum()
    return s.subtract(b, fill_value=0.0)


def welfare_levels(
    baseline_output, shock_output, *, id2Pt=None, id2EA=None, regions=None,
    year=DEFAULT_YEAR, markets=("E",), cs_scale=1.0,
    include_ci=True, annuity=None,
):
    """ΔCS / ΔPS / ΔCI / annuity / net per region (mia. kr.). `annuity` is
    {country: mio. kr./yr}; a region without its own key sums its countries."""
    regions = list(REGION_ORDER) if regions is None else list(regions)
    ci = _ci_delta_by_country(baseline_output, shock_output, year) if include_ci else None
    ann = {} if annuity is None else dict(annuity)

    rows = []
    for region in regions:
        areas = c_aggs[region]
        countries = _countries_in(region)

        _, _, total, _ = consumer_surplus(
            baseline_output, shock_output, year=year, areas=areas,
            markets=markets, scale=cs_scale,
        )
        dcs = float(total.loc["+".join(markets), "dCS"]) / 1000.0

        prof = producer_profit(
            baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
            areas=areas, year=year,
        )
        dps = float(_total_dprofit(prof)) / 1000.0

        dci = 0.0 if ci is None else sum(float(ci.get(c, 0.0)) for c in countries) / 1000.0
        cap = (ann[region] if region in ann
               else sum(ann.get(c, 0.0) for c in countries)) / 1000.0

        rows.append({"region": region, "dCS": dcs, "dPS": dps, "dCI": dci,
                     "annuity": -cap, "net": dcs + dps + dci - cap})
    return pd.DataFrame(rows).set_index("region")


def generation_by_pt(baseline_output, shock_output, *, id2Pt, id2EA=None,
                     regions=None, year=DEFAULT_YEAR):
    """Generation (TWh) per region x aggregated plant type, both states."""
    return _stack_frame(producer_generation, baseline_output, shock_output,
                        id2Pt=id2Pt, id2EA=id2EA, regions=regions, year=year,
                        base_col="q_base", shock_col="q_shock", scale=1.0)


def profit_by_pt(baseline_output, shock_output, *, id2Pt, id2EA=None,
                 regions=None, year=DEFAULT_YEAR):
    """Producer profit (mia. kr.) per region x aggregated plant type, both states."""
    return _stack_frame(producer_profit, baseline_output, shock_output,
                        id2Pt=id2Pt, id2EA=id2EA, regions=regions, year=year,
                        base_col="profit_base", shock_col="profit_shock", scale=1e-3)


# =========================================================================== #
# 2. data -- deltas
# =========================================================================== #
def trade_delta_by_partner(ctx, regions=None, *, years=DEFAULT_YEAR,
                           component="net", value=False):
    """Δ(shock - baseline) trade per region x partner. TWh, or mia. kr. if `value`."""
    regions = list(REGION_ORDER) if regions is None else list(regions)
    col = f"{component}_{'mDKK' if value else 'TWh'}"
    scale = 1e-3 if value else 1.0
    rows = {}
    for region in regions:
        t = full_trade_summary(ctx, country=region, years=years)
        t = t.drop(index="TOTAL", errors="ignore")[col].unstack("run")
        rows[region] = (t["shock"] - t["baseline"]) * scale
    df = pd.DataFrame(rows).T.fillna(0.0)
    cols = [p for p in PARTNER_ORDER if p in df.columns]
    return df.loc[regions, cols + [c for c in df.columns if c not in cols]]


def nx_delta_by_pt(ctx, share_dict, regions=None, *, years=DEFAULT_YEAR,
                   partner_map=None):
    """Δ(shock - baseline) net export per region x supply component (TWh).
    Components are plant-type buckets plus partner-origin bands (transit).
    Exports carry the origin zone's supply mix, imports the partner zone's."""
    regions = list(REGION_ORDER) if regions is None else list(regions)
    yrs = [years] if isinstance(years, int) else list(years)
    comp = list(share_dict["baseline"].columns)

    rows = {}
    for region in regions:
        members = set(c_aggs[region])
        net = pd.Series(0.0, index=comp)
        for state, sign in (("shock", 1.0), ("baseline", -1.0)):
            nx = ctx.NX_dict[state].reset_index()
            nx = nx[nx["t"].isin(yrs)]
            nx = nx[nx["EA"].isin(members) & ~nx["EAEA"].isin(members)]
            if nx.empty:
                continue
            sh = share_dict[state].reset_index()
            exp = nx.merge(sh, on=["t", "hFull", "EA"], how="left")
            imp = nx.merge(sh, left_on=["t", "hFull", "EAEA"],
                           right_on=["t", "hFull", "EA"], how="left",
                           suffixes=("", "_p"))
            for c in comp:
                up = (exp[c].clip(lower=0) * exp["lineExport"]).clip(lower=0).sum()
                dn = (imp[c].clip(lower=0) * imp["lineExport"]).clip(upper=0).sum()
                net[c] += sign * (up + dn)
        rows[region] = net
    return _collapse_comps(pd.DataFrame(rows).T.fillna(0.0).loc[regions])


def capture_delta_by_pt(ctx, *, country="DK", years=DEFAULT_YEAR, min_gen=100.0):
    """Δ capture price (DKK/MWh) per aggregated plant type for one region.
    Capture is a ratio: buckets are generation-weighted, not simple means.
    Buckets below `min_gen` (GWh, baseline) are dropped as immaterial."""
    cap = capture_summary(ctx, country=country, years=years).reset_index()

    cols, gen_base = {}, None
    for state, suf in (("baseline", "basel"), ("shock", "shock")):
        gen = cap.set_index("Pt")[f"gen_{suf}"].fillna(0.0)
        capt = cap.set_index("Pt")[f"capture_{suf}"].fillna(0.0)
        num = _pt_agg(gen * capt, what=f"capture/{country}/{state}")
        den = _pt_agg(gen, what=f"capture/{country}/{state}")
        cols[state] = num / den.where(den > 0)
        if state == "baseline":
            gen_base = den

    d = (pd.DataFrame(cols)["shock"] - pd.DataFrame(cols)["baseline"]).dropna()
    return d[gen_base.reindex(d.index) >= min_gen]


def _stack_delta(frames):
    """shock - baseline, region x PT_AGG_ORDER."""
    return frames["shock"] - frames["baseline"]


# =========================================================================== #
# 3. reshape helpers
# =========================================================================== #
def _total_dprofit(prof):
    """Scalar ΔProfit from whatever `producer_profit` returns (frame or tuple)."""
    if isinstance(prof, tuple):
        prof = prof[0]
    if isinstance(prof, pd.DataFrame):
        return prof["dProfit"].sum()
    return prof.sum()


def _pivot(df, stat):
    """(labels, {state: values}) for one statistic, in region order."""
    labels = list(dict.fromkeys(df.index.get_level_values("region")))
    vals = {
        s: np.array([df.loc[(r, s), stat] for r in labels], dtype=float)
        for s in STATES
    }
    return labels, vals


def _pt_agg(s, *, aggs=PT_AGGS, drop=PT_DROP, what=""):
    """Collapse a Pt-indexed Series onto PT_AGGS, dropping heat-only types.
    Unmapped types fail loudly."""
    inv = {pt: label for label, pts in aggs.items() for pt in pts}
    s = s[~s.index.isin(drop)]
    missing = sorted({p for p in s.index if p not in inv and abs(s[p]) > 1e-9})
    if missing:
        raise KeyError(f"{what}: unmapped plant types with non-zero values: {missing}")
    out = s.groupby(s.index.map(lambda p: inv.get(p, "UNMAPPED"))).sum()
    return out.reindex(PT_AGG_ORDER, fill_value=0.0)


def _collapse_comps(df, *, aggs=NX_COMP_AGGS, order=NX_COMP_ORDER):
    """Sum component columns onto their aggregate labels, in display order."""
    inv = {c: label for label, members in aggs.items() for c in members}
    unmapped = [c for c in df.columns if c not in inv and df[c].abs().max() > 1e-9]
    if unmapped:
        raise KeyError(f"unmapped components with non-zero values: {unmapped}")
    out = df.T.groupby(df.columns.map(lambda c: inv.get(c, "UNMAPPED"))).sum().T
    cols = [c for c in order if c in out.columns]
    return out[cols + [c for c in out.columns if c not in cols]]


def _by_pt(fn, baseline_output, shock_output, *, id2Pt, id2EA, areas, year,
           base_col, shock_col):
    """One region's baseline/shock Series indexed by Pt (t level dropped).
    split_nuclear: the nuclear ids (NUCLEAR_IDS) leave the CD bucket for Kernekraft."""
    df = fn(baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
            areas=areas, year=year, split_nuclear=True).droplevel("t")
    return df[base_col], df[shock_col]


def _stack_frame(fn, baseline_output, shock_output, *, id2Pt, id2EA, regions,
                 year, base_col, shock_col, scale):
    """{state: DataFrame(region x PT_AGG_ORDER)}."""
    regions = list(REGION_ORDER) if regions is None else list(regions)
    out = {s: {} for s in STATES}
    for region in regions:
        b, s = _by_pt(fn, baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
                      areas=c_aggs[region], year=year,
                      base_col=base_col, shock_col=shock_col)
        out["baseline"][region] = _pt_agg(b, what=f"{fn.__name__}/{region}") * scale
        out["shock"][region] = _pt_agg(s, what=f"{fn.__name__}/{region}") * scale
    return {k: pd.DataFrame(v).T.loc[regions, PT_AGG_ORDER] for k, v in out.items()}


def _plot_bars(labels, values, *, colors, title, ylabel, decimals,
               figsize=FIGSIZE_WIDE, bar_width=0.55, save_path=None, show=True):
    """Single signed series, one bar per label, per-label colour. No legend."""
    x = np.arange(len(labels), dtype=float)
    v = np.asarray(values, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)

    ax.bar(x, v, bar_width, color=[colors.get(l, MISSING_COLOR) for l in labels],
           linewidth=0.0, zorder=3)

    y_lims = _limits(v, pad_frac=0.22)
    ax.set_ylim(*y_lims)
    _bar_labels(ax, x, v, y_lims[1] - y_lims[0], decimals=decimals, signed=True)
    _style_panel(ax, labels, x, ylabel=ylabel)
    ax.axhline(0, color="0.2", lw=1, ls="--", zorder=2)

    _finish(fig, title=title, save_path=save_path, show=show)
    return fig, ax


def plot_capture_delta_by_pt(
    ctx, *, country="DK", years=DEFAULT_YEAR,
    figsize: tuple = FIGSIZE_WIDE, bar_width: float = 0.55, decimals: int = 1,
    save_path: str | None = None, show: bool = True,
):
    """Δ capture price by plant type for one region (default DK)."""
    d = capture_delta_by_pt(ctx, country=country, years=years) / 10.0
    return _plot_bars(
        list(d.index), d.to_numpy(),
        colors=PT_AGG_COLORS,
        title=FIG_TEXT["title_capture_delta"].format(
            region=col_names.get(country, country), year=_year_label(years)
        ),
        ylabel=FIG_TEXT["unit_price_ore"], decimals=decimals,
        figsize=figsize, bar_width=bar_width, save_path=save_path, show=show,
    )


# =========================================================================== #
# 4. shared drawing furniture
# =========================================================================== #
def _year_label(years):
    """Year tag for figure titles."""
    if years is None:
        return FIG_TEXT["all_years"]
    if isinstance(years, int):
        return str(years)
    ys = sorted(years)
    return str(ys[0]) if len(ys) == 1 else f"{ys[0]}-{ys[-1]}"


def _style_panel(ax, labels, x, *, ylabel=None):
    """House panel: filled background, subtle border, y-grid only, Danish ticks."""
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(PANEL_EDGE)
        spine.set_linewidth(PANEL_EDGE_LW)
    ax.grid(axis="y", lw=0.6, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([col_names.get(r, r) for r in labels])
    if ylabel:
        ax.set_ylabel(ylabel, loc="bottom")


def _bar_labels(ax, pos, vals, span, *, decimals=0, signed=False):
    """Float the value just outside each bar end."""
    for x, v in zip(pos, vals):
        up = v >= 0
        ax.annotate(
            dk_num(v, decimals=decimals, signed=signed),
            (x, v + (0.02 if up else -0.02) * span),
            ha="center", va="bottom" if up else "top",
            fontsize=10, color=TEXT_COLOR, zorder=5,
        )


def _limits(values, *, pad_frac=0.14):
    """y-limits with headroom for the floating labels; always includes 0."""
    hi, lo = max(values.max(), 0.0), min(values.min(), 0.0)
    pad = pad_frac * max(abs(hi), abs(lo), 1.0)
    return lo - (pad if lo < 0 else 0.0), hi + pad


def _finish(fig, *, title, save_path, show, rect=(0, 0.06, 1, 0.0),
            w_pad=None, wspace=None):
    """Shared title / layout / save / show tail."""
    kw = {} if w_pad is None else {"w_pad": w_pad}
    fig.tight_layout(rect=list(rect), **kw)
    x0 = min(ax.get_position().x0 for ax in fig.axes)
    fig.suptitle(title, x=x0, ha="left", fontweight="bold")
    fig.tight_layout(rect=list(rect), **kw)
    if wspace is not None:
        fig.subplots_adjust(wspace=wspace)   # must come after tight_layout
    if save_path:
        fig.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    if show:
        plt.show()


# =========================================================================== #
# 5. plot engines
# =========================================================================== #
def _draw_grouped(
    ax, labels, series, *, colors, series_labels, ylabel, decimals,
    signed=False, bar_width=0.38, edge_keys=(), y_lims=None,
):
    """Draw grouped bars for `series` into `ax`; return (handles, legend labels)."""
    keys = list(series)
    x = np.arange(len(labels), dtype=float)
    n = len(keys)
    if y_lims is None:
        y_lims = _limits(np.concatenate([series[k] for k in keys]))
    span = y_lims[1] - y_lims[0]
    for i, key in enumerate(keys):
        pos = x + (i - (n - 1) / 2) * bar_width
        ax.bar(pos, series[key], bar_width, color=colors[key],
               label=series_labels.get(key, key), linewidth=0.0, zorder=3,
               edgecolor=TEXT_COLOR if key in edge_keys else "none",
               hatch="//" if key in edge_keys else None)
        _bar_labels(ax, pos, series[key], span, decimals=decimals, signed=signed)
    ax.set_ylim(*y_lims)
    _style_panel(ax, labels, x, ylabel=ylabel)
    if y_lims[0] < 0:
        ax.axhline(0, color="0.2", lw=1, ls="--", zorder=2)
    handles, leg_labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return handles, leg_labels


def _plot_grouped(
    labels, series, *, colors, series_labels, title, ylabel, decimals,
    signed=False, figsize=FIGSIZE_WIDE, bar_width=0.38, legend_y=-0.06,
    edge_keys=(), save_path=None, show=True,
):
    """Grouped bars over `labels`; `series` maps key -> values, drawn in order."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    handles, leg_labels = _draw_grouped(
        ax, labels, series, colors=colors, series_labels=series_labels,
        ylabel=ylabel, decimals=decimals, signed=signed,
        bar_width=bar_width, edge_keys=edge_keys,
    )
    fig.legend(handles, leg_labels, loc="lower center",
               bbox_to_anchor=(0.5, legend_y), ncol=min(len(series), 3),
               frameon=False, fontsize=11)
    _finish(fig, title=title, save_path=save_path, show=show)
    return fig, ax


def _plot_levels(
    df, stat, *, title, ylabel, decimals, figsize, bar_width, legend_y,
    save_path, show, verbose,
):
    """Grouped baseline/shock bars over regions."""
    labels, vals = _pivot(df, stat)
    if verbose:
        print(df.round(2))
    return _plot_grouped(
        labels, vals, colors=STATE_COLORS, series_labels=STATE_LABELS,
        title=title, ylabel=ylabel, decimals=decimals, figsize=figsize,
        bar_width=bar_width, legend_y=legend_y, save_path=save_path, show=show,
    )


def _plot_stacked_states(
    frames, *, title, ylabel, decimals, figsize, bar_width, legend_y,
    save_path, show,
):
    """One axes: baseline|shock stacked bars paired per region.
    Sign-aware -- positive segments stack up, negative down, net marked."""
    regions = list(frames["baseline"].index)
    comps = [c for c in frames["baseline"].columns
             if max(frames[s][c].abs().max() for s in STATES) > 0]
    x = np.arange(len(regions), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)

    spans, nets, handles = [], [], {}
    for i, state in enumerate(STATES):
        df, pos = frames[state], x + (i - 0.5) * bar_width
        up, dn = np.zeros(len(regions)), np.zeros(len(regions))
        for c in comps:
            v = df[c].to_numpy(dtype=float)
            p, n = np.clip(v, 0, None), np.clip(v, None, 0)
            style = dict(color=PT_AGG_COLORS[c], linewidth=0.0, zorder=3,
                         alpha=1.0 if state == "baseline" else 0.72)
            bars = ax.bar(pos, p, bar_width, bottom=up, **style)
            ax.bar(pos, n, bar_width, bottom=dn, **style)
            handles.setdefault(c, bars[0])
            up += p
            dn += n
        spans += [up, dn]
        nets.append((pos, up + dn))

    y_lims = _limits(np.concatenate(spans), pad_frac=0.18)
    span = y_lims[1] - y_lims[0]
    for pos, net in nets:                       # net is not the visual top when signs mix
        ax.scatter(pos, net, marker="D", s=42, color=TEXT_COLOR, zorder=6)
        for xx, tv in zip(pos, net):
            ax.annotate(dk_num(tv, decimals=decimals),
                        (xx, tv + 0.03 * span), ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color=TEXT_COLOR, zorder=7)

    ax.set_ylim(*y_lims)
    _style_panel(ax, regions, x, ylabel=ylabel)
    if y_lims[0] < 0:
        ax.axhline(0, color="0.2", lw=1, ls="--", zorder=2)

    labels = [col_names.get(c, c) for c in comps] + [FIG_TEXT["total_label"]]
    leg = [handles[c] for c in comps] + [
        ax.scatter([], [], marker="D", s=42, color=TEXT_COLOR)
    ]
    fig.legend(leg, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y),
               ncol=min(len(labels), 4), frameon=False, fontsize=11)
    _finish(fig, title=title, save_path=save_path, show=show)
    return fig, ax


def _plot_stacked_delta(
    df, *, title, ylabel, decimals, figsize, bar_width, legend_y,
    save_path, show, hatch_keys=(), colors=None, total_label=None,
):
    """Signed stacked bars: positive segments up, negative down, total marked.
    Components in `hatch_keys` are hatched (partner-origin / transit bands)."""
    colors = area_colors if colors is None else colors
    total_label = TRADE_COMP_LABELS["net"] if total_label is None else total_label
    regions = list(df.index)
    comps = [c for c in df.columns if df[c].abs().max() > 0]
    x = np.arange(len(regions), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)

    up = np.zeros(len(regions))
    dn = np.zeros(len(regions))
    handles = {}
    for c in comps:
        v = df[c].to_numpy(dtype=float)
        pos, neg = np.clip(v, 0, None), np.clip(v, None, 0)
        style = dict(color=colors.get(c, MISSING_COLOR), linewidth=0.0, zorder=3)
        if c in hatch_keys:
            style.update(hatch="//", edgecolor="#666666")
        b = ax.bar(x, pos, bar_width, bottom=up, **style)
        ax.bar(x, neg, bar_width, bottom=dn, **style)
        handles[c] = b[0]
        up += pos
        dn += neg

    total = df[comps].sum(axis=1).to_numpy(dtype=float)
    y_lims = _limits(np.concatenate([up, dn]), pad_frac=0.18)
    span = y_lims[1] - y_lims[0]
    ax.scatter(x, total, marker="D", s=42, color=TEXT_COLOR, zorder=6)
    for xx, tv in zip(x, total):                    # total label, clear of the stack
        ax.annotate(dk_num(tv, decimals=decimals, signed=True),
                    (xx, tv + 0.03 * span), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=TEXT_COLOR, zorder=7)

    ax.set_ylim(*y_lims)
    _style_panel(ax, regions, x, ylabel=ylabel)
    ax.axhline(0, color="0.2", lw=1, ls="--", zorder=2)

    labels = [col_names.get(c, c) for c in comps] + [total_label]
    leg = [handles[c] for c in comps] + [
        ax.scatter([], [], marker="D", s=42, color=TEXT_COLOR)
    ]
    fig.legend(leg, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y),
               ncol=min(len(labels), 4), frameon=False, fontsize=11)
    _finish(fig, title=title, save_path=save_path, show=show)
    return fig, ax


# =========================================================================== #
# 6. figures -- levels
# =========================================================================== #
def plot_price_levels(
    ctx, regions=None, *, years=DEFAULT_YEAR,
    figsize: tuple = FIGSIZE_WIDE, bar_width: float = 0.38,
    legend_y: float = -0.06, decimals: int = 1,
    save_path: str | None = None, show: bool = True, verbose: bool = False,
):
    """Demand-weighted mean price by region, baseline vs. shock (grouped bars)."""
    df = price_levels(ctx, regions, years=years) / 10.0   # DKK/MWh -> øre/kWh
    return _plot_levels(
        df, "mean",
        title=FIG_TEXT["title_price_levels"].format(year=_year_label(years)),
        ylabel=FIG_TEXT["unit_price_ore"], decimals=decimals,
        figsize=figsize, bar_width=bar_width, legend_y=legend_y,
        save_path=save_path, show=show, verbose=verbose,
    )


def plot_trade_levels(
    ctx, regions=None, *, years=DEFAULT_YEAR, component="net", value=False,
    figsize: tuple = FIGSIZE_WIDE, bar_width: float = 0.38,
    legend_y: float = -0.06, decimals: int | None = None,
    save_path: str | None = None, show: bool = True, verbose: bool = False,
):
    """Net export by region, baseline vs. shock. `value=False` -> TWh, True -> mia. kr."""
    title = FIG_TEXT["title_trade_v" if value else "title_trade_q"].format(
        year=_year_label(years)
    )
    if component != "net":                       # keep the default title clean
        comp = TRADE_COMP_LABELS.get(component, component)
        title = f"{comp}: {title[0].lower()}{title[1:]}"
    return _plot_levels(
        trade_levels(ctx, regions, years=years, component=component),
        "bnDKK" if value else "TWh",
        title=title,
        ylabel=FIG_TEXT["unit_value" if value else "unit_energy_yr"],
        decimals=1 if decimals is None else decimals,
        figsize=figsize, bar_width=bar_width, legend_y=legend_y,
        save_path=save_path, show=show, verbose=verbose,
    )


def plot_welfare_levels(
    baseline_output, shock_output, *, id2Pt=None, id2EA=None, regions=None,
    year=DEFAULT_YEAR, markets=("E",), cs_scale=1.0,
    include_ci=True, annuity=None, series=WELFARE_SERIES,
    figsize: tuple | None = None, bar_width: float | None = None,
    legend_y: float = -0.06, decimals: int = 1,
    save_path: str | None = None, show: bool = True,
):
    """ΔCS, ΔPS and net by region -- partial-equilibrium welfare effect.
    `include_ci` adds the congestion rent and `annuity` the annuitised cable cost
    to the net bar; series=WELFARE_SERIES_FULL draws them as their own bars."""
    df = welfare_levels(
        baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
        regions=regions, year=year, markets=markets, cs_scale=cs_scale,
        include_ci=include_ci, annuity=annuity,
    )
    series = tuple(series)
    missing = [k for k in series if k not in df.columns]
    if missing:
        raise KeyError(f"welfare_levels has no column(s) {missing} for series={series}")

    if annuity:
        note = FIG_TEXT["welfare_net_note"]
    elif include_ci:
        note = FIG_TEXT["welfare_ci_note"]
    else:
        note = FIG_TEXT["delta_note"]

    if figsize is None:                      # 5 series x 4 regions needs the height
        figsize = (FIGSIZE_WIDE[0], 5) if len(series) > 3 else FIGSIZE_WIDE

    return _plot_grouped(
        list(df.index),
        {k: df[k].to_numpy(dtype=float) for k in series},
        colors=WELFARE_COLORS, series_labels=WELFARE_COMP_LABELS,
        title=FIG_TEXT["title_welfare"].format(year=_year_label(year), note=note),
        ylabel=FIG_TEXT["unit_value"], decimals=decimals, signed=True,
        edge_keys=("net",), figsize=figsize,
        bar_width=bar_width if bar_width else round(0.78 / len(series), 3),
        legend_y=legend_y, save_path=save_path, show=show,
    )


def plot_generation_stack(
    baseline_output, shock_output, *, id2Pt, id2EA=None, regions=None,
    year=DEFAULT_YEAR, figsize: tuple = FIGSIZE_WIDE, bar_width: float = 0.38,
    legend_y: float = -0.06, decimals: int = 1,
    save_path: str | None = None, show: bool = True,
):
    """Electricity generation (TWh) by plant type, baseline | shock."""
    frames = generation_by_pt(baseline_output, shock_output, id2Pt=id2Pt,
                              id2EA=id2EA, regions=regions, year=year)
    return _plot_stacked_states(
        frames, title=FIG_TEXT["title_gen_stack"].format(year=_year_label(year)),
        ylabel=FIG_TEXT["unit_energy_yr"], decimals=decimals, figsize=figsize,
        bar_width=bar_width, legend_y=legend_y, save_path=save_path, show=show,
    )


def plot_profit_stack(
    baseline_output, shock_output, *, id2Pt, id2EA=None, regions=None,
    year=DEFAULT_YEAR, figsize: tuple = FIGSIZE_WIDE, bar_width: float = 0.38,
    legend_y: float = -0.06, decimals: int = 1,
    save_path: str | None = None, show: bool = True,
):
    """Producer profit (mia. kr.) by plant type, baseline | shock."""
    frames = profit_by_pt(baseline_output, shock_output, id2Pt=id2Pt,
                          id2EA=id2EA, regions=regions, year=year)
    return _plot_stacked_states(
        frames, title=FIG_TEXT["title_profit_stack"].format(year=_year_label(year)),
        ylabel=FIG_TEXT["unit_value"], decimals=decimals, figsize=figsize,
        bar_width=bar_width, legend_y=legend_y, save_path=save_path, show=show,
    )


# =========================================================================== #
# 7. figures -- deltas
# =========================================================================== #
def plot_price_delta(
    ctx, regions=None, *, years=DEFAULT_YEAR, stats=("mean",), relative: bool = False,
    figsize: tuple = (12, 5), bar_width: float = 0.55, decimals: int = 1,
    save_path: str | None = None, show: bool = True, verbose: bool = False,
):
    """Shock - baseline price effect by region, one panel per entry in `stats`.
    `stats` accepts any column of `price_summary` ("mean", "std", "min", "max")."""
    df = price_levels(ctx, regions, years=years) / 10.0   # DKK/MWh -> øre/kWh
    stats = (stats,) if isinstance(stats, str) else tuple(stats)

    fig, axes = plt.subplots(1, len(stats), figsize=figsize, squeeze=False)
    axes = axes.ravel()
    fig.patch.set_facecolor(BG_COLOR)

    for ax, stat in zip(axes, stats):
        labels, vals = _pivot(df, stat)
        x = np.arange(len(labels), dtype=float)
        base, shock = vals["baseline"], vals["shock"]
        d = 100.0 * (shock - base) / base if relative else shock - base

        colors = [DELTA_COLORS["up" if v >= 0 else "down"] for v in d]
        ax.bar(x, d, bar_width, color=colors, linewidth=0.0, zorder=3)

        y_lims = _limits(d, pad_frac=0.22)
        ax.set_ylim(*y_lims)
        _bar_labels(ax, x, d, y_lims[1] - y_lims[0], decimals=decimals, signed=True)

        _style_panel(ax, labels, x,
                     ylabel=FIG_TEXT["unit_pct" if relative else "unit_price_ore"])
        ax.axhline(0, color="0.2", lw=1, ls="--", zorder=2)
        ax.set_title(PRICE_STAT_LABELS.get(stat, stat), loc="left", pad=2)
        if verbose:
            print(stat, dict(zip(labels, np.round(d, 3))))

    _finish(
        fig, show=show, save_path=save_path, rect=(0, 0.02, 1, 1.05),
        title=FIG_TEXT["title_price_delta"].format(
            year=_year_label(years),
            unit=FIG_TEXT["delta_rel" if relative else "delta_abs"],
            note=FIG_TEXT["delta_note"],
        ),
    )
    return fig, tuple(axes)


def plot_trade_delta_by_partner(
    ctx, regions=None, *, years=DEFAULT_YEAR, component="net", value=False,
    figsize: tuple = FIGSIZE_WIDE, bar_width: float = 0.45,
    legend_y: float = -0.1, decimals: int = 1,
    save_path: str | None = None, show: bool = True,
):
    """Δ net export decomposed by trade partner, one stacked bar per region."""
    df = trade_delta_by_partner(ctx, regions, years=years,
                                component=component, value=value)
    return _plot_stacked_delta(
        df,
        title=FIG_TEXT["title_trade_delta_v" if value else "title_trade_delta_q"].format(
            year=_year_label(years)
        ),
        ylabel=FIG_TEXT["unit_value" if value else "unit_energy_yr"],
        decimals=decimals, figsize=figsize, bar_width=bar_width,
        legend_y=legend_y, save_path=save_path, show=show,
    )


def plot_nx_delta_by_pt(
    ctx, share_dict, regions=None, *, years=DEFAULT_YEAR, partner_map=None,
    figsize: tuple = FIGSIZE_WIDE, bar_width: float = 0.45,
    legend_y: float = -0.20, decimals: int = 1,
    save_path: str | None = None, show: bool = True,
):
    """Δ net export decomposed by supply component, one stacked bar per region."""
    df = nx_delta_by_pt(ctx, share_dict, regions, years=years,
                        partner_map=partner_map)
    return _plot_stacked_delta(
        df, title=FIG_TEXT["title_nx_pt_delta"].format(year=_year_label(years)),
        ylabel=FIG_TEXT["unit_energy_yr"], decimals=decimals,
        hatch_keys=set(PARTNER_ORDER), colors=NX_COMP_COLORS, figsize=figsize,
        bar_width=bar_width, legend_y=legend_y, save_path=save_path, show=show,
    )


def plot_generation_delta(
    baseline_output, shock_output, *, id2Pt, id2EA=None, regions=None,
    year=DEFAULT_YEAR, figsize: tuple = FIGSIZE_WIDE, bar_width: float = 0.45,
    legend_y: float = -0.10, decimals: int = 1,
    save_path: str | None = None, show: bool = True,
):
    """Δ electricity generation (TWh) decomposed by plant type."""
    df = _stack_delta(generation_by_pt(baseline_output, shock_output, id2Pt=id2Pt,
                                       id2EA=id2EA, regions=regions, year=year))
    return _plot_stacked_delta(
        df, title=FIG_TEXT["title_gen_delta"].format(year=_year_label(year)),
        ylabel=FIG_TEXT["unit_energy_yr"], decimals=decimals,
        colors=PT_AGG_COLORS, figsize=figsize, bar_width=bar_width,
        legend_y=legend_y, save_path=save_path, show=show,
        total_label=FIG_TEXT["total_label"],
    )


def plot_profit_delta(
    baseline_output, shock_output, *, id2Pt, id2EA=None, regions=None,
    year=DEFAULT_YEAR, figsize: tuple = FIGSIZE_WIDE, bar_width: float = 0.45,
    legend_y: float = -0.10, decimals: int = 1,
    save_path: str | None = None, show: bool = True,
):
    """Δ producer profit (mia. kr.) decomposed by plant type."""
    df = _stack_delta(profit_by_pt(baseline_output, shock_output, id2Pt=id2Pt,
                                   id2EA=id2EA, regions=regions, year=year))
    return _plot_stacked_delta(
        df, title=FIG_TEXT["title_profit_delta"].format(year=_year_label(year)),
        ylabel=FIG_TEXT["unit_value"], decimals=decimals,
        colors=PT_AGG_COLORS, figsize=figsize, bar_width=bar_width,
        legend_y=legend_y, save_path=save_path, show=show,
        total_label=FIG_TEXT["total_label"],
    )



# =========================================================================== #
# 8. Shock v. shock
# =========================================================================== #

CABLE_PANEL_LABELS = ("+700 MW kabler", "\u2212700 MW kabler")


def plot_price_delta_2col(
    ctxs, regions=None, *, years=DEFAULT_YEAR, stat: str = "mean",
    relative: bool = False, panel_labels=CABLE_PANEL_LABELS, share_y: bool = True,
    figsize: tuple = (12, 5), bar_width: float = 0.55, decimals: int = 1,
    save_path: str | None = None, show: bool = True, verbose: bool = False,
):
    """Price effect of two shocks side by side -- one panel per context.
    `ctxs` is (ctx_left, ctx_right); both must cover the same regions."""
    if len(ctxs) != 2 or len(panel_labels) != 2:
        raise ValueError("plot_price_delta_2col needs exactly two contexts and two labels")

    labels_ref, deltas = None, []
    for ctx in ctxs:
        df = price_levels(ctx, regions, years=years) / 10.0   # DKK/MWh -> øre/kWh
        labels, vals = _pivot(df, stat)
        base, shock = vals["baseline"], vals["shock"]
        d = 100.0 * (shock - base) / base if relative else shock - base
        if labels_ref is None:
            labels_ref = list(labels)
        elif list(labels) != labels_ref:
            raise ValueError(f"region mismatch: {labels_ref} vs {list(labels)}")
        deltas.append(np.asarray(d, dtype=float))

    shared = _limits(np.concatenate(deltas), pad_frac=0.22) if share_y else None

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=share_y, squeeze=False)
    axes = axes.ravel()
    fig.patch.set_facecolor(BG_COLOR)
    x = np.arange(len(labels_ref), dtype=float)
    unit = FIG_TEXT["unit_pct" if relative else "unit_price_ore"]

    for i, (ax, d, panel) in enumerate(zip(axes, deltas, panel_labels)):
        y_lims = shared if shared is not None else _limits(d, pad_frac=0.22)
        colors = [DELTA_COLORS["up" if v >= 0 else "down"] for v in d]
        ax.bar(x, d, bar_width, color=colors, linewidth=0.0, zorder=3)
        ax.set_ylim(*y_lims)
        _bar_labels(ax, x, d, y_lims[1] - y_lims[0], decimals=decimals, signed=True)
        _style_panel(ax, labels_ref, x, ylabel=unit if i == 0 else "")
        ax.axhline(0, color="0.2", lw=1, ls="--", zorder=2)
        ax.set_title(panel, loc="left", pad=2)
        if verbose:
            print(panel, dict(zip(labels_ref, np.round(d, 3))))

    _finish(
        fig, show=show, save_path=save_path, rect=(0, 0.02, 1, 1.05),
        title=FIG_TEXT["title_price_delta"].format(
            year=_year_label(years),
            unit=FIG_TEXT["delta_rel" if relative else "delta_abs"],
            note=FIG_TEXT["delta_note"],
        ),
    )
    return fig, tuple(axes)


def plot_welfare_levels_2col(
    panels, *, id2Pt=None, id2EA=None, regions=None, year=DEFAULT_YEAR,
    markets=("E",), cs_scale=1.0, include_ci=True, series=WELFARE_SERIES,
    panel_labels=CABLE_PANEL_LABELS, share_y: bool = True,
    figsize: tuple | None = None, bar_width: float | None = None,
    legend_y: float = -0.06, decimals: int = 1,
    w_pad: float | None = 0.4, hide_inner_ticks: bool = True,
    save_path: str | None = None, show: bool = True, 
    wspace: float | None = 0.06,
):
    """ΔCS, ΔPS and net for two shocks side by side.
    `panels` is ((baseline, shock, annuity), (baseline, shock, annuity))."""
    if len(panels) != 2 or len(panel_labels) != 2:
        raise ValueError("plot_welfare_levels_2col needs exactly two panels and two labels")
    series = tuple(series)
    frames = []
    for baseline_output, shock_output, annuity in panels:
        df = welfare_levels(
            baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
            regions=regions, year=year, markets=markets, cs_scale=cs_scale,
            include_ci=include_ci, annuity=annuity,
        )
        missing = [k for k in series if k not in df.columns]
        if missing:
            raise KeyError(f"welfare_levels has no column(s) {missing} for series={series}")
        frames.append(df)
    if list(frames[0].index) != list(frames[1].index):
        raise ValueError(f"region mismatch: {list(frames[0].index)} vs {list(frames[1].index)}")
    labels = list(frames[0].index)
    annuities = [p[2] for p in panels]
    if any(a is not None for a in annuities):
        note = FIG_TEXT["welfare_net_note"]
    elif include_ci:
        note = FIG_TEXT["welfare_ci_note"]
    else:
        note = FIG_TEXT["delta_note"]
    if figsize is None:
        figsize = (FIGSIZE_WIDE[0], 5)
    if bar_width is None:
        bar_width = round(0.78 / len(series), 3)
    vals = [{k: df[k].to_numpy(dtype=float) for k in series} for df in frames]
    shared = (_limits(np.concatenate([v[k] for v in vals for k in series]))
              if share_y else None)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=share_y, squeeze=False)
    axes = axes.ravel()
    fig.patch.set_facecolor(BG_COLOR)
    handles = leg_labels = None
    for i, (ax, v, panel) in enumerate(zip(axes, vals, panel_labels)):
        h, lab = _draw_grouped(
            ax, labels, v, colors=WELFARE_COLORS, series_labels=WELFARE_COMP_LABELS,
            ylabel=FIG_TEXT["unit_value"] if i == 0 else "", decimals=decimals,
            signed=True, bar_width=bar_width, edge_keys=("net",), y_lims=shared,
        )
        ax.set_title(panel, loc="left", pad=2)
        if hide_inner_ticks and i == 1:
            ax.tick_params(left=False)
        if handles is None:
            handles, leg_labels = h, lab
    fig.legend(handles, leg_labels, loc="lower center",
               bbox_to_anchor=(0.5, legend_y), ncol=min(len(series), 3),
               frameon=False, fontsize=11)
    _finish(
        fig, show=show, save_path=save_path, w_pad=w_pad, wspace=wspace,
        title=FIG_TEXT["title_welfare"].format(year=_year_label(year), note=note),
    )
    return fig, tuple(axes)


# =========================================================================== #
# 9. Economic security of supply -- ΔCS by price level
# =========================================================================== #
def _hour_bins(x, w, n_bins):
    """Bin index 0..n_bins-1 per row: rows sorted by x, bins of equal hours (weights w)."""
    o = x.sort_values(kind="mergesort").index
    mid = w[o].cumsum() - w[o] / 2                      # hour-midpoint of each state
    b = np.minimum((mid / w.sum() * n_bins).astype(int), n_bins - 1)
    return b.reindex(x.index)


def _price_bin_labels(n_bins):
    """Percentile labels for equal-hour bins, cheapest first."""
    edges = [round(100 * k / n_bins) for k in range(n_bins + 1)]
    return [FIG_TEXT["price_bin_fmt"].format(lo=lo, hi=hi)
            for lo, hi in zip(edges[:-1], edges[1:])]


def _cs_detail(out):
    """The per-(h, area) frame inside whatever `consumer_surplus` returns."""
    frames = out if isinstance(out, tuple) else (out,)
    seen = []
    for f in frames:
        if not isinstance(f, pd.DataFrame):
            continue
        f = f.reset_index()
        seen.append(list(f.columns))
        if {"h", "area", "dCS"} <= set(f.columns):
            return f
    raise KeyError(f"consumer_surplus returned no frame with h/area/dCS; frames had {seen}")


def cs_by_price_bin(baseline_output, shock_output, regions=None, *,
                    year=DEFAULT_YEAR, n_bins=5, cs_scale=1.0):
    """ΔCS (mia. kr.) per region x equal-hour bin of baseline electricity price.
    Bins are formed per zone on its own price (hWeight hours); bin k sums the zones' k-th bins."""
    regions = list(REGION_ORDER) if regions is None else list(regions)
    hw = baseline_output("hWeight")
    p0 = baseline_output("pE").rename("pE").reset_index()
    p0 = p0[p0["t"].astype(int) == int(year)].set_index(["h", "EA"])["pE"]

    rows = {}
    for region in regions:
        out = consumer_surplus(baseline_output, shock_output, year=year,
                               areas=c_aggs[region], markets=("E",), scale=cs_scale)
        detail = _cs_detail(out)
        acc = np.zeros(n_bins)
        for ea, g in detail.groupby("area"):
            dcs = g.groupby("h")["dCS"].sum()
            price = p0.xs(ea, level="EA").reindex(dcs.index)
            w = hw.reindex(dcs.index)
            if price.isna().any() or w.isna().any():
                raise ValueError(f"pE/hWeight missing for some states in {ea}, {year}")
            b = _hour_bins(price, w, n_bins)
            acc += dcs.groupby(b).sum().reindex(range(n_bins), fill_value=0.0).to_numpy()
        rows[region] = acc 
    return pd.DataFrame(rows, index=_price_bin_labels(n_bins)).T


def plot_cs_by_price_bin(
    baseline_output, shock_output, regions=None, *, year=DEFAULT_YEAR,
    n_bins: int = 5, cs_scale: float = 1.0,
    figsize: tuple = FIGSIZE_WIDE, bar_width: float | None = None,
    legend_y: float = -0.12, decimals: int = 0,
    save_path: str | None = None, show: bool = True, verbose: bool = False,
):
    """ΔCS by equal-hour bins of baseline price (cheapest -> most expensive), one series
    per region. Bins sum to the ΔCS in `plot_welfare_levels` (markets=("E",))."""
    df = cs_by_price_bin(baseline_output, shock_output, regions, year=year,
                         n_bins=n_bins, cs_scale=cs_scale)
    if verbose:
        print(df.assign(total=df.sum(axis=1)).round(3))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    handles, leg_labels = _draw_grouped(
        ax, list(df.columns), {r: df.loc[r].to_numpy(dtype=float) for r in df.index},
        colors={r: area_colors.get(r, MISSING_COLOR) for r in df.index},
        series_labels=col_names, ylabel='mio. kr.', decimals=decimals,
        signed=True, bar_width=bar_width if bar_width else round(0.78 / len(df), 3),
    )
    ax.set_xlabel(FIG_TEXT["price_bin_axis"], loc="left")
    fig.legend(handles, leg_labels, loc="lower center", bbox_to_anchor=(0.5, legend_y),
               ncol=min(len(df), 4), frameon=False, fontsize=11)
    _finish(fig, title=FIG_TEXT["title_cs_price_bin"].format(year=_year_label(year)),
            save_path=save_path, show=show)
    return fig, ax