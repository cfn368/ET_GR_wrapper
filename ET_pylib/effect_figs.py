"""
Effect figures — prices, trade, production, profit and welfare by region
=========================================================================

Bar figures of baseline vs. shock and of shock - baseline per region. The data
helpers at the top return frames; the figures below draw them. Text and colours
come from `style`, drawing pieces from `plot_base`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .groups import (
    STATES, REGION_ORDER, PARTNER_ORDER, PT_AGGS, PT_DROP, PT_AGG_ORDER,
    NX_COMP_AGGS, NX_COMP_ORDER, WELFARE_SERIES, ANALYSIS_YEAR, c_aggs,
)
from .style import (
    FIG_TEXT, FIGSIZE_WIDE, STATE_COLORS, STATE_LABELS, DELTA_COLORS, PT_AGG_COLORS,
    NX_COMP_COLORS, WELFARE_COLORS, WELFARE_COMP_LABELS, PRICE_STAT_LABELS,
    TRADE_COMP_LABELS, MISSING_COLOR, area_colors, col_names,
)
from .market import price_summary, full_trade_summary, capture_summary
from .welfare import producer_profit, producer_generation, welfare_levels, cs_by_price_bin
from .plot_base import (
    year_label, new_figure, finish, limits, zero_line, style_panel, signed_bars,
    draw_grouped, plot_grouped, bottom_legend, stack_bars, net_markers, stack_legend,
)


# ==================== ==================== ==================== ====================
# 1. data: levels and deltas by region

def price_levels(ctx, regions=None, *, years=ANALYSIS_YEAR):
    """`price_summary` stacked over regions, indexed by (region, run)."""
    regions = list(REGION_ORDER) if regions is None else list(regions)
    return pd.concat([price_summary(ctx, country=r, years=years) for r in regions])


def trade_levels(ctx, regions=None, *, years=ANALYSIS_YEAR, component="net"):
    """All-partner trade totals per region, indexed by (region, run), columns TWh /
    bnDKK. `component`: "net", "export" or "import"."""
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


def trade_delta_by_partner(ctx, regions=None, *, years=ANALYSIS_YEAR,
                           component="net", value=False):
    """Shock - baseline trade per region x partner; TWh, or mia. kr. if `value`."""
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


def nx_delta_by_pt(ctx, share_dict, regions=None, *, years=ANALYSIS_YEAR):
    """Shock - baseline net export per region x supply component (TWh). Exports
    carry the origin zone's supply mix, imports the partner zone's."""
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
                           right_on=["t", "hFull", "EA"], how="left", suffixes=("", "_p"))
            for c in comp:
                up = (exp[c].clip(lower=0) * exp["lineExport"]).clip(lower=0).sum()
                dn = (imp[c].clip(lower=0) * imp["lineExport"]).clip(upper=0).sum()
                net[c] += sign * (up + dn)
        rows[region] = net
    return _collapse_comps(pd.DataFrame(rows).T.fillna(0.0).loc[regions])


def capture_delta_by_pt(ctx, *, country="DK", years=ANALYSIS_YEAR, min_gen=100.0):
    """Δ capture price (DKK/MWh) per aggregated plant type, generation-weighted;
    buckets below `min_gen` GWh of baseline generation are dropped."""
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


def _stack_frame(fn, baseline_output, shock_output, *, id2Pt, id2EA, regions, year,
                 base_col, shock_col, scale):
    """{state: frame region x PT_AGG_ORDER}, nuclear split out of CD."""
    regions = list(REGION_ORDER) if regions is None else list(regions)
    out = {s: {} for s in STATES}
    for region in regions:
        df = fn(baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
                areas=c_aggs[region], year=year, split_nuclear=True).droplevel("t")
        out["baseline"][region] = _pt_agg(df[base_col], what=f"{fn.__name__}/{region}") * scale
        out["shock"][region] = _pt_agg(df[shock_col], what=f"{fn.__name__}/{region}") * scale
    return {k: pd.DataFrame(v).T.loc[regions, PT_AGG_ORDER] for k, v in out.items()}


def generation_by_pt(baseline_output, shock_output, *, id2Pt, id2EA=None,
                     regions=None, year=ANALYSIS_YEAR):
    """Generation (TWh) per region x aggregated plant type, both states."""
    return _stack_frame(producer_generation, baseline_output, shock_output,
                        id2Pt=id2Pt, id2EA=id2EA, regions=regions, year=year,
                        base_col="q_base", shock_col="q_shock", scale=1.0)


def profit_by_pt(baseline_output, shock_output, *, id2Pt, id2EA=None,
                 regions=None, year=ANALYSIS_YEAR):
    """Producer profit (mia. kr.) per region x aggregated plant type, both states."""
    return _stack_frame(producer_profit, baseline_output, shock_output,
                        id2Pt=id2Pt, id2EA=id2EA, regions=regions, year=year,
                        base_col="profit_base", shock_col="profit_shock", scale=1e-3)


# ==================== ==================== ==================== ====================
# 2. reshape helpers

def _pivot(df, stat):
    """(labels, {state: values}) for one statistic, in region order."""
    labels = list(dict.fromkeys(df.index.get_level_values("region")))
    vals = {s: np.array([df.loc[(r, s), stat] for r in labels], dtype=float) for s in STATES}
    return labels, vals


def _pt_agg(s, *, aggs=PT_AGGS, drop=PT_DROP, what=""):
    """Collapse a Pt-indexed Series onto PT_AGGS, heat-only types dropped.
    Unmapped types with non-zero values fail loudly."""
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


# ==================== ==================== ==================== ====================
# 3. plot engines

def _plot_levels(df, stat, *, title, ylabel, decimals, figsize, bar_width, legend_y,
                 save_path, show, verbose):
    """Grouped baseline/shock bars over regions."""
    labels, vals = _pivot(df, stat)
    if verbose:
        print(df.round(2))
    return plot_grouped(
        labels, vals, colors=STATE_COLORS, series_labels=STATE_LABELS,
        title=title, ylabel=ylabel, decimals=decimals, figsize=figsize,
        bar_width=bar_width, legend_y=legend_y, save_path=save_path, show=show,
    )


def _plot_stacked_states(frames, *, title, ylabel, decimals, figsize, bar_width,
                         legend_y, save_path, show):
    """Baseline | shock stacked bars paired per region, net marked per bar."""
    regions = list(frames["baseline"].index)
    comps = [c for c in frames["baseline"].columns
             if max(frames[s][c].abs().max() for s in STATES) > 0]
    x = np.arange(len(regions), dtype=float)

    fig, ax = new_figure(figsize=figsize)
    spans, nets, handles = [], [], {}
    for i, state in enumerate(STATES):
        pos = x + (i - 0.5) * bar_width
        up, dn, h = stack_bars(ax, pos, frames[state], comps, PT_AGG_COLORS, bar_width,
                               alpha=1.0 if state == "baseline" else 0.72)
        for c, b in h.items():
            handles.setdefault(c, b)
        spans += [up, dn]
        nets.append((pos, up + dn))

    y_lims = limits(np.concatenate(spans), pad_frac=0.18)
    for pos, net in nets:                       # net is not the visual top when signs mix
        net_markers(ax, pos, net, y_lims[1] - y_lims[0], decimals=decimals, signed=False)

    ax.set_ylim(*y_lims)
    style_panel(ax, regions, x, ylabel=ylabel)
    if y_lims[0] < 0:
        zero_line(ax)
    stack_legend(fig, ax, handles, comps, FIG_TEXT["total_label"], legend_y=legend_y)
    finish(fig, title=title, save_path=save_path, show=show)
    return fig, ax


def _plot_stacked_delta(df, *, title, ylabel, decimals, figsize, bar_width, legend_y,
                        save_path, show, hatch_keys=(), colors=None, total_label=None):
    """Signed stacked bars, one per region, total marked. Components in
    `hatch_keys` (partner-origin bands) are hatched."""
    colors = area_colors if colors is None else colors
    total_label = TRADE_COMP_LABELS["net"] if total_label is None else total_label
    regions = list(df.index)
    comps = [c for c in df.columns if df[c].abs().max() > 0]
    x = np.arange(len(regions), dtype=float)

    fig, ax = new_figure(figsize=figsize)
    up, dn, handles = stack_bars(ax, x, df, comps, colors, bar_width, hatch_keys=hatch_keys)
    total = df[comps].sum(axis=1).to_numpy(dtype=float)
    y_lims = limits(np.concatenate([up, dn]), pad_frac=0.18)
    net_markers(ax, x, total, y_lims[1] - y_lims[0], decimals=decimals, signed=True)

    ax.set_ylim(*y_lims)
    style_panel(ax, regions, x, ylabel=ylabel)
    zero_line(ax)
    stack_legend(fig, ax, handles, comps, total_label, legend_y=legend_y)
    finish(fig, title=title, save_path=save_path, show=show)
    return fig, ax


# ==================== ==================== ==================== ====================
# 4. figures: levels

def plot_price_levels(ctx, regions=None, *, years=ANALYSIS_YEAR, figsize=FIGSIZE_WIDE,
                      bar_width=0.38, legend_y=-0.06, decimals=1, save_path=None,
                      show=True, verbose=False):
    """Demand-weighted mean price by region, baseline vs. shock."""
    df = price_levels(ctx, regions, years=years) / 10.0   # DKK/MWh -> øre/kWh
    return _plot_levels(
        df, "mean",
        title=FIG_TEXT["title_price_levels"].format(year=year_label(years)),
        ylabel=FIG_TEXT["unit_price_ore"], decimals=decimals,
        figsize=figsize, bar_width=bar_width, legend_y=legend_y,
        save_path=save_path, show=show, verbose=verbose,
    )


def plot_trade_levels(ctx, regions=None, *, years=ANALYSIS_YEAR, component="net",
                      value=False, figsize=FIGSIZE_WIDE, bar_width=0.38, legend_y=-0.06,
                      decimals=1, save_path=None, show=True, verbose=False):
    """Net export by region, baseline vs. shock; TWh, or mia. kr. if `value`."""
    title = FIG_TEXT["title_trade_v" if value else "title_trade_q"].format(year=year_label(years))
    if component != "net":
        comp = TRADE_COMP_LABELS.get(component, component)
        title = f"{comp}: {title[0].lower()}{title[1:]}"
    return _plot_levels(
        trade_levels(ctx, regions, years=years, component=component),
        "bnDKK" if value else "TWh",
        title=title, ylabel=FIG_TEXT["unit_value" if value else "unit_energy_yr"],
        decimals=decimals, figsize=figsize, bar_width=bar_width, legend_y=legend_y,
        save_path=save_path, show=show, verbose=verbose,
    )


def plot_welfare_levels(baseline_output, shock_output, *, id2Pt=None, id2EA=None,
                        regions=None, year=ANALYSIS_YEAR, markets=("E",), cs_scale=1.0,
                        include_ci=True, annuity=None, series=WELFARE_SERIES,
                        figsize=None, bar_width=None, legend_y=-0.06, decimals=1,
                        save_path=None, show=True):
    """ΔCS, ΔPS and net by region. `include_ci` adds the congestion rent and
    `annuity` the annuitised investment to the net bar; series=WELFARE_SERIES_FULL
    draws them as bars of their own."""
    df = welfare_levels(baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
                        regions=regions, year=year, markets=markets, cs_scale=cs_scale,
                        include_ci=include_ci, annuity=annuity)
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

    return plot_grouped(
        list(df.index), {k: df[k].to_numpy(dtype=float) for k in series},
        colors=WELFARE_COLORS, series_labels=WELFARE_COMP_LABELS,
        title=FIG_TEXT["title_welfare"].format(year=year_label(year), note=note),
        ylabel=FIG_TEXT["unit_value"], decimals=decimals, signed=True,
        edge_keys=("net",), figsize=figsize,
        bar_width=bar_width if bar_width else round(0.78 / len(series), 3),
        legend_y=legend_y, save_path=save_path, show=show,
    )


def plot_generation_stack(baseline_output, shock_output, *, id2Pt, id2EA=None, regions=None,
                          year=ANALYSIS_YEAR, figsize=FIGSIZE_WIDE, bar_width=0.38,
                          legend_y=-0.06, decimals=1, save_path=None, show=True):
    """Electricity generation (TWh) by plant type, baseline | shock."""
    return _plot_stacked_states(
        generation_by_pt(baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
                         regions=regions, year=year),
        title=FIG_TEXT["title_gen_stack"].format(year=year_label(year)),
        ylabel=FIG_TEXT["unit_energy_yr"], decimals=decimals, figsize=figsize,
        bar_width=bar_width, legend_y=legend_y, save_path=save_path, show=show,
    )


def plot_profit_stack(baseline_output, shock_output, *, id2Pt, id2EA=None, regions=None,
                      year=ANALYSIS_YEAR, figsize=FIGSIZE_WIDE, bar_width=0.38,
                      legend_y=-0.06, decimals=1, save_path=None, show=True):
    """Producer profit (mia. kr.) by plant type, baseline | shock."""
    return _plot_stacked_states(
        profit_by_pt(baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
                     regions=regions, year=year),
        title=FIG_TEXT["title_profit_stack"].format(year=year_label(year)),
        ylabel=FIG_TEXT["unit_value"], decimals=decimals, figsize=figsize,
        bar_width=bar_width, legend_y=legend_y, save_path=save_path, show=show,
    )


# ==================== ==================== ==================== ====================
# 5. figures: deltas

def plot_price_delta(ctx, regions=None, *, years=ANALYSIS_YEAR, stats=("mean",),
                     relative=False, figsize=(12, 5), bar_width=0.55, decimals=1,
                     save_path=None, show=True, verbose=False):
    """Shock - baseline price by region, one panel per `price_summary` statistic."""
    df = price_levels(ctx, regions, years=years) / 10.0   # DKK/MWh -> øre/kWh
    stats = (stats,) if isinstance(stats, str) else tuple(stats)

    fig, axes = new_figure(1, len(stats), figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for ax, stat in zip(axes, stats):
        labels, vals = _pivot(df, stat)
        base, shock = vals["baseline"], vals["shock"]
        d = 100.0 * (shock - base) / base if relative else shock - base
        signed_bars(ax, labels, d, [DELTA_COLORS["up" if v >= 0 else "down"] for v in d],
                    bar_width=bar_width, decimals=decimals,
                    ylabel=FIG_TEXT["unit_pct" if relative else "unit_price_ore"])
        ax.set_title(PRICE_STAT_LABELS.get(stat, stat), loc="left", pad=2)
        if verbose:
            print(stat, dict(zip(labels, np.round(d, 3))))

    finish(fig, show=show, save_path=save_path, rect=(0, 0.02, 1, 1.05),
           title=FIG_TEXT["title_price_delta"].format(
               year=year_label(years),
               unit=FIG_TEXT["delta_rel" if relative else "delta_abs"],
               note=FIG_TEXT["delta_note"]))
    return fig, tuple(axes)


def plot_capture_delta_by_pt(ctx, *, country="DK", years=ANALYSIS_YEAR, figsize=FIGSIZE_WIDE,
                             bar_width=0.55, decimals=1, save_path=None, show=True):
    """Δ capture price by aggregated plant type for one region."""
    d = capture_delta_by_pt(ctx, country=country, years=years) / 10.0
    labels = list(d.index)
    fig, ax = new_figure(figsize=figsize)
    signed_bars(ax, labels, d.to_numpy(dtype=float),
                [PT_AGG_COLORS.get(l, MISSING_COLOR) for l in labels],
                bar_width=bar_width, decimals=decimals, ylabel=FIG_TEXT["unit_price_ore"])
    finish(fig, save_path=save_path, show=show,
           title=FIG_TEXT["title_capture_delta"].format(
               region=col_names.get(country, country), year=year_label(years)))
    return fig, ax


def plot_trade_delta_by_partner(ctx, regions=None, *, years=ANALYSIS_YEAR, component="net",
                                value=False, figsize=FIGSIZE_WIDE, bar_width=0.45,
                                legend_y=-0.1, decimals=1, save_path=None, show=True):
    """Δ net export decomposed by trade partner, one stacked bar per region."""
    return _plot_stacked_delta(
        trade_delta_by_partner(ctx, regions, years=years, component=component, value=value),
        title=FIG_TEXT["title_trade_delta_v" if value else "title_trade_delta_q"].format(
            year=year_label(years)),
        ylabel=FIG_TEXT["unit_value" if value else "unit_energy_yr"],
        decimals=decimals, figsize=figsize, bar_width=bar_width,
        legend_y=legend_y, save_path=save_path, show=show,
    )


def plot_nx_delta_by_pt(ctx, share_dict, regions=None, *, years=ANALYSIS_YEAR,
                        figsize=FIGSIZE_WIDE, bar_width=0.45, legend_y=-0.20, decimals=1,
                        save_path=None, show=True):
    """Δ net export decomposed by supply component, one stacked bar per region."""
    return _plot_stacked_delta(
        nx_delta_by_pt(ctx, share_dict, regions, years=years),
        title=FIG_TEXT["title_nx_pt_delta"].format(year=year_label(years)),
        ylabel=FIG_TEXT["unit_energy_yr"], decimals=decimals,
        hatch_keys=set(PARTNER_ORDER), colors=NX_COMP_COLORS, figsize=figsize,
        bar_width=bar_width, legend_y=legend_y, save_path=save_path, show=show,
    )


def plot_generation_delta(baseline_output, shock_output, *, id2Pt, id2EA=None, regions=None,
                          year=ANALYSIS_YEAR, figsize=FIGSIZE_WIDE, bar_width=0.45,
                          legend_y=-0.10, decimals=1, save_path=None, show=True):
    """Δ electricity generation (TWh) decomposed by plant type."""
    f = generation_by_pt(baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
                         regions=regions, year=year)
    return _plot_stacked_delta(
        f["shock"] - f["baseline"],
        title=FIG_TEXT["title_gen_delta"].format(year=year_label(year)),
        ylabel=FIG_TEXT["unit_energy_yr"], decimals=decimals,
        colors=PT_AGG_COLORS, figsize=figsize, bar_width=bar_width,
        legend_y=legend_y, save_path=save_path, show=show,
        total_label=FIG_TEXT["total_label"],
    )


def plot_profit_delta(baseline_output, shock_output, *, id2Pt, id2EA=None, regions=None,
                      year=ANALYSIS_YEAR, figsize=FIGSIZE_WIDE, bar_width=0.45,
                      legend_y=-0.10, decimals=1, save_path=None, show=True):
    """Δ producer profit (mia. kr.) decomposed by plant type."""
    f = profit_by_pt(baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
                     regions=regions, year=year)
    return _plot_stacked_delta(
        f["shock"] - f["baseline"],
        title=FIG_TEXT["title_profit_delta"].format(year=year_label(year)),
        ylabel=FIG_TEXT["unit_value"], decimals=decimals,
        colors=PT_AGG_COLORS, figsize=figsize, bar_width=bar_width,
        legend_y=legend_y, save_path=save_path, show=show,
        total_label=FIG_TEXT["total_label"],
    )


def plot_cs_by_price_bin(baseline_output, shock_output, regions=None, *, year=ANALYSIS_YEAR,
                         n_bins=5, cs_scale=1.0, figsize=FIGSIZE_WIDE, bar_width=None,
                         legend_y=-0.12, decimals=0, save_path=None, show=True,
                         verbose=False):
    """ΔCS by equal-hour bins of the baseline price (cheapest -> dearest), one series
    per region. Bins sum to the ΔCS of `plot_welfare_levels` (markets=("E",))."""
    df = cs_by_price_bin(baseline_output, shock_output, regions, year=year,
                         n_bins=n_bins, cs_scale=cs_scale)
    if verbose:
        print(df.assign(total=df.sum(axis=1)).round(3))

    fig, ax = new_figure(figsize=figsize)
    handles, leg_labels = draw_grouped(
        ax, list(df.columns), {r: df.loc[r].to_numpy(dtype=float) for r in df.index},
        colors={r: area_colors.get(r, MISSING_COLOR) for r in df.index},
        series_labels=col_names, ylabel=FIG_TEXT["unit_value_mio"], decimals=decimals,
        signed=True, bar_width=bar_width if bar_width else round(0.78 / len(df), 3),
    )
    ax.set_xlabel(FIG_TEXT["price_bin_axis"], loc="left")
    bottom_legend(fig, handles, leg_labels, legend_y=legend_y, ncol=min(len(df), 4))
    finish(fig, title=FIG_TEXT["title_cs_price_bin"].format(year=year_label(year)),
           save_path=save_path, show=show)
    return fig, ax
