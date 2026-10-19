"""Elektricitets- og fjernvarmebalancefigurer.

Plotlag til energibalance-pipelinen: læser de hFull-ekspanderede frames fra
``energy_balance_build`` (``supply_dict`` / ``demand_dict`` / ``NX_dict``) og
tegner basis-vs-stød stakkede arealfigurer, nettoeksportpaneler og
nettoeksport fordelt på anlægstype.

Dict-nøglerne er og bliver engelske ("baseline"/"shock") -- de kommer fra
build-laget og deles med summaries/scarcity/partial_welfare. Dansk sprog er
udelukkende et rendering-lag: se ``STATE_LABELS`` og ``FIG_TEXT`` i groups.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .groups import (
    META_COLS, META_COLS_HA, COL_ORDER, PT_COLS, EA_ORDER, NUCLEAR_PT, c_aggs_countries,
    DEFAULT_LABELS, DEFAULT_COLORS, MISSING_COLOR, BG_COLOR, PANEL_EDGE, PANEL_EDGE_LW,
    STATE_BASE, STATE_SHOCK, STATES, STATE_LABELS, FIG_TEXT,
)
from .energy_balance_build import ElBalanceData  # runtime: isinstance() i build_supply_shares


# =========================================================================== #
# 1. kontekst og markedsdefinitioner
# =========================================================================== #
@dataclass
class ElBalanceContext:
    """Alt plotfunktionerne skal bruge, samlet så kaldstederne holdes korte"""

    supply_dict: Mapping[str, pd.DataFrame]
    demand_dict: Mapping[str, pd.DataFrame]
    NX_dict: Mapping[str, pd.DataFrame]
    area_colors: Mapping[str, str]
    col_names: Mapping[str, str]
    EA_cols: Iterable[str]                          # serier der tegnes med skravering
    db_dict: Mapping[str, object] | None = None
    state_labels: Mapping[str, str] | None = None   # None -> STATE_LABELS

    @classmethod
    def from_data(cls, data: ElBalanceData, area_colors, col_names, EA_cols,
                  state_labels=None):
        """Byg en plotkontekst fra :func:`build_el_balance_data`-output."""
        return cls(
            supply_dict=data.supply_dict,
            demand_dict=data.demand_dict,
            NX_dict=data.NX_dict,
            area_colors=area_colors,
            col_names=col_names,
            EA_cols=EA_cols,
            db_dict=data.db_dict,
            state_labels=state_labels,
        )

    @classmethod
    def from_dicts(cls, supply_dict, demand_dict, NX_dict, area_colors, col_names,
                   EA_cols, state_labels=None):
        """Byg en kontekst direkte fra de tre DataFrame-dicts (fx efter unpickling)
        -- kræver hverken ``db_dict`` eller en levende GpyDB."""
        return cls(supply_dict, demand_dict, NX_dict, area_colors, col_names,
                   EA_cols, state_labels=state_labels)


@dataclass(frozen=True)
class _Market:
    """Det der adskiller el-diagrammet fra fjernvarmediagrammet."""
    area_key: str        # "EA" / "HA"
    price: str           # "pE" / "pH"
    price_base: str      # "pE_baseline" / "pH_baseline"
    price_label: str
    meta_cols: tuple     # kolonner der IKKE er plotbare serier
    has_trade: bool      # el har handel mellem områder; varme har ikke
    title_key: str       # FIG_TEXT-nøgle til suptitle


EL_MARKET = _Market(
    "EA", "pE", "pE_baseline", FIG_TEXT["ylabel_price_el"],
    tuple(META_COLS), True, "title_balance_el",
)
DH_MARKET = _Market(
    "HA", "pH", "pH_baseline", FIG_TEXT["ylabel_price_dh"],
    tuple(META_COLS_HA), False, "title_balance_dh",
)


# =========================================================================== #
# 2. små hjælpere
# =========================================================================== #
def _state_label(state, state_labels=None):
    """Dansk visningsnavn for en scenarietilstand; falder tilbage til rå nøgle."""
    labels = STATE_LABELS if state_labels is None else state_labels
    return labels.get(state, state)


def _value_cols(df, meta) -> list:
    return [c for c in df.columns if c not in meta]


def _for_area(df, area, years, area_key):
    return df[(df[area_key] == area) & (df["t"].isin(years))]


def _label(col, col_names):
    return col_names[col] if col in col_names else DEFAULT_LABELS.get(col, col)


def _color(col, area_colors):
    return area_colors[col] if col in area_colors else DEFAULT_COLORS.get(col, MISSING_COLOR)


def _area_universe(nx):
    """Alle områdenavne, læst af nettoeksport-indekset (intet db_dict behøves)."""
    return sorted(
        set(nx.index.get_level_values("EA")) | set(nx.index.get_level_values("EAEA"))
    )


def _drop_unconnected_eas(df, all_eas, connected):
    drop = [e for e in all_eas if e not in connected]
    return df.drop(columns=[c for c in drop if c in df.columns])


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
    value = min(extremes) if agg == "min" else max(extremes)
    return value * margin


def _price_extent(supply_dict, area, years, market):
    prices = [
        _for_area(supply_dict[s], area, years, market.area_key)[market.price]
        for s in STATES
        if not _for_area(supply_dict[s], area, years, market.area_key).empty
    ]
    if not prices:
        return 0.0, 1.0
    allp = pd.concat(prices)
    return min(allp.min(), 0.0) * 1.05, allp.max() * 1.05


def _series_annual_twh(supply_dict, demand_dict, area, year, series, market):
    """Årlig energi pr. serie (TWh), største numeriske værdi på tværs af begge
    tilstande og begge sider. Rækkerne er rigtige timer (hFull) i GWh pr. time,
    så årets energi er den rene sum / 1000 (ingen hWeight)."""
    totals = {s: 0.0 for s in series}
    for data_dict in (supply_dict, demand_dict):
        for state in STATES:
            sub = _for_area(data_dict[state], area, [year], market.area_key)
            if sub.empty:
                continue
            for s in series:
                if s in sub.columns:
                    totals[s] = max(totals[s], abs(sub[s].sum()) / 1000.0)
    return totals


# =========================================================================== #
# 3. balancefigur (el + fjernvarme)
# =========================================================================== #
def _plot_balance(
    supply_dict, demand_dict, nx_dict, area, *,
    market, area_colors, col_names, hatch_cols,
    years, sort_by, legend_min_twh, figsize, save_path_fmt, show, verbose,
    legend_y=-0.3, only=None, align_zero=False, state_labels=None,
):
    """Fælles motor: basisforløb vs. stødforløb side om side for ét område/marked."""
    if years is None:
        years = sorted(supply_dict[STATE_BASE]["t"].unique())
    elif isinstance(years, int):
        years = [years]

    price_lims = _price_extent(supply_dict, area, years, market)

    if market.has_trade and nx_dict is not None:
        connected = _connected_eas(nx_dict, area)
        all_eas = _area_universe(nx_dict[STATE_BASE])
        prep = lambda df: _drop_unconnected_eas(
            _for_area(df, area, years, market.area_key), all_eas, connected
        )
    else:
        prep = lambda df: _for_area(df, area, years, market.area_key)

    supply = {s: prep(supply_dict[s]) for s in STATES}
    demand = {s: prep(demand_dict[s]) for s in STATES}

    series = _value_cols(supply[STATE_BASE], market.meta_cols)
    # Kun serier med indhold for området i mindst én tilstand (ellers ~50 tomme lag og legendeposter).
    series = [s for s in series
              if any(supply[st][s].abs().sum() + demand[st][s].abs().sum() > 0 for st in STATES)]
    if only is not None:
        series = [s for s in series if s in only]   # behold rækkefølge/farver; kun subset
    colors = [_color(c, area_colors) for c in series]

    if NUCLEAR_PT in series:                        # kernekraft -> bunden af stakken
        i = series.index(NUCLEAR_PT)
        series = [series[i], *series[:i], *series[i + 1:]]
        colors = [colors[i], *colors[:i], *colors[i + 1:]]

    hatched = set(hatch_cols)

    if only is not None:
        sup_tot = max(supply[s][series].sum(axis=1).max() for s in STATES)
        y_lims = (0.0, sup_tot * 1.05)
    else:
        y_lims = (
            _stacked_extent(demand_dict, area, years, "min", market),
            _stacked_extent(supply_dict, area, years, "max", market),
        )

    results = []
    for year in years:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        fig.patch.set_facecolor(BG_COLOR)  # udfyldt lærred, ingen hvid ramme

        handles = labels = None
        for col, (ax, state) in enumerate(zip(axes, STATES)):
            order = ["hFull"] if sort_by == "hFull" else [sort_by, "hFull"]
            sup_t = (
                supply[state][supply[state]["t"] == year]
                .sort_values(order).reset_index(drop=True)
            )
            dem_t = (
                demand[state][demand[state]["t"] == year]
                .sort_values(order).reset_index(drop=True)
            )
            h, lab = _draw_panel(
                ax, sup_t, dem_t, series, colors, hatched,
                market=market, state=state, y_lims=y_lims, price_lims=price_lims,
                col_names=col_names, show_ylabel=(col == 0), show_price_axis=(col == 1),
                align_zero=align_zero, state_labels=state_labels,
            )
            if handles is None:  # panelerne deler serier -> hent kun én gang
                handles, labels = h, lab

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(PANEL_EDGE)
                spine.set_linewidth(PANEL_EDGE_LW)

        annual = _series_annual_twh(supply_dict, demand_dict, area, year, series, market)
        if verbose:
            print(f"{area} {year} årlig TWh:",
                  {s: round(annual[s], 4) for s in series})
        keep = [annual[s] >= legend_min_twh for s in series]
        leg_handles = [hd for hd, k in zip(handles, keep) if k]
        leg_labels = [lb for lb, k in zip(labels, keep) if k]

        if leg_labels:
            fig.legend(
                leg_handles, leg_labels,
                loc="lower center", bbox_to_anchor=(0.5, legend_y),
                ncol=min(len(leg_labels), 3), frameon=False, fontsize=11,
            )
        fig.suptitle(
            FIG_TEXT[market.title_key].format(
                area=col_names.get(area, area), year=year
            ),
            x=0.124, ha="left", fontweight="bold",
        )

        fig.tight_layout(rect=[0, 0.06, 1, 0.0])
        results.append((fig, tuple(axes)))
        if save_path_fmt:
            fig.savefig(save_path_fmt.format(area=area, year=year), dpi=300,
                        facecolor=fig.get_facecolor())
        if show:
            plt.show()

    return results


def _draw_panel(
    ax, sup_t, dem_t, series, colors, hatched,
    *, market, state, y_lims, price_lims, col_names, show_ylabel, show_price_axis,
    align_zero=False, state_labels=None,
):
    """Tegn én tilstands balance i ``ax``; returnerer (handles, labels)."""
    ax.set_facecolor(BG_COLOR)                      # fyld, matcher figuren
    for spine in ax.spines.values():                # drop den kasseformede ramme
        spine.set_visible(False)

    # udbud (positivt) og derefter efterspørgsel (negativt), samme farver/rækkefølge
    sup_t[series].plot(
        ax=ax, kind="area", stacked=True, linewidth=0.0, ylim=y_lims, color=colors
    )
    handles, _ = ax.get_legend_handles_labels()  # kun fra udbudslaget
    labels = [_label(s, col_names) for s in series]

    dem_t[series].plot(
        ax=ax, kind="area", stacked=True, linewidth=0.0, ylim=y_lims,
        color=colors, legend=False,
    )

    # skravér handelsserierne (kun el; tom for varme)
    for collection, col in zip(ax.collections, series * 2):
        if col in hatched:
            collection.set_hatch("//")
            collection.set_edgecolor("#666666")

    ax.axhline(0, color="0.2", lw=1, ls="--")
    ax.grid(lw=0.6, alpha=0.35)
    ax.tick_params()
    if show_ylabel:
        ax.set_ylabel(FIG_TEXT["unit_energy"], loc="bottom")
    ax.set_xlabel(FIG_TEXT["xlabel_hours_price"], loc="left")
    if len(sup_t):
        ax.set_xlim(0, len(sup_t) - 1)

    l0, l1 = y_lims
    if align_zero and l0 < 0 < l1 and price_lims[1] > 0:
        f = (0 - l0) / (l1 - l0)
        p_hi = price_lims[1]
        price_ylim = (f * p_hi / (f - 1), p_hi) if f < 0.999 else price_lims
    else:
        price_ylim = price_lims

    ax2 = ax.twinx()
    ax2.patch.set_visible(False)                    # lad panelfyldet skinne igennem
    for spine in ax2.spines.values():               # drop også tvillingens ramme
        spine.set_visible(False)
    sup_t[market.price_base].plot(ax=ax2, ylim=price_ylim, color="#0D0AF5", legend=False)
    sup_t[market.price].plot(ax=ax2, ylim=price_ylim, color="#0D0AF5", style="--", legend=False)
    ax2.set_ylim(price_ylim)
    if show_price_axis:
        ax2.set_ylabel(market.price_label, loc="bottom")
        if align_zero:
            ax2.set_yticks([t for t in ax2.get_yticks() if t >= 0])  # skjul negative pristicks
    else:
        ax2.tick_params(right=False, labelright=False)
    ax2.grid(False)

    if ax.get_legend() is not None:  # drop panellegenden; vi bruger en fælles
        ax.get_legend().remove()
    ax.set_title(_state_label(state, state_labels), loc="left")
    return handles, labels


def plot_energy_balance_el(
    ctx: ElBalanceContext,
    ea: str,
    *,
    years: Sequence | int | None = None,
    sort_by: str = "pE_baseline",
    legend_min_twh: float = 0.0,
    figsize: tuple = (12, 5),
    save_path_fmt: str | None = None,
    show: bool = True,
    verbose: bool = False,
    legend_y: float = -0.3,
    only=None,
    align_zero: bool = False,
    state_labels=None,
):
    """Elbalance, basisforløb vs. stødforløb (1x2) for ét område `ea`.

    Returnerer ``[(fig, (ax_basis, ax_stød)), ...]`` -- én post pr. år.
    """
    return _plot_balance(
        ctx.supply_dict, ctx.demand_dict, ctx.NX_dict, ea,
        market=EL_MARKET, area_colors=ctx.area_colors, col_names=ctx.col_names,
        hatch_cols=ctx.EA_cols, years=years, sort_by=sort_by,
        legend_min_twh=legend_min_twh, figsize=figsize,
        save_path_fmt=save_path_fmt, show=show, verbose=verbose,
        legend_y=legend_y, only=only, align_zero=align_zero,
        state_labels=state_labels or ctx.state_labels,
    )


def plot_energy_balance_dh(
    ctx: ElBalanceContext,
    ha: str,
    *,
    years: Sequence | int | None = None,
    sort_by: str = "pH_baseline",
    legend_min_twh: float = 0.0,
    figsize: tuple = (12, 5),
    save_path_fmt: str | None = None,
    show: bool = True,
    verbose: bool = False,
    legend_y: int = -0.2,
    state_labels=None,
):
    """Fjernvarmebalance, basisforløb vs. stødforløb (1x2) for ét varmeområde `ha`"""
    return _plot_balance(
        ctx.supply_dict, ctx.demand_dict, None, ha,
        market=DH_MARKET, area_colors=ctx.area_colors, col_names=ctx.col_names,
        hatch_cols=set(), years=years, sort_by=sort_by,
        legend_min_twh=legend_min_twh, figsize=figsize,
        save_path_fmt=save_path_fmt, show=show, verbose=verbose,
        legend_y=legend_y, state_labels=state_labels or ctx.state_labels,
    )


# =========================================================================== #
# 4. nettoeksportfigur (kun el)
# =========================================================================== #
def _nx_partner_order(NX_dict, ea):
    """Forbundne handelspartnere til `ea`, ordnet efter COL_ORDER."""
    connected = _connected_eas(NX_dict, ea)
    return [x for x in COL_ORDER if x in connected]


def _nx_extent(NX_dict, ea, years, margin=1.05):
    """(min, max) for de stakkede import-/eksporttotaler på tværs af tilstande (GWh)."""
    mins, maxs = [], []
    for state in STATES:
        df = NX_dict[state]
        mask = df.index.get_level_values("t").isin(years) & (
            df.index.get_level_values("EA") == ea
        )
        le = df.loc[mask, "lineExport"]
        if le.empty:
            continue
        maxs.append(le.clip(lower=0).groupby(level=["t", "hFull"]).sum().max())
        mins.append(le.clip(upper=0).groupby(level=["t", "hFull"]).sum().min())
    lo = (min(mins) * margin if mins else 0.0) * 1000  # match balancefigurens GWh-skala
    hi = (max(maxs) * margin if maxs else 0.0) * 1000
    return lo, hi


def _nx_price_extent(NX_dict, ea, years, margin=1.05):
    """(min, max) til prisaksen på tværs af begge tilstande."""
    prices = []
    for state in STATES:
        df = NX_dict[state]
        mask = df.index.get_level_values("t").isin(years) & (
            df.index.get_level_values("EA") == ea
        )
        if mask.any():
            prices.append(df.loc[mask, "pE_EA"])
    if not prices:
        return 0.0, 1.0
    allp = pd.concat(prices)
    return min(allp.min(), 0.0) * margin, allp.max() * margin


def _align_zero(ax_left, ax_right):
    """Placér 0 i samme højde på begge akser, med hver akses spænd bevaret."""
    l0, l1 = ax_left.get_ylim()
    if l1 == l0:
        return
    f = (0 - l0) / (l1 - l0)          # andel af venstre akse under nul
    r0, r1 = ax_right.get_ylim()
    span = r1 - r0
    new_r0 = 0 - f * span             # flyt prisvinduet så dets 0 ligger ved f
    ax_right.set_ylim(new_r0, new_r0 + span)


def _draw_nx_panel(ax, pos, neg, partners, colors, *, state, y_lims, price_lims,
                   price_baseline, price, col_names, show_ylabel, show_price_axis,
                   state_labels=None):
    """Tegn én tilstands nettoeksport i ``ax``; returnerer (handles, labels)."""
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)

    pos[partners].plot(
        ax=ax, kind="area", stacked=True, linewidth=0.0, ylim=y_lims, color=colors
    )
    handles, _ = ax.get_legend_handles_labels()  # fra eksportlaget (positivt)
    labels = [_label(p, col_names) for p in partners]

    neg[partners].plot(
        ax=ax, kind="area", stacked=True, linewidth=0.0, ylim=y_lims,
        color=colors, legend=False,
    )

    ax.axhline(0, color="0.2", lw=1, ls="--")
    ax.grid(lw=0.6, alpha=0.35)
    ax.tick_params()
    if show_ylabel:
        ax.set_ylabel(FIG_TEXT["unit_energy"], loc="bottom")
    ax.set_xlabel(FIG_TEXT["xlabel_hours_price"], loc="left")
    if len(pos):
        ax.set_xlim(0, len(pos) - 1)

    # pris på sekundæraksen (matcher balancefigurerne)
    ax2 = ax.twinx()
    ax2.patch.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    price_baseline.plot(ax=ax2, ylim=price_lims, color="#0D0AF5", legend=False)
    price.plot(ax=ax2, ylim=price_lims, color="#0D0AF5", style="--", legend=False)
    if show_price_axis:
        ax2.set_ylabel(FIG_TEXT["ylabel_price_el"], loc="bottom")
    else:
        ax2.tick_params(right=False, labelright=False)
    ax2.grid(False)

    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_title(_state_label(state, state_labels), loc="left")
    return handles, labels


def plot_net_export(
    ctx: ElBalanceContext,
    ea: str,
    *,
    years: Sequence | int | None = None,
    sort_by: str = "pE_EA_baseline",
    figsize: tuple = (12, 5),
    legend_y: float = -0.1,
    save_path_fmt: str | None = None,
    show: bool = True,
    state_labels=None,
):
    """Nettoeksport af el, basisforløb vs. stødforløb (1x2), for ét område `ea`"""
    NX = ctx.NX_dict
    if NX is None:
        raise ValueError("plot_net_export kræver NX_dict (byg med build_el_balance_data).")
    labels_map = state_labels or ctx.state_labels

    if years is None:
        years = sorted(NX[STATE_BASE].index.get_level_values("t").unique())
    elif isinstance(years, int):
        years = [years]

    partners = _nx_partner_order(NX, ea)
    colors = [_color(p, ctx.area_colors) for p in partners]
    y_lims = _nx_extent(NX, ea, years)
    price_lims = _nx_price_extent(NX, ea, years)

    results = []
    for year in years:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        fig.patch.set_facecolor(BG_COLOR)

        handles = labels = None
        for col, (ax, state) in enumerate(zip(axes, STATES)):
            sub = NX[state].xs((year, ea), level=["t", "EA"]).sort_values([sort_by, "hFull"])
            le = sub["lineExport"]
            pos = (le.clip(lower=0).unstack("EAEA", sort=False).reset_index(drop=True)
                   .reindex(columns=partners, fill_value=0) * 1000)
            neg = (le.clip(upper=0).unstack("EAEA", sort=False).reset_index(drop=True)
                   .reindex(columns=partners, fill_value=0) * 1000)
            price_b = (sub["pE_EA_baseline"].groupby(level="hFull", sort=False)
                       .first().reset_index(drop=True))
            price_s = (sub["pE_EA"].groupby(level="hFull", sort=False)
                       .first().reset_index(drop=True))
            h, lab = _draw_nx_panel(
                ax, pos, neg, partners, colors,
                state=state, y_lims=y_lims, price_lims=price_lims,
                price_baseline=price_b, price=price_s,
                col_names=ctx.col_names,
                show_ylabel=(col == 0), show_price_axis=(col == 1),
                state_labels=labels_map,
            )
            if handles is None:
                handles, labels = h, lab

        for ax in axes:  # diskret panelkant (matcher balancefigurerne)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(PANEL_EDGE)
                spine.set_linewidth(PANEL_EDGE_LW)

        if labels:
            fig.legend(
                handles, labels,
                loc="lower center", bbox_to_anchor=(0.5, legend_y),
                ncol=min(len(labels), 3), frameon=False, fontsize=11,
            )
        fig.suptitle(
            FIG_TEXT["title_net_export"].format(area=ctx.col_names.get(ea, ea), year=year),
            x=0.124, ha="left", fontweight="bold",
        )

        fig.tight_layout(rect=[0, 0.06, 1, 0.0])
        results.append((fig, tuple(axes)))
        if save_path_fmt:
            fig.savefig(save_path_fmt.format(area=ea, year=year), dpi=300,
                        facecolor=fig.get_facecolor())
        if show:
            plt.show()

    return results


# =========================================================================== #
# 5. nettoeksport *pr. anlægstype* (EA eller land) -- attribuering af linjeflows
# =========================================================================== #
def build_supply_shares(data, *, aggregates: dict | None = None):
    """Forsyningssammensætning som andele pr. (t,EA,h,hFull), inkl. import, for
    alle områder, plus ét aggregat pr. post i `aggregates` (label -> medlemszoner)."""
    if aggregates is None:
        aggregates = c_aggs_countries
    supply_dict = data.supply_dict if isinstance(data, ElBalanceData) else data
    meta = set(META_COLS) | {"pE_baseline"}
    shares = {}
    for state, df in supply_dict.items():
        pcols = [c for c in PT_COLS if c in df.columns]          # anlægstyper
        tcols = [c for c in EA_ORDER if c in df.columns]         # handel (import)
        keep = ["t", "EA", "h", "hFull"]
        base = df[keep + pcols + tcols].copy()

        extra = []
        for label, members in aggregates.items():
            ext_tcols = [c for c in tcols if c not in members]   # drop intern handel
            agg = base[base["EA"].isin(members)].groupby(
                ["t", "h", "hFull"], as_index=False
            )[pcols + ext_tcols].sum()
            agg["EA"] = label
            extra.append(agg.reindex(columns=keep + pcols + tcols, fill_value=0))

        full = pd.concat([base] + extra, ignore_index=True)
        comp = pcols + tcols
        tot = full[comp].sum(axis=1)
        full[comp] = full[comp].div(tot.where(tot != 0, 1.0), axis=0)  # skala går ud
        shares[state] = full.set_index(["t", "EA", "h", "hFull"])
    return shares


def _aggregate_partner(df, label):
    """Kollaps partnerzoner i `df` til én `label`-partner"""
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
    """Nettoeksport EA->EAEA opdelt på forsyningskomponent"""
    lvl = share_dict[STATE_BASE].index.get_level_values("EA")
    if ea not in lvl:
        raise KeyError(f"{ea} findes ikke i share_dict (byg det fra supply_dict pr. EA).")
    if eaea not in lvl:
        raise KeyError(f"{eaea} findes ikke i share_dict -- tilføj via aggregates= i build_supply_shares.")
    partner_zones = list((partner_map or {}).get(eaea, [eaea]))

    comp = list(share_dict[STATE_BASE].columns)
    out = {}
    for state in nx_dict:
        nx = nx_dict[state].reset_index()
        nx = nx[(nx["EA"] == ea) & (nx["EAEA"].isin(partner_zones))].copy()
        if len(partner_zones) > 1:                      # slå partnerzoner sammen -> land
            nx = _aggregate_partner(nx, eaea)

        sh_ea = share_dict[state].xs(ea, level="EA").reset_index()[["t", "hFull"] + comp]
        sh_pa = share_dict[state].xs(eaea, level="EA").reset_index()[["t", "hFull"] + comp]
        exp = nx.merge(sh_ea, on=["t", "hFull"], how="left")
        imp = nx.merge(sh_pa, on=["t", "hFull"], how="left")
        for c in comp:
            exp[c] = (exp[c].clip(lower=0) * exp["lineExport"]).clip(lower=0)  # op
            imp[c] = (imp[c].clip(lower=0) * imp["lineExport"]).clip(upper=0)  # ned
        out[state] = (exp, imp)
    return out


def build_nx_by_pt_agg(nx_dict, share_dict, ea, partners, *, partner_map=None,
                       partner_label="ALL"):
    """Nettoeksport EA -> alle `partners`, summeret pr. time, pr. forsyningskomponent.
    Hver partners importflow dekomponeres med sit eget miks før summen."""
    comp = list(share_dict[STATE_BASE].columns)
    per_partner = {
        p: build_nx_by_pt(nx_dict, share_dict, ea, p, partner_map=partner_map)
        for p in partners
    }
    out = {}
    for state in nx_dict:
        frames = []
        for side, clip in ((0, "lower"), (1, "upper")):     # 0 = eksport, 1 = import
            df = pd.concat(
                [per_partner[p][state][side] for p in partners], ignore_index=True
            )
            df["lineExport"] = df["lineExport"].clip(**{clip: 0})
            agg = df.groupby(["t", "hFull"], as_index=False).agg(
                {**{c: "sum" for c in comp}, "lineExport": "sum",
                 "pE_EA": "first", "pE_EA_baseline": "first"}
            )
            agg["EA"], agg["EAEA"] = ea, partner_label
            frames.append(agg)
        out[state] = tuple(frames)
    return out


def plot_net_export_by_pt_agg(
    ctx: ElBalanceContext, ea: str, partners, *, share_dict, partner_map=None,
    partner_label: str = "ALL",
    years: Sequence | int | None = None, sort_by: str = "pE_EA_baseline",
    legend_min_twh: float = 0.0,
    legend_min_share: float = 0.0,
    legend_max_n: int | None = None,
    figsize: tuple = (12, 5), legend_y: float = -0.3,
    save_path_fmt: str | None = None, show: bool = True, state_labels=None,
):
    """Samlet nettoeksport EA -> alle `partners`, opdelt på forsyningskomponent,
    basis vs. stød (1x2). Aggregeret modstykke til plot_net_export_by_pt."""
    NX = ctx.NX_dict
    if NX is None:
        raise ValueError("plot_net_export_by_pt_agg kræver en el-kontekst (NX_dict).")
    labels_map = state_labels or ctx.state_labels
    if years is None:
        years = sorted(NX[STATE_BASE].index.get_level_values("t").unique())
    elif isinstance(years, int):
        years = [years]

    out = build_nx_by_pt_agg(NX, share_dict, ea, partners,
                             partner_map=partner_map, partner_label=partner_label)
    comp = list(share_dict[STATE_BASE].columns)
    comps = _nx_pt_present(out, comp)
    if NUCLEAR_PT in comps:                         # kernekraft -> bunden af stakken
        comps = [NUCLEAR_PT, *(c for c in comps if c != NUCLEAR_PT)]
    colors = [_color(c, ctx.area_colors) for c in comps]
    hatched = set(ctx.EA_cols)

    for state in out:                                   # skalér flows til GWh
        out[state] = tuple(
            df.assign(**{c: df[c].mul(1000) for c in comps}) for df in out[state]
        )

    annual = _nx_pt_annual_twh(out, comps)              # behold alle bånd; tynd legenden ud
    legend_comps = _legend_comps(
        annual, comps,
        min_twh=legend_min_twh, min_share=legend_min_share, max_n=legend_max_n,
    )

    def _stack(df):
        return df[comps].sum(axis=1)
    ymax = max(_stack(out[s][0]).max() for s in out) * 1.05
    ymin = min(_stack(out[s][1]).min() for s in out) * 1.05

    results = []
    for year in years:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        fig.patch.set_facecolor(BG_COLOR)
        handles = None
        for col, (ax, state) in enumerate(zip(axes, STATES)):
            exp, imp = (df[df["t"] == year].sort_values([sort_by, "hFull"])
                        .reset_index(drop=True) for df in out[state])
            h = _draw_nx_pt_panel(
                ax, exp, imp, comps, colors, hatched,
                y_lims=(ymin, ymax), show_ylabel=(col == 0),
            )
            ax.set_title(_state_label(state, labels_map), loc="left")
            if handles is None:
                handles = h

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(PANEL_EDGE)
                spine.set_linewidth(PANEL_EDGE_LW)

        labels = [_label(c, ctx.col_names) for c in legend_comps]
        leg_handles = [handles[comps.index(c)] for c in legend_comps]
        if labels:
            fig.legend(leg_handles, labels, loc="lower center",
                       bbox_to_anchor=(0.5, legend_y),
                       ncol=min(len(labels), 3), frameon=False, fontsize=11)
        fig.suptitle(
            FIG_TEXT["title_net_export_pt"].format(
                area=ctx.col_names.get(ea, ea),
                partner=ctx.col_names.get(partner_label, partner_label),
                year=year,
            ),
            x=0.124, ha="left", fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.0])
        results.append((fig, tuple(axes)))
        if save_path_fmt:
            fig.savefig(save_path_fmt.format(area=ea, year=year), dpi=300,
                        facecolor=fig.get_facecolor())
        if show:
            plt.show()
    return results


def _nx_pt_present(out, comp, eps=1e-7):
    """Komponenter med ikke-forsvindende flow i begge tilstande, i COL_ORDER."""
    tot = pd.Series(0.0, index=comp)
    for state in out:
        for df in out[state]:
            tot = tot.add(df[comp].abs().sum(), fill_value=0)
    present = [c for c in comp if tot.get(c, 0) > eps]
    return [c for c in COL_ORDER if c in present] + [c for c in present if c not in COL_ORDER]


def _nx_pt_annual_twh(out, comps):
    """Årlig energi pr. komponent (TWh), summeret over eksport- og importsiden og
    gennemsnittet over tilstande. Frames er GWh-flow pr. hFull; /1000 -> TWh."""
    tot = {c: 0.0 for c in comps}
    for state in out:
        for df in out[state]:                       # (exp, imp)
            for c in comps:
                tot[c] += df[c].abs().sum() / 1000.0
    n = len(out) or 1
    return {c: v / n for c, v in tot.items()}


def _legend_comps(annual, comps, *, min_twh=0.0, min_share=0.0, max_n=None):
    """Komponenter der fortjener en legendepost. Bånd tegnes uanset -- kun
    legenden tyndes ud. Rækkefølgen fra `comps` (COL_ORDER) bevares."""
    total = sum(annual.values())
    keep = [
        c for c in comps
        if annual[c] >= min_twh and (total <= 0 or annual[c] / total >= min_share)
    ]
    if max_n is not None and len(keep) > max_n:
        top = sorted(keep, key=lambda c: annual[c], reverse=True)[:max_n]
        keep = [c for c in keep if c in set(top)]
    return keep


def _draw_nx_pt_panel(ax, exp, imp, comps, colors, hatched, *, y_lims, show_ylabel):
    """Tegn én tilstands nettoeksport pr. anlægstype i ``ax``; returnerer handles.
    Timerne er prissorteret opstrøms; der tegnes ingen prisakse."""
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)

    exp[comps].plot(ax=ax, kind="area", stacked=True, linewidth=0.0, ylim=y_lims, color=colors)
    handles, _ = ax.get_legend_handles_labels()
    imp[comps].plot(ax=ax, kind="area", stacked=True, linewidth=0.0, ylim=y_lims,
                    color=colors, legend=False)

    n = len(comps)
    for i, col in enumerate(comps):          # skravér de importerede (handels-)bånd
        if col in hatched:
            for k in (i, i + n):
                ax.collections[k].set_hatch("//")
                ax.collections[k].set_edgecolor("#666666")

    ax.axhline(0, color="0.2", lw=1, ls="--")
    ax.grid(lw=0.6, alpha=0.35)
    if show_ylabel:
        ax.set_ylabel(FIG_TEXT["unit_energy"], loc="bottom")
    ax.set_xlabel(FIG_TEXT["xlabel_hours_src_price"], loc="left")
    if len(exp):
        ax.set_xlim(0, len(exp) - 1)

    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return handles


def plot_net_export_by_pt(
    ctx: ElBalanceContext, ea: str, eaea: str, *, share_dict, partner_map=None,
    years: Sequence | int | None = None, sort_by: str = "pE_EA_baseline",
    legend_min_twh: float = 0.0,
    legend_min_share: float = 0.0,
    legend_max_n: int | None = None,
    figsize: tuple = (12, 5), legend_y: float = -0.3,
    save_path_fmt: str | None = None, show: bool = True, state_labels=None,
):
    """Nettoeksport EA->EAEA opdelt på forsyningskomponent, basis vs. stød (1x2)"""
    NX = ctx.NX_dict
    if NX is None:
        raise ValueError("plot_net_export_by_pt kræver en el-kontekst (NX_dict).")
    labels_map = state_labels or ctx.state_labels
    if years is None:
        years = sorted(NX[STATE_BASE].index.get_level_values("t").unique())
    elif isinstance(years, int):
        years = [years]

    out = build_nx_by_pt(NX, share_dict, ea, eaea, partner_map=partner_map)
    comp = list(share_dict[STATE_BASE].columns)
    comps = _nx_pt_present(out, comp)
    colors = [_color(c, ctx.area_colors) for c in comps]
    hatched = set(ctx.EA_cols)

    for state in out:                                   # skalér flows til GWh
        out[state] = tuple(
            df.assign(**{c: df[c].mul(1000) for c in comps}) for df in out[state]
        )

    annual = _nx_pt_annual_twh(out, comps)              # behold alle bånd; tynd legenden ud
    legend_comps = _legend_comps(
        annual, comps,
        min_twh=legend_min_twh, min_share=legend_min_share, max_n=legend_max_n,
    )

    def _stack(df):
        return df[comps].sum(axis=1)
    ymax = max(_stack(out[s][0]).max() for s in out) * 1.05
    ymin = min(_stack(out[s][1]).min() for s in out) * 1.05

    results = []
    for year in years:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        fig.patch.set_facecolor(BG_COLOR)
        handles = None
        for col, (ax, state) in enumerate(zip(axes, STATES)):
            exp, imp = (df[df["t"] == year].sort_values([sort_by, "hFull"])
                        .reset_index(drop=True) for df in out[state])
            h = _draw_nx_pt_panel(
                ax, exp, imp, comps, colors, hatched,
                y_lims=(ymin, ymax), show_ylabel=(col == 0),
            )
            ax.set_title(_state_label(state, labels_map), loc="left")
            if handles is None:
                handles = h

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(PANEL_EDGE)
                spine.set_linewidth(PANEL_EDGE_LW)

        labels = [_label(c, ctx.col_names) for c in legend_comps]
        leg_handles = [handles[comps.index(c)] for c in legend_comps]
        if labels:
            fig.legend(leg_handles, labels, loc="lower center",
                       bbox_to_anchor=(0.5, legend_y),
                       ncol=min(len(labels), 3), frameon=False, fontsize=11)
        fig.suptitle(
            FIG_TEXT["title_net_export_pt"].format(
                area=ctx.col_names.get(ea, ea),
                partner=ctx.col_names.get(eaea, eaea),
                year=year,
            ),
            x=0.124, ha="left", fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.0])
        results.append((fig, tuple(axes)))
        if save_path_fmt:
            fig.savefig(save_path_fmt.format(area=ea, eaea=eaea, year=year),
                        dpi=300, facecolor=fig.get_facecolor())
        if show:
            plt.show()
    return results