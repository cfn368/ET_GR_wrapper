"""
Rendering layer — labels, colours, figure text and matplotlib style
====================================================================

Code and dict keys stay English; everything rendered is Danish. Plot and print
modules take every string and colour from here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import pandas as pd

from .groups import NUCLEAR_PT, PARTNER_ORDER, c_aggs, HA_AGGS


# ==================== ==================== ==================== ====================
# 1. labels

STATE_BASE_LABEL = "Basisforløb"
STATE_LABELS     = {"baseline": STATE_BASE_LABEL, "shock": "Stød"}
STATE_SHORT      = {"baseline": "basis", "shock": "stød"}      # column heads of printed tables

SCENARIO_LABELS = {
    "stød":   "Stødforløb",
    "nuc":    "Kernekraftforløb",
    "wind":   "Vindforløb",
    "demand": "Efterspørgselsforløb",
}

col_names_en = {
    # plant types
    "ES": "Electricity storage",
    "HS": "Heat storage",
    "HYS": "Hydro (pump storage)",
    "HY": "Hydro (reservoir storage)",
    "RO": "Run-of-river",
    "PV": "Photovoltaic",
    "PVES": "PV with battery (clusters)",
    "SH": "Solar heat",
    "WL": "Wind (onshore)",
    "WS": "Wind (offshore)",
    "CD": "Condensation",
    "EX": "Extraction (CHP)",
    "BH": "Boiler heater",
    "EPE": "Exogenous electricity production",
    "EPH": "Exogenous district heat production",
    "FD": "FD",                                   # TODO: confirm what FD is
    "BP": "Backpressure (CHP)",
    "BB": "Backpressure with bypass (CHP)",
    "EH": "Electric heaters",
    "EF": "Electrofuels (PtX)",
    "PX": "Power-to-X (PtX)",
    "EV_V2G": "Electric vehicles (V2G)",
    "Nuclear": "Nuclear",
    # consumption types
    "classic": "Classic demand",
    "hsdc": "Data centres",
    "EV": "Electric transport",
    "EV_V2G_charge": "Electric vehicles (V2G charging)",
    "HP": "Individual heat pumps",
    "district_heat": "District heat demand",
    "CCS": "CCS electricity use",
    "distrLoss": "Distribution losses",
    "trLoss": "Transmission losses",
    # price zones
    "DK1": "Denmark (West)",
    "DK2": "Denmark (East)",
    "DK3": "Denmark (Bornholm)",
    # district-heat areas
    "DK1_C": "West Denmark, central", "DK1_D": "West Denmark, decentral",
    "DK2_C": "East Denmark, central", "DK2_D": "East Denmark, decentral",
    "NorthSea": "North Sea",
    "NOS": "Norway (South)",
    "NOM": "Norway (Middle)",
    "NON": "Norway (North)",
    "SE1": "Sweden (North)",
    "SE2": "Sweden (Middle north)",
    "SE3": "Sweden (Middle south)",
    "SE4": "Sweden (South)",
    "FI": "Finland",
    "EELVLT": "Baltics",
    "GBNIIE": "United Kingdom & Ireland",
    "NL": "Netherlands",
    "DELU": "Germany & Luxembourg",
    "BE": "Belgium",
    "FR": "France",
    "CH": "Switzerland",
    "AT": "Austria",
    "PL": "Poland",
    "CZSK": "Czechia & Slovakia",
    "HUSI": "Hungary & Slovenia",
    "IT": "Italy",
    "ESPT": "Spain & Portugal",
    # country aggregates
    "DK": "Denmark",
    "NO": "Norway",
    "SE": "Sweden",
    "Scandi": "Scandinavia",
    "ALL": "abroad",
    "ROW": "Rest of Europe",
}

col_names_da = {
    # plant types
    "ES": "Ellagring",
    "HS": "Varmelagring",
    "HYS": "Vandkraft (pumpelager)",
    "HY": "Vandkraft (magasin)",
    "RO": "Vandkraft (flodkraft)",
    "PV": "Solceller",
    "PVES": "Solceller med batteri (klynger)",
    "SH": "Solvarme",
    "WL": "Landvind",
    "WS": "Havvind",
    "CD": "Kondensværk",
    "EX": "Udtagsværk (kraftvarme)",
    "BH": "Varmekedel",
    "EPE": "Eksogen elproduktion",
    "EPH": "Eksogen fjernvarmeproduktion",
    "FD": "FD",                                   # TODO: bekræft hvad FD er
    "BP": "Modtryksværk (kraftvarme)",
    "BB": "Modtryksværk med bypass (kraftvarme)",
    "EH": "Elkedler og varmepumper",
    "EF": "Elektrofuels (PtX)",
    "PX": "Power-to-X med varme",
    "EV_V2G": "Elbiler (V2G)",
    "Nuclear": "Kernekraft",
    # consumption types
    "classic": "Klassisk elforbrug",
    "hsdc": "Datacentre",
    "EV": "Eltransport",
    "EV_V2G_charge": "Elbiler (V2G-opladning)",
    "HP": "Individuelle varmepumper",
    "district_heat": "Fjernvarmeforbrug",
    "CCS": "CCS-elforbrug",
    "distrLoss": "Distributionstab",
    "trLoss": "Transmissionstab",
    # price zones
    "DK1": "Danmark (Vest)",
    "DK2": "Danmark (Øst)",
    "DK3": "Danmark (Bornholm)",
    # district-heat areas
    "DK1_C": "Vestdanmark, central", "DK1_D": "Vestdanmark, decentral",
    "DK2_C": "Østdanmark, central", "DK2_D": "Østdanmark, decentral",
    "NorthSea": "Nordsøen",
    "NOS": "Norge (syd)",
    "NOM": "Norge (midt)",
    "NON": "Norge (nord)",
    "SE1": "Sverige (nord)",
    "SE2": "Sverige (midtnord)",
    "SE3": "Sverige (midtsyd)",
    "SE4": "Sverige (syd)",
    "FI": "Finland",
    "EELVLT": "Baltikum",
    "GBNIIE": "Storbritannien og Irland",
    "NL": "Nederlandene",
    "DELU": "Tyskland og Luxembourg",
    "BE": "Belgien",
    "FR": "Frankrig",
    "CH": "Schweiz",
    "AT": "Østrig",
    "PL": "Polen",
    "CZSK": "Tjekkiet og Slovakiet",
    "HUSI": "Ungarn og Slovenien",
    "IT": "Italien",
    "ESPT": "Spanien og Portugal",
    # country aggregates
    "DK": "Danmark",
    "NO": "Norge",
    "SE": "Sverige",
    "Scandi": "Skandinavien",
    "ALL": "udlandet",
    "ROW": "Øvrige Europa",
}

# Active label map. Swap to `col_names_en` for an English deliverable.
col_names = col_names_da

PRICE_STAT_LABELS = {
    "mean": "Gennemsnitspris",
    "std":  "Prisvolatilitet (std.afv.)",
    "min":  "Minimumspris",
    "max":  "Maksimumspris",
}

TRADE_COMP_LABELS = {
    "net":    "Nettoeksport",
    "export": "Eksport",
    "import": "Import",
}

WELFARE_COMP_LABELS = {
    "dCS":     "Forbrugeroverskud",
    "dPS":     "Producentoverskud",
    "dCI":     "Flaskehalsindtægt",
    "annuity": "Annuitet (investering)",
    "net":     "Nettoeffekt",
}

FIG_TEXT = {
    # units
    "unit_price":     "kr./MWh",
    "unit_price_ore": "øre/kWh",
    "unit_pct":       "pct.",
    "unit_energy":    "GWh",
    "unit_energy_yr": "TWh",
    "unit_value":     "mia. kr.",
    "unit_value_mio": "mio. kr.",

    # shared fragments
    "all_years":      "alle år",
    "delta_abs":      "ændring",
    "delta_rel":      "relativ ændring",
    "delta_note":     "kernekraftforløb − basisforløb",
    "total_label":    "I alt",
    "total_label_nx": "Nettoeksport",

    # axis labels
    "xlabel_hours_price":     "Timer, sorteret efter områdepris",
    "xlabel_hours_src_price": "Timer, sorteret efter pris i afsenderområde",
    "ylabel_price_el":        "Elpris (kr./MWh)",
    "ylabel_price_dh":        "Fjernvarmepris (kr./MWh)",
    "ylabel_price_el_ore":    "Elpris (øre/kWh)",
    "ylabel_price_dh_ore":    "Fjernvarmepris (øre/kWh)",
    "ylabel_dp":              "Stød − basisforløb (øre/kWh)",
    "xlabel_base_price":      "Pris, basisforløb (øre/kWh)",
    "xlabel_price_quintile":  "Kvintil pris, basisforløb",

    # titles: prices
    "title_price_levels":  "Efterspørgselsvægtet elpris pr. region i {year}",
    "title_price_delta":   "Priseffekt af stødet pr. region i {year} ({unit}, {note})",
    "title_capture_delta": "Ændring i fangstpris pr. produktionstype i {region}, {year}",
    "title_price_scatter": "Figur 3: Kernekraft sænker priserne i de dyreste timer",
    "panel_price_change":  "Ændring i pris per time",
    "panel_cs_change":     "Ændring i forbrugeroverskud",
    "legend_state_point":  "Time (skaleret efter klyngestørrelse)",
    "legend_binned_mean":  "Time-vægtet gns. per kvintil",

    # titles: trade
    "title_trade_q":       "Nettoeksport af el pr. region i {year}",
    "title_trade_v":       "Værdi af nettoeksport af el pr. region i {year}",
    "title_trade_delta_q": "Ændring i nettoeksport pr. handelspartner i {year}",
    "title_trade_delta_v": "Ændring i eksportværdi pr. handelspartner i {year}",
    "title_nx_pt_delta":   "Ændring i nettoeksport pr. produktionstype i {year}",
    "title_net_export":    "Nettoeksport af el for {area} i {year}",
    "title_net_export_pt": "Figur 2: Nettoeksport fra {area} "
                           "fordelt på anlægstype i {year}",

    # titles: production, profit, welfare
    "title_gen_stack":    "Elproduktion pr. region i {year}",
    "title_gen_delta":    "Ændring i elproduktion pr. produktionstype i {year}",
    "title_profit_stack": "Producentoverskud pr. anlægstype pr. region i {year}",
    "title_profit_delta": "Ændring i producentoverskud pr. produktionstype i {year}",
    "title_welfare":      "Ændring i velfærd pr. region i {year} ({note})",
    "welfare_ci_note":    "inkl. flaskehalsindtægt",
    "welfare_net_note":   "inkl. flaskehalsindtægt og annuitet",
    "title_syscost_gap":  "Velfærdseffekt efter forskel i systemomkostninger, {area} {year}, "
                          "rente {rate}",
    "xlabel_syscost_gap": "Systemomkostning, VE − kernekraft (øre/kWh)",
    "ylabel_net_welfare": "Netto velfærdseffekt (mio. kr./år)",
    "mark_syscost_now":   "Hovedscenarie: {gap} øre/kWh, {net} mio. kr.",
    "mark_syscost_even":  "Ligevægt: {gap} øre/kWh",

    # legend group headers: energy balance and net export by component
    "legend_group_supply":  "Produktion og lagring",
    "legend_group_demand":  "Forbrug og tab",
    "legend_group_trade":   "Import og eksport",
    "legend_group_transit": "Transit",

    # titles: energy balance
    "title_balance_el": "Figur 1: Elbalance over året for {area} i {year}",
    "title_balance_dh": "Fjernvarmebalance over året for {area} i {year}",

    # consumer surplus by price bin
    "title_cs_price_bin": "Ændring i forbrugeroverskud fordelt på prisniveau, {year}",
    "price_bin_axis":     "Timer rangeret efter baselinepris (billigst til dyrest)",
    "price_bin_fmt":      "{lo}–{hi} %",
}

# Figure text that differs between the electricity ("E") and district-heat ("H") market.
MARKET_TEXT = {
    "E": {**{k: FIG_TEXT[k] for k in ("title_price_levels", "title_price_delta",
                                      "title_capture_delta", "title_price_scatter",
                                      "title_gen_stack", "title_gen_delta",
                                      "title_cs_price_bin")},
          "title_price_table": "Priser og knaphed — {years}",
          "title_cs_table":    "Forbrugeroverskud, el — {year}",
          "title_qty_table":   "Elproduktion — {areas}, {year}"},
    "H": {
        "title_price_levels":  "Efterspørgselsvægtet fjernvarmepris pr. region i {year}",
        "title_price_delta":   "Effekt af stødet på fjernvarmeprisen pr. region i {year} "
                               "({unit}, {note})",
        "title_capture_delta": "Ændring i fangstpris for fjernvarme pr. produktionstype "
                               "i {region}, {year}",
        "title_price_scatter": "Ændring i fjernvarmepris og forbrugeroverskud pr. time",
        "title_gen_stack":     "Fjernvarmeproduktion pr. region i {year}",
        "title_gen_delta":     "Ændring i fjernvarmeproduktion pr. produktionstype i {year}",
        "title_cs_price_bin":  "Ændring i forbrugeroverskud på fjernvarme fordelt på "
                               "prisniveau, {year}",
        "title_price_table":   "Fjernvarmepriser og knaphed — {years}",
        "title_cs_table":      "Forbrugeroverskud, fjernvarme — {year}",
        "title_qty_table":     "Fjernvarmeproduktion — {areas}, {year}",
    },
}
assert set(MARKET_TEXT["E"]) == set(MARKET_TEXT["H"]), "MARKET_TEXT keys out of sync"


def _dk_swap(txt):
    return txt.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def dk_num(value, decimals: int = 0, signed: bool = False) -> str:
    """Danish number format: '.' thousands, ',' decimal; sign taken from value < 0."""
    sign = "-" if value < 0 else ("+" if signed else "")
    return sign + _dk_swap(f"{abs(value):,.{decimals}f}")


def area_label(areas):
    """Danish name of an area scope: a known aggregate by name, else its zones."""
    areas = (areas,) if isinstance(areas, str) else tuple(areas)
    for name, members in (*c_aggs.items(), *HA_AGGS.items()):
        if set(members) == set(areas):
            return col_names.get(name, name)
    return col_names.get(areas[0], areas[0]) if len(areas) == 1 else "+".join(areas)


def price_bin_labels(n_bins):
    """Percentile labels for equal-hour bins, cheapest first."""
    edges = [round(100 * k / n_bins) for k in range(n_bins + 1)]
    return [FIG_TEXT["price_bin_fmt"].format(lo=lo, hi=hi)
            for lo, hi in zip(edges[:-1], edges[1:])]



# ==================== ==================== ==================== ====================
# 2. printed tables

TOTAL_LABEL = "I alt"


def print_table(df, fmts, *, title, unit=None, heads=None, groups=None, total=None,
                rules=(), notes=(), nan="—"):
    """The one printed-table layout: title (unit), optional group header, column
    header, rule, rows, rule, optional 'I alt' row, notes, blank line. The index
    holds the Danish row labels; numbers in Danish format.

    `fmts` is one format spec, a list per column or a dict by column; `heads` the
    column headers (default: column names); `groups` a group name per column
    (consecutive equal names share a centred header); `total` None, "all" or the
    columns to total; `rules` row positions to put a rule before; `notes` lines
    printed under the table. None cells print blank."""
    cols = list(df.columns)
    if isinstance(fmts, str):
        fmts = [fmts] * len(cols)
    elif isinstance(fmts, dict):
        fmts = [fmts[c] for c in cols]
    heads = [str(c) for c in cols] if heads is None else list(heads)

    def cell(v, f):
        if v is None:
            return ""
        if pd.isna(v):
            return nan
        return _dk_swap(format(v, f))

    body = [[cell(v, f) for v, f in zip(row, fmts)] for row in df.itertuples(index=False)]
    labels = [str(k) for k in df.index]
    tot = None
    if total is not None:
        tcols = cols if total == "all" else list(total)
        sums = df[tcols].sum()
        tot = [cell(sums[c], f) if c in tcols else "" for c, f in zip(cols, fmts)]

    rows_all = body + ([tot] if tot else [])
    widths = [max([len(h)] + [len(r[j]) for r in rows_all]) + 2 for j, h in enumerate(heads)]
    if groups is not None:
        j = 0
        while j < len(cols):                         # widen columns so group names fit
            k = j
            while k + 1 < len(cols) and groups[k + 1] == groups[j]:
                k += 1
            need = len(groups[j]) + 2 - sum(widths[j:k + 1])
            if need > 0:                             # spread the extra width over the group
                for i in range(j, k + 1):
                    widths[i] += need // (k - j + 1)
                widths[k] += need % (k - j + 1)
            j = k + 1
    label_w = max([len(l) for l in labels] + [len(TOTAL_LABEL) if tot else 0, 4]) + 2
    rule = "─" * (label_w + sum(widths))

    out = [f"{title} ({unit})" if unit else title]
    if groups is not None:
        line, j = " " * label_w, 0
        while j < len(cols):
            k = j
            while k + 1 < len(cols) and groups[k + 1] == groups[j]:
                k += 1
            span = sum(widths[j:k + 1])
            line += (groups[j] or "").center(span)
            j = k + 1
        out.append(line.rstrip())
    out.append(" " * label_w + "".join(h.rjust(w) for h, w in zip(heads, widths)))
    out.append(rule)
    for i, (lab, r) in enumerate(zip(labels, body)):
        if i in rules:
            out.append(rule)
        out.append(lab.ljust(label_w) + "".join(c.rjust(w) for c, w in zip(r, widths)))
    out.append(rule)
    if tot:
        out.append(TOTAL_LABEL.ljust(label_w) + "".join(c.rjust(w) for c, w in zip(tot, widths)))
    out += list(notes)
    print("\n".join(out) + "\n")


# ==================== ==================== ==================== ====================
# 3. colours

area_colors = {
    # plant types
    "ES":  "#0B3C5D", "HS":  "#0B3C5D", "HYS": "#6BAED6", "HY":  "#9ECAE1",
    "RO":  "#C6DBEF", "PV":  "#F5A623", "PVES": "#E0901A", "SH":  "#F5A623",
    "WL":  "#2E8B57", "WS":  "#66C2A5", "CD":  "#6F4E37", "EX":  "#6F4E37",
    "EPE": "#8B5A2B", "BH":  "#6F4E37", "EPH": "#8B5A2B", "FD":  "#A0785A",
    "BP":  "#D2B48C", "BB":  "#C49A6C", "EH":  "#E78AC3", "PX":  "#8E63CE",
    "EF":  "#B39DDB", "EV_V2G": "#1F6F8B", "Nuclear": "#000000",
    # consumption types
    "classic": "#B22222", "hsdc": "#7A1A1A", "EV": "#D95F5F",
    "EV_V2G_charge": "#1F6F8B", "HP": "#E8908A", "district_heat": "#B22222", "CCS": "#5A5A5A",
    "distrLoss": "#F4A3A3", "trLoss": "#F7C6C6",
    # price zones
    "DK1": "#A05A5A", "DK2": "#C27B7B", "DK3": "#D9A3A3", "NorthSea": "#3E6E8E",
    # district-heat areas (zone hue, decentral lighter)
    "DK1_C": "#A05A5A", "DK1_D": "#C98F8F", "DK2_C": "#C27B7B", "DK2_D": "#DDB0B0",
    "NOS": "#5B7C99", "NON": "#7F9DB8", "NOM": "#A6BED3",
    "SE1": "#C9A227", "SE2": "#D8B43C", "SE3": "#E5C85A", "SE4": "#F0DC7A",
    "FI": "#8FB996", "EELVLT": "#6E9C76",
    "DELU": "#8C8C8C", "NL": "#C28A52", "BE": "#B07A4A", "FR": "#4F6FA8",
    "GBNIIE": "#75618C", "CH": "#9E4A4A", "AT": "#B8B8B8",
    "PL": "#A3A3C2", "CZSK": "#8585A8", "HUSI": "#6B6B8E",
    "IT": "#5E9E8C", "ESPT": "#D9A066",
    # country aggregates (each takes its first zone's hue)
    "DK": "#A05A5A", "NO": "#5B7C99", "SE": "#C9A227", "ROW": "#A9A9A9",
}

PT_AGG_COLORS = {
    "Vandkraft":  "#6BAED6",
    "Vind":       "#2E8B57",
    "Sol":        "#F5A623",
    "Batterier":  "#0B3C5D",
    "Regulerbar": "#6F4E37",
    "Varme":      "#D2B48C",
    "Kernekraft": "#000000",
    # district-heat buckets
    "Kraftvarme":     "#D2B48C",
    "Elvarme":        "#E78AC3",
    "Kedler":         "#8B5A2B",
    "Solvarme":       "#F5A623",
    "Varmelager":     "#0B3C5D",
    "Eksogen varme":  "#A0785A",
    "Overskudsvarme": "#8E63CE",
}

# navy = baseline, red = shock (matches the Excel workbooks)
STATE_COLORS = {"baseline": "#0B3C5D", "shock": "#B22222"}
DELTA_COLORS = {"up": "#B22222", "down": "#0B3C5D"}

WELFARE_COLORS = {
    "dCS":     "#B22222",     # red
    "dPS":     "#0B3C5D",     # navy
    "dCI":     "#5B7C99",     # slate -- congestion rent
    "annuity": "#6F4E37",     # brown -- the only negative-by-construction bar
    "net":     "#C9A227",     # gold -- the summary bar
}

DEFAULT_LABELS = {NUCLEAR_PT: col_names.get(NUCLEAR_PT, "Kernekraft")}
DEFAULT_COLORS = {NUCLEAR_PT: "#000000"}
MISSING_COLOR  = "#CCCCCC"
NX_COMP_COLORS = {**PT_AGG_COLORS, **{k: area_colors[k] for k in PARTNER_ORDER
                                      if k in area_colors}}

PRICE_LINE_COLOR = "#0D0AF5"   # price on the twin axis of the balance figures
HATCH_EDGE       = "#666666"   # edge of hatched trade bands
ZERO_LINE        = "0.2"
SCATTER_COLOR    = "#C7372A"
SCATTER_NEUTRAL  = "#9AA5AA"   # neutral, so a linked line stands out


def color_of(key, colors=area_colors):
    """Colour for a series key, with the nuclear default and a grey fallback."""
    return colors[key] if key in colors else DEFAULT_COLORS.get(key, MISSING_COLOR)


def label_of(key, names=col_names):
    """Danish label for a series key, with the nuclear default and the raw key as fallback."""
    return names[key] if key in names else DEFAULT_LABELS.get(key, key)


# ==================== ==================== ==================== ====================
# 4. figure style

BG_COLOR      = "#E8EDEF"      # figure/axes fill, replacing white + boxed frame
TEXT_COLOR    = "#2F3C41"
PANEL_EDGE    = "#C9D2D5"
PANEL_EDGE_LW = 0.8
FIGSIZE_WIDE  = (12, 4)        # house width is always 12

# 1x2 balance figures: gap between the panels (share of axes width), title and legend
# offsets (inches) and the column cap of the grouped legend.
PAIR_WSPACE     = 0.06
TITLE_GAP_IN    = 0.20         # suptitle bottom above the panel titles
LEGEND_GAP_IN   = 0.12         # legend box top below the lowest x-label (box adds padding)
LEGEND_MAX_COLS = 5

_AEJ_RCPARAMS = {
    "font.family":          "serif",
    "font.size":            15,
    "axes.linewidth":       1.0,
    "lines.linewidth":      1.2,
    "axes.spines.top":      True,
    "axes.spines.right":    True,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "legend.frameon":       True,
    "legend.fancybox":      True,
    "legend.borderaxespad": 0.4,
    "legend.handlelength":  2.0,
    "legend.handletextpad": 0.6,
    "legend.labelspacing":  0.35,
    "savefig.bbox":         "tight",
    "savefig.dpi":          300,
    "text.color":           TEXT_COLOR,
    "axes.labelcolor":      TEXT_COLOR,
    "axes.titlecolor":      TEXT_COLOR,
    "xtick.color":          TEXT_COLOR,
    "ytick.color":          TEXT_COLOR,
    "axes.labelsize":       11,
    "xtick.labelsize":      11,
    "ytick.labelsize":      11,
    "axes.titlesize":       11,
    "legend.fontsize":      11,
}


def set_aej() -> None:
    """Apply the house matplotlib defaults."""
    plt.rcParams.update(_AEJ_RCPARAMS)
