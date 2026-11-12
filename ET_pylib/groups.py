"""
Sets and conventions — the model vocabulary of the KF26 wrapper
================================================================

Areas and heat areas, country aggregates, plant- and consumption-type renames,
nuclear ids, the column conventions of the balance frames and the two markets
("E" electricity, "H" district heat). Data only: everything rendered (labels,
colours, figure text) lives in `style`.
"""

from __future__ import annotations

from dataclasses import dataclass


# ==================== ==================== ==================== ====================
# 1. states and areas

STATE_BASE  = "baseline"
STATE_SHOCK = "shock"
STATES      = (STATE_BASE, STATE_SHOCK)

# Electricity areas of the disaggEA geography, in display order. No ROW in KF26.
EA_ORDER = [
    "DK1", "DK2", "DK3", "NorthSea",
    "NON", "NOM", "NOS",
    "SE1", "SE2", "SE3", "SE4",
    "FI", "EELVLT",
    "DELU", "NL", "BE", "FR", "GBNIIE", "CH", "AT",
    "PL", "CZSK", "HUSI",
    "IT", "ESPT",
    "ROW",
]
EA_cols = set(EA_ORDER)

# `c_aggs` includes Scandi; `c_aggs_countries` does not -- use the latter wherever
# a zone must map to exactly one country. DK3 is Energiø Bornholm and counts as DK.
c_aggs = {
    "DK":     ("DK1", "DK2", "DK3"),
    "NO":     ("NOS", "NON", "NOM"),
    "SE":     ("SE1", "SE2", "SE3", "SE4"),
    "Scandi": ("DK1", "DK2", "DK3", "NOS", "NON", "NOM", "SE1", "SE2", "SE3", "SE4"),
}
c_aggs_countries = {k: v for k, v in c_aggs.items() if k != "Scandi"}
EA_TO_COUNTRY    = {ea: c for c, eas in c_aggs_countries.items() for ea in eas}
SCAND_ZONES      = c_aggs["Scandi"]

# District-heat areas (HA market): central (_C) and decentral (_D) per Danish zone.
# A heat region is Denmark or one zone; DK3 has no heat area.
HA_ORDER = ["DK1_C", "DK1_D", "DK2_C", "DK2_D"]
HA_cols  = set(HA_ORDER)
HA_AGGS  = {
    "DK":  ("DK1_C", "DK1_D", "DK2_C", "DK2_D"),
    "DK1": ("DK1_C", "DK1_D"),
    "DK2": ("DK2_C", "DK2_D"),
}
_REGION_AGGS = {"E": c_aggs, "H": HA_AGGS}


def as_list(x):
    """A single value (str, number) as [x], an iterable as a list, None as None, so
    every multi-value argument also takes one value."""
    if x is None:
        return None
    if isinstance(x, (str, bytes)) or not hasattr(x, "__iter__"):
        return [x]
    return list(x)


def areas_of(region, market="E"):
    """Areas behind a region in `market` ("E" zones, "H" heat areas): an aggregate
    ("DK", "Scandi"; "DK1" on the heat side) gives its members, any other name is
    taken as a region of one."""
    if market not in _REGION_AGGS:
        raise KeyError(f"unknown market {market!r}; choose one of {sorted(_REGION_AGGS)}")
    aggs = _REGION_AGGS[market]
    return tuple(aggs[region]) if region in aggs else (region,)


def expand_areas(areas, market="E"):
    """Areas behind one or more areas, aggregates expanded; None stays None."""
    if areas is None:
        return None
    return tuple(a2 for a in as_list(areas) for a2 in areas_of(a, market))


# Trade partners: country aggregates plus every other area on its own.
_SINGLE_AREAS = [ea for ea in EA_ORDER if ea not in EA_TO_COUNTRY]
PARTNER_MAP   = {**c_aggs_countries, **{ea: (ea,) for ea in _SINGLE_AREAS}}

ANALYSIS_YEAR = 2040


# ==================== ==================== ==================== ====================
# 2. plant and consumption types

# `EV` is both a plant type (vehicle-to-grid storage) and a consumption type
# (transport demand abroad). The plant is renamed before supply and demand meet.
PT_RENAME    = {"EV": "EV_V2G"}
# Danish road transport is `transport`, abroad it is `EV`: one reporting category.
CTYPE_RENAME = {"transport": "EV", "EVStorage": "EV_V2G_charge"}

# Final consumers (the DC-IIA classes) after CTYPE_RENAME; their sum is qDemandE_DCIIA.
DCIIA_TYPES    = ("classic", "hsdc", "EV", "HP")
DH_DCIIA_TYPES = ("district_heat",)
AREAS_WITHOUT_DCIIA = ("DK3",)       # no final demand: left out of regional price weights

# Heat-only types: UP per heat capacity, no ECap.
HEAT_PTS = {"BH", "EPH", "SH", "HS"}


def _members(m):
    """Bucket members as tuples -- a bare string would iterate as characters."""
    return {k: (v,) if isinstance(v, str) else tuple(v) for k, v in m.items()}


# ==================== ==================== ==================== ====================
# 3. nuclear

# The nuclear shock adds its own CD plants through DREAM's added-plant machinery.
# NUC_ID_BY_EA is read by ET_shocks.nucShock; NUCLEAR_IDS maps id -> first nuclear year.
NUCLEAR_PT        = "Nuclear"
NUC_ID_BY_EA      = {"DK1": "idNUC_DK1", "DK2": "idNUC_DK2"}
# CHP variant (EX): one unit per Danish heat area, each zone's GWe split over its areas.
NUC_HA_BY_EA      = {"DK1": ("DK1_C", "DK1_D"), "DK2": ("DK2_C", "DK2_D")}
NUC_ID_BY_HA      = {ha: f"idNUC_{ha}" for has in NUC_HA_BY_EA.values() for ha in has}
NUC_START_YEAR    = 2035
# Hardcoded basis of the energy-equivalent wind cut (ET_shocks.nucShock.energy_cuts):
# offshore wind CF in the 2040 baseline (SEh / ECap x 8760) and nuclear el in the CHP run
# (6.21 TWh at 0.85 GWe, scaled to the intended 1 GWe; update after the rerun).
WS_CF_2040        = {"DK1": 0.520, "DK2": 0.467}
NUC_TWH_CHP_2040  = 7.30
_NUC_ALL_IDS     = (*NUC_ID_BY_EA.values(), *NUC_ID_BY_HA.values())   # CD and EX variants
NUCLEAR_IDS_BY_GEO = {
    "disaggEA": {i: NUC_START_YEAR for i in _NUC_ALL_IDS},
    "aggEA":    {i: NUC_START_YEAR for i in _NUC_ALL_IDS},
}
NUCLEAR_IDS = dict(NUCLEAR_IDS_BY_GEO["disaggEA"])


def use_geography(geo):
    """Set the active nuclear ids in place, so every module that imported
    NUCLEAR_IDS sees the change."""
    if geo not in NUCLEAR_IDS_BY_GEO:
        raise KeyError(f"unknown geography {geo!r}; choose one of {sorted(NUCLEAR_IDS_BY_GEO)}")
    NUCLEAR_IDS.clear()
    NUCLEAR_IDS.update(NUCLEAR_IDS_BY_GEO[geo])
    return dict(NUCLEAR_IDS)


# ==================== ==================== ==================== ====================
# 4. buckets and display orders

PT_AGGS = _members({
    "Sol":        ("PV", "PVES"),
    "Vind":       ("WS", "WL"),
    "Batterier":  ("ES", "EV", "EV_V2G"),
    "Vandkraft":  ("HY", "HYS", "RO"),
    "Regulerbar": ("CD", "EPE", "FD", "BP", "BB", "EX"),
    "Varme":      ("BH", "EPH", "SH", "HS", "EH"),
    "Kernekraft": (NUCLEAR_PT,),
})
PT_DROP = tuple(PT_AGGS["Varme"]) + ("PX", "EF")

REGION_ORDER        = ["DK", "NO", "SE", "Scandi"]
PARTNER_ORDER       = ["DK", "NO", "SE"] + _SINGLE_AREAS
PT_AGG_ORDER        = ["Kernekraft", "Vandkraft", "Vind", "Sol", "Batterier",
                       "Regulerbar", "Varme"]
WELFARE_SERIES      = ("dCS", "dPS", "net")
WELFARE_SERIES_FULL = ("dCS", "dPS", "dCI", "annuity", "net")

NX_COMP_AGGS  = {**PT_AGGS, **PARTNER_MAP}
NX_COMP_ORDER = PT_AGG_ORDER + PARTNER_ORDER

# District-heat production buckets (SHh); every heat-producing type is kept.
PT_AGGS_DH = _members({
    "Kernekraft":     (NUCLEAR_PT,),
    "Kraftvarme":     ("BP", "BB", "EX"),
    "Elvarme":        ("EH",),
    "Kedler":         ("BH",),
    "Solvarme":       ("SH",),
    "Varmelager":     ("HS",),
    "Eksogen varme":  ("EPH",),
    "Overskudsvarme": ("PX",),
})
PT_AGG_ORDER_DH  = ["Kernekraft", "Kraftvarme", "Elvarme", "Kedler", "Solvarme",
                    "Varmelager", "Eksogen varme", "Overskudsvarme"]
REGION_ORDER_DH  = ["DK1", "DK2", "DK"]

assert set(PT_AGGS) == set(PT_AGG_ORDER), "PT_AGGS / PT_AGG_ORDER out of sync"
assert set(PT_AGGS_DH) == set(PT_AGG_ORDER_DH), "PT_AGGS_DH / PT_AGG_ORDER_DH out of sync"
assert set(PARTNER_MAP) == set(PARTNER_ORDER), "PARTNER_MAP / PARTNER_ORDER out of sync"


# ==================== ==================== ==================== ====================
# 5. balance-frame columns

# electricity (EA market), price pE
_META_BUILD = ["t", "EA", "h", "hFull", "hWeight", "pE"]
META_COLS   = _META_BUILD + ["pE_baseline"]
PT_COLS = [
    "ES", "HYS", "EV_V2G", "HY", "RO", "PVES", "PV", "WL", "WS",
    "CD", NUCLEAR_PT, "EPE", "FD", "BP", "BB", "EX", "EH", "PX", "EF",
]
CTYPE_COLS  = ["classic", "hsdc", "EV", "EV_V2G_charge", "HP", "CCS", "distrLoss", "trLoss"]
COL_ORDER   = PT_COLS + CTYPE_COLS + EA_ORDER
_FULL_ORDER = _META_BUILD + COL_ORDER

# Legend groups of the electricity balance figures, as (group key, members). Electric
# heat and PtX are plant types in the model but draw power, so they sit with demand.
_PT_LOADS        = ("EH", "PX", "EF")
LEGEND_GROUPS_EL = (
    ("supply", tuple(c for c in PT_COLS if c not in _PT_LOADS)),
    ("demand", _PT_LOADS + tuple(CTYPE_COLS)),
    ("trade",  tuple(EA_ORDER)),
)
LEGEND_GROUPS_NX = (
    ("supply",  tuple(PT_COLS)),
    ("transit", tuple(EA_ORDER)),
)

# Storage-like series, clipped at 0 on the side where they do not belong.
CLAMP_SUPPLY = ("ES", "HYS", "EV_V2G")           # discharge only on the supply side
CLAMP_DEMAND = ("ES", "HYS", "EV_V2G_charge")     # charging only on the demand side
SPLIT_COLS   = ("PVES",)                          # net injection: + supply, - demand

# district heating (HA market), price pH, no inter-area trade
_META_BUILD_HA  = ["t", "HA", "h", "hFull", "hWeight", "pH"]
META_COLS_HA    = _META_BUILD_HA + ["pH_baseline"]
COL_ORDER_HA    = ["HS", "SH", "EH", "BH", "EPH", "BP", "BB", "EX", NUCLEAR_PT, "PX",
                   "district_heat", "CCS"]
_FULL_ORDER_HA  = _META_BUILD_HA + COL_ORDER_HA
CLAMP_SUPPLY_HA = ("HS", "SH")
CLAMP_DEMAND_HA = ("HS",)

# Legend groups of the district-heat balance figures; heat demand and CCS heat use
# are the only demand-side series.
_HA_LOADS        = ("district_heat", "CCS")
LEGEND_GROUPS_DH = (
    ("supply", tuple(c for c in COL_ORDER_HA if c not in _HA_LOADS)),
    ("demand", _HA_LOADS),
)


# ==================== ==================== ==================== ====================
# 6. markets: what separates electricity ("E") from district heat ("H")

@dataclass(frozen=True)
class Market:
    """Symbols, areas and buckets of one market. Rendered text lives in `style`."""
    key: str             # "E" / "H"
    area_key: str        # "EA" / "HA"
    price: str           # "pE" / "pH"
    supply_sym: str      # "SEh" / "SHh", pre-summed over cluster hours
    demand_sym: str      # final (DC-IIA) demand, (t, h, area)
    id2area: str         # "id2EA" / "id2HA"
    weight_cols: tuple   # DC-IIA columns of the demand frame: regional price weights
    no_demand: tuple     # areas without final demand
    area_cols: frozenset # single areas of the market
    region_order: tuple  # default regions of the figures
    pt_aggs: dict
    pt_drop: tuple
    pt_agg_order: tuple


MARKETS = {
    "E": Market("E", "EA", "pE", "SEh", "qDemandE_DCIIA", "id2EA", DCIIA_TYPES,
                AREAS_WITHOUT_DCIIA, frozenset(EA_cols), tuple(REGION_ORDER),
                PT_AGGS, PT_DROP, tuple(PT_AGG_ORDER)),
    "H": Market("H", "HA", "pH", "SHh", "qDemandH_DCIIA", "id2HA", DH_DCIIA_TYPES,
                (), frozenset(HA_cols), tuple(REGION_ORDER_DH),
                PT_AGGS_DH, (), tuple(PT_AGG_ORDER_DH)),
}


def market_of(market):
    """The `Market` behind "E" / "H"; raises on anything else."""
    if market not in MARKETS:
        raise KeyError(f"unknown market {market!r}; choose one of {sorted(MARKETS)}")
    return MARKETS[market]

