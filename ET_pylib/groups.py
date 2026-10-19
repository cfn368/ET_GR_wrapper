"""
Sets and conventions — the model vocabulary of the KF26 wrapper
================================================================

Areas, country aggregates, plant- and consumption-type renames, nuclear ids and
the column conventions of the balance frames. Data only: everything rendered
(labels, colours, figure text) lives in `style`.
"""

from __future__ import annotations


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

def as_list(x):
    """A single value (str, number) as [x], an iterable as a list, None as None, so
    every multi-value argument also takes one value."""
    if x is None:
        return None
    if isinstance(x, (str, bytes)) or not hasattr(x, "__iter__"):
        return [x]
    return list(x)


def areas_of(region):
    """Zones behind a region: an aggregate ("DK", "Scandi") gives its members, any
    other name (a zone such as "DK1") is taken as a region of one."""
    return tuple(c_aggs[region]) if region in c_aggs else (region,)


def expand_areas(areas):
    """Zones behind one or more areas, aggregates expanded; None stays None."""
    if areas is None:
        return None
    return tuple(ea for a in as_list(areas) for ea in areas_of(a))


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
NUC_START_YEAR    = 2035
NUCLEAR_IDS_BY_GEO = {
    "disaggEA": {i: NUC_START_YEAR for i in NUC_ID_BY_EA.values()},
    "aggEA":    {i: NUC_START_YEAR for i in NUC_ID_BY_EA.values()},
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

assert set(PT_AGGS) == set(PT_AGG_ORDER), "PT_AGGS / PT_AGG_ORDER out of sync"
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
