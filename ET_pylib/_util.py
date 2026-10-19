"""
Shared helpers — database access, year filtering, id maps and table printing
=============================================================================

Every module reads the solved GpyDB pickles through these helpers, so a symbol
is fetched, year-filtered and mapped to plant type / area in one way only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .groups import NUCLEAR_PT


# ==================== ==================== ==================== ====================
# 1. years

def years_list(years):
    """`years` (int, str, iterable) as a sorted list of ints, or None for 'all'."""
    if years is None:
        return None
    if isinstance(years, (int, np.integer, str)):
        years = [years]
    return sorted({int(y) for y in years})


def filter_years(obj, years, *, what="frame"):
    """Keep only `years` on a `t` column or index level. No-op for years=None or
    an object without `t`; raises on an empty cut (a str year would match nothing)."""
    yrs = years_list(years)
    if yrs is None:
        return obj
    if "t" in list(getattr(obj, "columns", [])):
        have = obj["t"].astype("int64")
        keep = have.isin(yrs)
    elif isinstance(obj.index, pd.MultiIndex) and "t" in obj.index.names:
        i = obj.index.names.index("t")
        have = obj.index.levels[i].astype("int64")
        keep = np.asarray(have.isin(yrs))[obj.index.codes[i]]
    elif obj.index.name == "t":
        have = obj.index.astype("int64")
        keep = have.isin(yrs)
    else:
        return obj
    out = obj[keep]
    if len(out) == 0:
        raise ValueError(f"years={yrs} matched no rows in {what}; t holds {sorted(set(have))}")
    return out


# ==================== ==================== ==================== ====================
# 2. database access

def has_sym(db, sym):
    return sym in set(map(str, db.symbols.keys()))


def sym_frame(db, sym, years=None):
    """A parameter/variable as a flat frame with its value column named `sym`."""
    return filter_years(db(sym).rename(sym).reset_index(), years, what=sym)


def sym_year(db, sym, year):
    """A symbol for one year with `t` dropped; a single-level index is cast to str."""
    out = filter_years(db(sym), year, what=sym).droplevel("t")
    if out.index.nlevels == 1:
        out.index = out.index.astype(str)
    return out


def set_frame(db, sym, years=None):
    """A set (or mapping) as a frame of its index levels."""
    return filter_years(db(sym).to_frame()[[]].reset_index(), years, what=sym)


def mapping(db, sym, key):
    """A mapping set as a frame, required one-to-one on `key` so a merge on it
    can never duplicate rows."""
    df = set_frame(db, sym)
    dup = df[key].duplicated()
    if dup.any():
        raise ValueError(f"{sym} maps some {key} to more than one value: "
                         f"{sorted(df.loc[dup, key].astype(str).unique())[:10]}")
    return df


def id_map(m):
    """id (str) -> category Series from a 2-level mapping (symbol, vals or MultiIndex)."""
    mi = m.vals if hasattr(m, "vals") else m
    return pd.Series(mi.get_level_values(-1), index=mi.get_level_values(0).astype(str))


def level_isin(index, level, values):
    """Boolean mask of `index` rows whose `level` value is in `values`; tests the
    level's unique values and maps through the codes (fast on large frames)."""
    i = index.names.index(level)
    return np.asarray(index.levels[i].isin(list(values)))[index.codes[i]]


def require_mapped(df, cols, what):
    """Raise if a merge left some rows without their mapping."""
    for c in cols:
        if df[c].isna().any():
            raise ValueError(f"{what}: {int(df[c].isna().sum())} rows without {c} after the merge")


# ==================== ==================== ==================== ====================
# 3. model conventions shared across modules

def relabel_nuclear(df, nuclear_ids):
    """Pt = Nuclear on the nuclear ids from their first nuclear year on."""
    if not nuclear_ids:
        return df
    start = df["id"].astype(str).map(nuclear_ids)
    mask = start.notna() & (df["t"].astype("int64") >= start.fillna(0).astype("int64"))
    df.loc[mask, "Pt"] = NUCLEAR_PT
    return df


def cluster_members(db):
    """Plant ids behind a shared PV-battery connection (cleared as one net injection)."""
    if not has_sym(db, "id2Cluster"):
        return set()
    return set(set_frame(db, "id2Cluster")["id"].astype(str))


def active_line_export(db, years=None):
    """lineExport on the rows the market clears on: activeLineEA[t,line,EA,EAEA] with
    both areas active. lineExport also carries levels on inactive combinations, e.g.
    DK2<->DELU on the Bornholm line DELU__DK3, which no balance counts."""
    missing = {"activeLineEA", "activeEA"} - set(map(str, db.symbols.keys()))
    if missing:
        raise KeyError(f"{sorted(missing)} is missing: cannot tell active lines from inactive")
    line = sym_frame(db, "lineExport", years)
    act = set_frame(db, "activeLineEA", years)
    ea = set_frame(db, "activeEA", years)
    act = (act.merge(ea, on=["t", "EA"])
              .merge(ea.rename(columns={"EA": "EAEA"}), on=["t", "EAEA"]))
    return line.merge(act, on=["t", "line", "EA", "EAEA"], how="inner")


def equal_hour_bins(x, w, n_bins):
    """Bin index (0 = cheapest) per row: rows sorted by x, n_bins bins of equal
    hours w; a row goes to the bin holding its hour-midpoint."""
    o = x.sort_values(kind="stable").index
    ws = w[o]
    mid = ws.cumsum() - ws / 2
    b = np.minimum((mid / ws.sum() * n_bins).astype(int), n_bins - 1)
    return pd.Series(np.asarray(b), index=o).reindex(x.index)


# ==================== ==================== ==================== ====================
# 4. table printing

def print_table(df, cols, heads, fmts, *, title, w, label_w=6, total=None,
                rule_before=None, nan=None, blank=True):
    """Right-aligned table: title, header, rule, one row per index key, rule, and
    optionally a total row. `fmts` is one format spec or one per column; `total`
    is None, "all" or the columns to total; `nan` replaces missing cells."""
    fmts = [fmts] * len(cols) if isinstance(fmts, str) else list(fmts)
    cell = lambda v, f: (f"{nan:>{w}}" if nan is not None and pd.isna(v)
                         else f"{format(v, f):>{w}}")
    print(title)
    hdr = f'{"":<{label_w + 2}}' + "".join(f"{h:>{w}}" for h in heads)
    rule = "  " + "─" * (len(hdr) - 2)
    print(hdr)
    print(rule)
    for key, row in df.iterrows():
        if rule_before is not None and key == rule_before:
            print(rule)
        print(f"  {key:<{label_w}}" + "".join(cell(row[c], f) for c, f in zip(cols, fmts)))
    print(rule)
    if total is not None:
        tcols = list(cols) if total == "all" else list(total)
        tot = df[tcols].sum()
        print(f'  {"Total":<{label_w}}' + "".join(
            f"{format(tot[c], f):>{w}}" if c in tcols else f'{"":>{w}}'
            for c, f in zip(cols, fmts)))
    if blank:
        print()
