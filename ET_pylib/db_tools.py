"""Database helpers shared by the analysis modules."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


class _Symbol:
    """A database symbol with its values replaced; everything else from the original."""

    def __init__(self, original, vals):
        self._original = original
        self.vals = vals

    def __getattr__(self, name):
        return getattr(self._original, name)


class _Symbols(Mapping):
    def __init__(self, db, overrides):
        self._db, self._over = db, overrides

    def __getitem__(self, name):
        sym = self._db.symbols[name]
        return _Symbol(sym, self._over[name]) if name in self._over else sym

    def __iter__(self):
        return iter(self._db.symbols)

    def __len__(self):
        return len(self._db.symbols)

    def __contains__(self, name):
        return name in self._db.symbols


class ShockedDB:
    """A solved shock database with the shock's exogenous inputs laid on top.

    solve_shock writes back only solved (endogenous) levels, so the saved shock
    database carries the baseline's capacities and costs. `inputs` ({symbol:
    Series}, e.g. ET_shocks.nucShock.shock_inputs) overrides those records; every
    other record, and every other symbol, is the database's own. Reads through
    db(sym), db[sym].vals and db.symbols[sym].vals all see the overrides."""

    def __init__(self, db, inputs):
        self._db = db
        over = {}
        for sym, new in inputs.items():
            base = db(sym)
            names = list(base.index.names)
            if set(new.index.names) != set(names):
                raise ValueError(f"{sym}: inputs indexed by {list(new.index.names)}, "
                                 f"database by {names}")
            new = new.reorder_levels(names) if len(names) > 1 else new
            merged = new.combine_first(base).rename(base.name)
            merged = merged.reindex(base.index.union(new.index))
            over[sym] = merged.astype("float64") if merged.dtype == object else merged
        self._over = over
        self.symbols = _Symbols(db, over)

    def __call__(self, name):
        return self._over[name] if name in self._over else self._db(name)

    def __getitem__(self, name):
        return self.symbols[name]

    def __getattr__(self, name):
        return getattr(self._db, name)

    @property
    def overridden(self):
        """The symbols this wrapper changes."""
        return sorted(self._over)