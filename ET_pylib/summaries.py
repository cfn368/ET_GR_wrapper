"""Produce tables for overview of shock implications on key metrics"""

import pandas as pd
from .groups import c_aggs, EA_TO_COUNTRY, DCIIA_TYPES, ANALYSIS_YEAR

_EA_TO_COUNTRY = EA_TO_COUNTRY
DEFAULT_YEAR = ANALYSIS_YEAR


def _resolve_region(ea, country):
    """(ea | country) -> (tuple of EAs, label). Fails loudly on an unknown country."""
    if country is not None:
        if country not in c_aggs:
            raise KeyError(f"unknown country {country!r}; known: {sorted(c_aggs)}")
        return tuple(c_aggs[country]), country
    if ea is None:
        raise ValueError("pass ea= or country=")
    return (ea,), ea


def _regional_price(df, *, price="pE", weight_cols=DCIIA_TYPES):
    """Collapse areas -> one demand-weighted price per (t, hFull). Returns
    columns t / hFull / <price>_region / w. Weight: the DC-IIA consumers (end-user
    demand; qDemandE_DCIIA in the model)."""
    weight_cols = [c for c in weight_cols if c in df.columns]
    if not weight_cols:
        raise KeyError("no demand-weight columns in the frame")
    w = df[weight_cols].sum(axis=1).abs()
    pcol = f"{price}_region"
    g = (df.assign(_w=w, _wp=w * df[price])
           .groupby(["t", "hFull"], as_index=False)
           .agg(_wp=("_wp", "sum"), _w=("_w", "sum")))
    g[pcol] = g["_wp"] / g["_w"]
    return g.rename(columns={"_w": "w"})[["t", "hFull", pcol, "w"]]


def price_summary(ctx, ea=None, *, country=None, years=DEFAULT_YEAR):
    """Demand-weighted price moments (mean/std) + min/max on the demand-weighted
    *regional* hourly price"""
    target_eas, label = _resolve_region(ea, country)
    dem = ctx.demand_dict
    rows = []
    for state in ("baseline", "shock"):
        df = dem[state][["t", "EA", "hFull", "pE", *[c for c in DCIIA_TYPES if c in dem[state].columns]]]
        if years is not None:
            yrs = [years] if isinstance(years, int) else years
            df = df[df["t"].isin(yrs)]
        df = df[df["EA"].isin(target_eas)]
        reg = _regional_price(df)
        p, w = reg["pE_region"], reg["w"]
        W = w.sum()
        mean = (w * p).sum() / W
        var = (w * (p - mean) ** 2).sum() / W
        skew = (w * (p - mean) ** 3).sum() / W / var ** 1.5
        exkurt = (w * (p - mean) ** 4).sum() / W / var ** 2 - 3
        rows.append({"run": state, "mean": mean,
                     "min": p.min(), "max": p.max(), "std": var ** 0.5,
                     "skew": skew, "exkurt": exkurt})
    return (pd.DataFrame(rows).assign(region=label)
              .set_index(["region", "run"]).round(3))


def trade_summary(ctx, ea=None, *, country=None, years=DEFAULT_YEAR, value=False, drop_empty=True):
    """Trade flows by partner, for a single EA, country, or Scandinavia.

    Export is valued at the origin's own price (EA)
    Import is valued at the partner's price (EAEA)"""
    NX = ctx.NX_dict
    if country is not None:
        target_eas, origin_label = _resolve_region(None, country)
        target_eas = set(target_eas)
        relabel = lambda x: origin_label if x in target_eas else _EA_TO_COUNTRY.get(x, x)

    rows = []
    for state in ("baseline", "shock"):
        df = NX[state]
        if years is not None:
            yrs = [years] if isinstance(years, int) else years
            df = df[df.index.get_level_values("t").isin(yrs)]
        if ea is not None:
            exp = df[df.index.get_level_values("EA") == ea]
            imp = df[df.index.get_level_values("EAEA") == ea]
            exp_q = exp["lineExport"].clip(lower=0)
            imp_q = imp["lineExport"].clip(lower=0)
            if value:
                exp_q = exp_q * exp["pE_EA"]
                imp_q = imp_q * imp["pE_EA"]
            g = pd.DataFrame({
                "export": exp_q.groupby(level="EAEA").sum(),
                "import": imp_q.groupby(level="EA").sum(),
            }).fillna(0.0)
            g.index.name = "partner"
        else:
            s = df[["lineExport", "pE_EA", "pE_EAEA"]].reset_index()
            s["origin"] = s["EA"].map(relabel)
            s["partner"] = s["EAEA"].map(relabel)
            s = s[(s["origin"] == origin_label) & (s["partner"] != origin_label)]
            flow = s["lineExport"]
            exp_q = flow.clip(lower=0)
            imp_q = (-flow).clip(lower=0)
            if value:
                exp_q = exp_q * s["pE_EA"]
                imp_q = imp_q * s["pE_EAEA"]
            g = pd.DataFrame({
                "export": exp_q.groupby(s["partner"]).sum(),
                "import": imp_q.groupby(s["partner"]).sum(),
            }).fillna(0.0)
        g["net"] = g["export"] - g["import"]
        g.insert(0, "run", state)
        rows.append(g.reset_index())
    out = (pd.concat(rows, ignore_index=True)
             .set_index(["partner", "run"]).sort_index().round(2))
    if drop_empty:
        active = (out[["export", "import"]].abs().groupby(level="partner").max() > 0).any(axis=1)
        out = out[out.index.get_level_values("partner").isin(active[active].index)]
    return out


def full_trade_summary(ctx, ea=None, *, country=None, years=DEFAULT_YEAR, drop_empty=True):
    """Quantity (TWh) and value (mDKK) side by side per partner"""
    q = trade_summary(ctx, ea=ea, country=country, years=years,
                      value=False, drop_empty=drop_empty)
    v = trade_summary(ctx, ea=ea, country=country, years=years,
                      value=True, drop_empty=drop_empty)
    out = q.join(v, lsuffix="_TWh", rsuffix="_mDKK")
    totals = out.groupby(level="run").sum(numeric_only=True)
    totals.index = pd.MultiIndex.from_product(
        [["TOTAL"], totals.index], names=out.index.names
    )
    return pd.concat([out, totals]).round(2)


def scarcity_summary(ctx, ea=None, *, country=None, years=DEFAULT_YEAR,
                     null_threshold=20.0, mangel_factor=2.0):
    """Hour-fraction of near-zero (pE < null_threshold) and scarcity
    (pE > mangel_factor × annual mean) prices, on the demand-weighted regional pE."""
    dem = ctx.demand_dict
    target_eas, label = _resolve_region(ea, country)
    rows = []
    for state in ("baseline", "shock"):
        df = dem[state][["t", "EA", "hFull", "pE", *[c for c in DCIIA_TYPES if c in dem[state].columns]]]
        if years is not None:
            yrs = [years] if isinstance(years, int) else years
            df = df[df["t"].isin(yrs)]
        df = df[df["EA"].isin(target_eas)]
        per_hour = _regional_price(df)   # t, hFull, pE_region, w

        annual = (per_hour.groupby("t")
                  .apply(lambda g: (g["w"] * g["pE_region"]).sum() / g["w"].sum())
                  .rename("annual_mean"))
        per_hour = per_hour.assign(annual_mean=per_hour["t"].map(annual))

        is_null = per_hour["pE_region"] < null_threshold
        is_mangel = per_hour["pE_region"] > mangel_factor * per_hour["annual_mean"]
        N = len(per_hour)   # real hours in the window (8760 per year)
        rows.append({"run": state,
                     "nulpristime": 100 * is_null.sum() / N,
                     "mangeltime": 100 * is_mangel.sum() / N})
    return (pd.DataFrame(rows).assign(region=label)
              .set_index(["region", "run"]).sort_index().round(4))


def full_summary(ctx, ea=None, *, country=None, years=DEFAULT_YEAR):
    """Price moments + scarcity shares, side by side. Same signature as
    ``price_summary`` / ``scarcity_summary``."""
    p = price_summary(ctx, ea=ea, country=country, years=years)
    s = scarcity_summary(ctx, ea=ea, country=country, years=years)
    return p.join(s)


# --- terms of trade =============================================
_TOT_STATE, _TOT_PARTNER = "state", "partner"   # full_trade_summary index levels

_TOT_COLS = ["X_TWh_base", "X_TWh_shock", "M_TWh_base", "M_TWh_shock",
             "pX_base", "pX_shock", "pM_base", "pM_shock",
             "ToT_base", "ToT_shock", "dToT", "dToT_pct"]


def _tot_record(df):
    """Unit values and ToT from one partner's frame, indexed by state only."""
    if isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"expected a flat state index, got levels {df.index.names}")
    rec = {}
    for tag, state in (("base", "baseline"), ("shock", "shock")):
        r = df.loc[state]
        if isinstance(r, pd.DataFrame):             # >1 row: multi-year, ambiguous
            raise ValueError(f"{len(r)} rows for state {state!r}; pass a single year")
        rec[f"X_TWh_{tag}"] = r["export_TWh"]
        rec[f"M_TWh_{tag}"] = r["import_TWh"]
        rec[f"pX_{tag}"] = r["export_mDKK"] / r["export_TWh"] if r["export_TWh"] else float("nan")
        rec[f"pM_{tag}"] = r["import_mDKK"] / r["import_TWh"] if r["import_TWh"] else float("nan")
        rec[f"ToT_{tag}"] = rec[f"pX_{tag}"] / rec[f"pM_{tag}"] if rec[f"pM_{tag}"] else float("nan")
    rec["dToT"] = rec["ToT_shock"] - rec["ToT_base"]
    rec["dToT_pct"] = 100 * rec["dToT"] / rec["ToT_base"] if rec["ToT_base"] else float("nan")
    return rec


def terms_of_trade(ctx, *, country=None, countries=None, years=DEFAULT_YEAR):
    """National terms of trade = flow-weighted export price / import price.
    Export priced at the origin's own price (pE_EA), import at the partner's
    (pE_EAEA) -- the convention inherited from `trade_summary`."""
    if countries is None:
        countries = [country] if country is not None else ["DK", "NO", "SE", "Scandi"]
    rows = {
        c: _tot_record(full_trade_summary(ctx, country=c, years=years, drop_empty=True)
                       .xs("TOTAL", level=_TOT_PARTNER))
        for c in countries
    }
    return pd.DataFrame.from_dict(rows, orient="index")[_TOT_COLS]


def terms_of_trade_partner(ctx, *, country="DK", years=DEFAULT_YEAR, include_total=True):
    """Bilateral terms of trade for `country` against each trading partner.
    pX is `country`'s own price in hours it exports to that partner; pM is the
    partner's price in hours it imports. TOTAL must reconcile with `terms_of_trade`."""
    ts = full_trade_summary(ctx, country=country, years=years, drop_empty=True)
    partners = [p for p in ts.index.get_level_values(_TOT_PARTNER).unique() if p != "TOTAL"]
    if include_total:
        partners += ["TOTAL"]
    rows = {p: _tot_record(ts.xs(p, level=_TOT_PARTNER)) for p in partners}
    out = pd.DataFrame.from_dict(rows, orient="index")[_TOT_COLS]
    out.index.name = "partner"
    return out


def _print_tot(title, tot, cols, heads, fmts, *, w, label_w, rule_before="TOTAL"):
    """Shared ToT table printer: header, rows, optional pre-TOTAL rule."""
    print(title)
    hdr = f'{"":<{label_w}}' + "".join(f"{h:>{w}}" for h in heads)
    rule = "  " + "─" * (len(hdr) - 2)
    print(hdr)
    print(rule)
    for key, row in tot.iterrows():
        if key == rule_before:
            print(rule)
        print(f"  {key:<{label_w - 2}}" + "".join(
            f"{format(row[c], f):>{w}}" for c, f in zip(cols, fmts)))
    print(rule)


def print_terms_of_trade(ctx, *, countries=("DK", "NO", "SE", "Scandi"), years=DEFAULT_YEAR):
    """National terms-of-trade table (export price / import price)."""
    yr = years if isinstance(years, int) else "+".join(map(str, years))
    _print_tot(
        f"Terms of trade — {yr}  (pX,pM in DKK/MWh, ToT = pX/pM)",
        terms_of_trade(ctx, countries=list(countries), years=years),
        ["pX_base", "pX_shock", "pM_base", "pM_shock",
         "ToT_base", "ToT_shock", "dToT", "dToT_pct"],
        ["pX(b)", "pX(s)", "pM(b)", "pM(s)", "ToT(b)", "ToT(s)", "ΔToT", "%Δ"],
        [",.1f", ",.1f", ",.1f", ",.1f", ".3f", ".3f", "+.3f", "+.2f"],
        w=9, label_w=8, rule_before=None,
    )


def print_terms_of_trade_partner(ctx, *, country="DK", years=DEFAULT_YEAR,
                                 include_total=True):
    """Bilateral terms-of-trade table for one country, one row per partner."""
    yr = years if isinstance(years, int) else "+".join(map(str, years))
    _print_tot(
        f"Terms of trade, {country} per partner — {yr}  "
        f"(X/M in TWh, pX/pM in DKK/MWh, ToT = pX/pM)",
        terms_of_trade_partner(ctx, country=country, years=years,
                               include_total=include_total),
        _TOT_COLS,
        ["X(b)", "X(s)", "M(b)", "M(s)", "pX(b)", "pX(s)", "pM(b)", "pM(s)",
         "ToT(b)", "ToT(s)", "ΔToT", "%Δ"],
        [",.2f", ",.2f", ",.2f", ",.2f", ",.0f", ",.0f", ",.0f", ",.0f",
         ".3f", ".3f", "+.3f", "+.2f"],
        w=8, label_w=8,
    )