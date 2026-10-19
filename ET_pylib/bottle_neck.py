"""Congestion diagnostics: when are interconnectors effectively saturated,
and what congestion income accrues to the TSOs."""

from __future__ import annotations

import pandas as pd

from .groups import c_aggs, EA_TO_COUNTRY
from .energy_balance_build import active_line_export

# Physical-line utilisation is Phi((eta*p_j - p_i)/sigma_l): smooth, so it never
# binds. Saturation is a threshold choice, not a model event -- see docstrings.
DEFAULT_THRESHOLD = 0.98

SCAND_ZONES = tuple(c_aggs["Scandi"])
ZONE2COUNTRY = {ea: EA_TO_COUNTRY[ea] for ea in SCAND_ZONES}


def _scand(df, zones=SCAND_ZONES):
    """Keep only lines with both endpoints inside Scandinavia."""
    ea = df.index.get_level_values("EA")
    eaea = df.index.get_level_values("EAEA")
    return df[ea.isin(zones) & eaea.isin(zones)]


def line_utilisation(db, year, *, direction="Exp"):
    """Per-(line, direction, state) gross utilisation rate u in [0,1].
    u = lineExport / (hWeight * trCap * capF); trCap in TW, lineExport in TWh."""
    V = lambda s: db.symbols[s].vals.rename(s)

    le = active_line_export(db, years=[year])
    tc = V("trCap").reset_index().rename(columns={"trCap": "cap"})
    cf = V(f"capF{direction}").reset_index().rename(columns={f"capF{direction}": "cf"})
    hw = V("hWeight").reset_index()

    df = (le.merge(tc, on=["t", "line", "EA", "EAEA"], how="left")
            .merge(cf, on=["t", "line"], how="left")
            .merge(hw, on=["h"], how="left"))

    if df[["cap", "cf", "hWeight"]].isna().any().any():
        raise ValueError("merge left NaNs in cap/capF/hWeight -- check keys")

    df = df[df["cap"] > 0].copy()
    df["u"] = df["lineExport"] / (df["hWeight"] * df["cap"] * df["cf"])
    return df


def _prices(db, year):
    """Hourly zone price pE for `year`, as a (h, EA) -> pE frame."""
    p = db.symbols["pE"].vals.rename("pE").reset_index()
    return p[p["t"].astype(int) == int(year)][["h", "EA", "pE"]]


def saturated_hours(db, year, *, threshold=DEFAULT_THRESHOLD, lines=None):
    """Hours per year each line-direction runs at >= `threshold` of capacity,
    with zone prices averaged over those saturated hours only."""
    df = line_utilisation(db, year)
    if lines is not None:
        df = df[df["line"].isin(lines)]

    p = _prices(db, year)
    df = df.merge(p, on=["h", "EA"], how="left")
    df = df.merge(p.rename(columns={"EA": "EAEA", "pE": "pE_EAEA"}),
                  on=["h", "EAEA"], how="left")
    if df[["pE", "pE_EAEA"]].isna().any().any():
        raise ValueError("price merge left NaNs -- check pE zone coverage")

    df["hit"] = df["u"] >= threshold
    df["w"] = df["hWeight"] * df["hit"]          # weight: saturated hours only

    def _agg(x):
        w = x["w"].sum()
        if w == 0:                                # never saturated -> no price
            return pd.Series({"hours": 0.0, "cap_GW": x["cap"].iloc[0] * 1000,
                              "pris_EA": float("nan"), "pris_EAEA": float("nan")})
        return pd.Series({
            "hours": w,
            "cap_GW": x["cap"].iloc[0] * 1000,
            "pris_EA": (x["pE"] * x["w"]).sum() / w,
            "pris_EAEA": (x["pE_EAEA"] * x["w"]).sum() / w,
        })

    out = df.groupby(["line", "EA", "EAEA"]).apply(_agg, include_groups=False)
    out["spread"] = out["pris_EAEA"] - out["pris_EA"]
    out = out[out["hours"] > 0]
    return out.sort_values("hours", ascending=False)


# --------------------------------------------------------------------------- #
# congestion income
# --------------------------------------------------------------------------- #
def _ci_by_direction(db, year):
    """CI (mio. DKK) and flow (TWh) per line-direction -- the shared core.
    CI = (eta*p_EAEA - p_EA) * lineExport. lineExport is TWh, pre-summed over
    cluster hours -> plain sum over h, never hWeight. TWh * DKK/MWh = mio. DKK."""
    V = lambda s: db.symbols[s].vals.rename(s)

    # Only the rows the market clears on: lineExport also holds levels on inactive
    # line/area combinations (e.g. DK2<->DELU on the Bornholm line DELU__DK3).
    le = active_line_export(db, years=[year])
    eta = V("trEff").reset_index().rename(columns={"trEff": "eta"})
    p = V("pE").reset_index()
    p = p[p["t"].astype(int) == int(year)][["h", "EA", "pE"]]

    df = (le.merge(eta, on=["t", "line"], how="left")
            .merge(p, on=["h", "EA"], how="left")
            .merge(p.rename(columns={"EA": "EAEA", "pE": "pE_EAEA"}),
                   on=["h", "EAEA"], how="left"))
    if df[["eta", "pE", "pE_EAEA"]].isna().any().any():
        raise ValueError("merge left NaNs in eta/pE -- check keys")

    df["CI"] = (df["eta"] * df["pE_EAEA"] - df["pE"]) * df["lineExport"]
    return df.groupby(["line", "EA", "EAEA"]).agg(
        CI=("CI", "sum"), TWh=("lineExport", "sum")
    ).reset_index()


def congestion_income(db, year, *, tso_share=0.5):
    """Congestion income per line-direction and per EA (mio. DKK)."""
    d = _ci_by_direction(db, year)
    by_line = d.set_index(["line", "EA", "EAEA"]).sort_values("CI", ascending=False)

    half = by_line["CI"] * tso_share
    by_ea = (half.groupby("EA").sum()
               .add(half.groupby("EAEA").sum(), fill_value=0)
               .rename("CI_mioDKK").sort_values(ascending=False))
    return by_line, by_ea.to_frame()


def ci_by_line(db, year, *, scand_only=True):
    """Total CI per physical line (both directions summed), mio. DKK.
    This is the whole line's rent, not one TSO's share -- no halving."""
    d = _ci_by_direction(db, year)
    if scand_only:
        d = d[d["EA"].isin(SCAND_ZONES) | d["EAEA"].isin(SCAND_ZONES)]

    out = d.groupby("line").agg(CI_mioDKK=("CI", "sum"), TWh=("TWh", "sum"))
    out = out[out["TWh"] > 0]
    return out.sort_values("CI_mioDKK", ascending=False).round(1)


def ci_by_country(db, year, *, tso_share=0.5):
    """CI accruing to each country's TSOs (mio. DKK), plus a Scandi total.
    Each direction pays tso_share to the zone at each end; internal lines
    (e.g. Storebaelt) therefore return 100% to the country via both halves."""
    d = _ci_by_direction(db, year)
    half = d.assign(half=d["CI"] * tso_share)

    by_zone = (half.groupby("EA")["half"].sum()
                 .add(half.groupby("EAEA")["half"].sum(), fill_value=0))
    by_zone = by_zone[by_zone.index.isin(SCAND_ZONES)]

    zones = by_zone.rename("CI_mioDKK").to_frame()
    zones["land"] = zones.index.map(ZONE2COUNTRY)

    by_country = zones.groupby("land")["CI_mioDKK"].sum().sort_values(ascending=False)
    by_country.loc["Scandi i alt"] = by_country.sum()
    return zones.round(1), by_country.round(1).to_frame()