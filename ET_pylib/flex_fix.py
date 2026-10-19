"""Value of nuclear's dispatch flexibility in GR: force the plants to run at u_min of
capacity in every hour and value the extra output at the model's prices."""

from .annuity import NUC_SYSCOST_ORE_KWH, ORE_KWH_PR_TWH
from .groups import NUCLEAR_IDS


def _long(db, sym, year):
    """Symbol for one year as a frame, index levels as str, t dropped."""
    df = db.symbols[sym].vals.rename(sym).reset_index()
    if "t" in df.columns:
        df = df[df["t"].astype(str) == str(year)].drop(columns="t")
    return df.astype({c: str for c in df.columns if c != sym})


def nuc_flex_value(db, *, year, u_min=1.0, syscost=NUC_SYSCOST_ORE_KWH):
    """Per nuclear id + total: output, extra TWh when forced to u_min of capacity,
    avg. price of the extra output and welfare change dW (mio. kr./yr).
    dW < 0; the value of the model's flexibility is -dW."""
    ids = [str(i) for i, start in NUCLEAR_IDS.items() if int(year) >= start]
    m = db.symbols["id2EA"].vals.to_frame(index=False).astype(str)
    m.columns = ["id", "EA"]

    df = _long(db, "SEh", year)
    df = df[df["id"].isin(ids)].merge(m, on="id")
    if set(df["id"]) != set(ids):
        raise KeyError(f"nuclear ids missing from SEh in {year}: {sorted(set(ids) - set(df['id']))}")
    for sym, on in (("pE", ["h", "EA"]), ("hWeight", ["h"]), ("ECap", ["id"]), ("MC", ["id"])):
        df = df.merge(_long(db, sym, year), on=on, how="left")
    if df[["pE", "hWeight", "ECap", "MC"]].isna().any().any():
        raise ValueError("missing pE, hWeight, ECap or MC after merge")

    q = df["SEh"] / 1000.0                                           # GWh -> TWh
    df["dq"] = (u_min * df["hWeight"] * df["ECap"] / 1000.0 - q).clip(lower=0.0)
    df["dW"] = (df["pE"] - df["MC"] - syscost * ORE_KWH_PR_TWH) * df["dq"]
    df["q"], df["p_dq"] = q, df["pE"] * df["dq"]

    s = df.groupby("id")[["q", "dq", "p_dq", "dW"]].sum()
    s.loc["total"] = s.sum()
    s["p_extra"] = s["p_dq"] / s["dq"]
    return s.rename(columns={"q": "TWh", "dq": "dTWh"})[["TWh", "dTWh", "p_extra", "dW"]]