"""Nuclear power shock for the KF26 stacked model, built on DREAM's kf26_added_plants."""
import numpy as np
import pandas as pd

from ET_pylib.groups import (
    NUC_ID_BY_EA as NUC_IDS, NUC_HA_BY_EA, NUC_ID_BY_HA, WS_CF_2040, NUC_TWH_CHP_2040,
)
from ET_pylib.costs import nuc_wind_swap  # noqa: F401  re-exported for 2_Analysis


def nuc_plants(gw, start_year, *, Eff, VOM, UranTax, FOM, sigmaID,
               ft="Ft21", bft="URA", end_year=None, chp=None):
    """New nuclear plants per area in `gw` ({EA: GWe}): condensing, or EX units per heat area
    with `chp` (see `chp_split`). Eff, VOM in DKK/MWh el and FOM in mio. DKK/GW/year refer
    to condensing operation; UranTax in DKK/MWh uranium input."""
    from pyGRE.makeModels.kf26_added_plants import AddedPlant
    unknown = set(gw) - set(NUC_IDS)
    if unknown:
        raise KeyError(f"No nuclear id for {sorted(unknown)}. Valid areas: {sorted(NUC_IDS)}")
    inputs = dict(
        Eff=Eff,
        VOM=VOM + UranTax / Eff,
        FOM=FOM,
        sigmaID=sigmaID,
        capFCap=1,
        CO2Cap=1,
        uCO2e_id=0,
        uNOx=0,
        Desulp=0,
        elTransportCost=0,
        fuelPriceAdjustment=0,
    )
    extra = {} if end_year is None else {"end_year": end_year}
    if chp is None:
        plants = [AddedPlant(id=NUC_IDS[ea], ea=ea, pt="CD", ecap=v, start_year=start_year,
                             ft=ft, bft=bft, inputs=dict(inputs), **extra)
                  for ea, v in gw.items() if v > 0]
    else:
        # AddedPlant takes the condensing capacity of an EX unit and sets ECap = ecap/(1 + Cv/Cm)
        # itself (checked on the KF26 CHP run), so the nameplate GW goes in, not u["ECap"].
        plants = [AddedPlant(id=NUC_ID_BY_HA[ha], ea=ea, ha=ha, pt="EX", ecap=u["GW"],
                             cm=u["Cm"], cv=u["Cv"], start_year=start_year, ft=ft, bft=bft,
                             inputs={**inputs, **ex_inputs(u, Eff=Eff, VOM=VOM, UranTax=UranTax,
                                                           FOM=FOM, **chp)}, **extra)
                  for ea, v in gw.items() if v > 0
                  for ha, u in chp_split(v, chp["shares"][ea], x=chp["x"], f=chp["f"],
                                         loss=chp["loss"]).items()]
    if not plants:
        raise ValueError("No nuclear capacity given.")
    return plants


def energy_cuts(gw, start_year, *, nuc_twh=NUC_TWH_CHP_2040, ws_cf=WS_CF_2040, pt="WS"):
    """Offshore wind cut whose lost production equals the nuclear el output `nuc_twh`
    (TWh/yr), split over areas by the nameplate in `gw` ({EA: GWe}): GW = TWh / (CF x 8.76)."""
    from pyGRE.makeModels.kf26_added_plants import fixed_cuts
    unknown = set(gw) - set(ws_cf)
    if unknown:
        raise KeyError(f"No wind capacity factor for {sorted(unknown)}. Known: {sorted(ws_cf)}")
    total = sum(v for v in gw.values() if v > 0)
    if total <= 0 or nuc_twh <= 0:
        raise ValueError(f"Need positive nuclear capacity and output, got {gw}, {nuc_twh} TWh")
    cut = {ea: nuc_twh * v / total / (ws_cf[ea] * 8.76) for ea, v in gw.items() if v > 0}
    return fixed_cuts(cut, pt=pt, start_year=start_year)


def heat_shares(db_base, year):
    """Share of each zone's district-heat demand (DHt_HA) per heat area in `year`:
    {EA: {HA: share}} over NUC_HA_BY_EA."""
    from pyGRE.makeModels.kf26_all_years import values_of
    dh = values_of(db_base, "DHt_HA").rename("DHt").reset_index()
    dh = dh[dh["t"].astype(int) == int(year)].set_index(dh["HA"].astype(str))["DHt"].astype(float)
    out = {}
    for ea, has in NUC_HA_BY_EA.items():
        v = dh.reindex(has)
        if v.isna().any() or v.sum() <= 0:
            raise ValueError(f"DHt_HA missing or zero for {list(has)} in {year}:\n{v.to_string()}")
        out[ea] = (v / v.sum()).to_dict()
    return out


def chp_split(gw, shares, *, x, f, loss):
    """EX units behind `gw` GWe nameplate, split by heat-area `shares`. A share x of the
    electricity goes to heat at f MWh heat per MWh el, less `loss` in transmission:
    HCap = f(1 - loss) x GW, ECap = (1 - x) GW, Cv = 1/(f(1 - loss)), Cm = ECap/HCap."""
    if not 0 < x < 1:
        raise ValueError(f"x must lie in (0, 1), got {x}")
    if f <= 0 or not 0 <= loss < 1:
        raise ValueError(f"Need f > 0 and 0 <= loss < 1, got f={f}, loss={loss}")
    if abs(sum(shares.values()) - 1) > 1e-9:
        raise ValueError(f"Heat-area shares must sum to 1, got {shares}")
    f_net = f * (1 - loss)
    return {ha: dict(GW=gw * s, ECap=(1 - x) * gw * s, HCap=f_net * x * gw * s,
                     Cm=(1 - x) / (f_net * x), Cv=1 / f_net, x=x)
            for ha, s in shares.items()}


def ex_inputs(unit, *, Eff, VOM, UranTax, FOM, sigmaID_EH, **_):
    """GR inputs of one EX unit (Cm and Cv go on the plant itself). Eff, VOM and FOM sit at
    the back-pressure point (per MWh el, per GW ECap), so they are scaled by 1/(1 - x);
    fuel use and FOM totals are unchanged."""
    k = 1 - unit["x"]
    return dict(Eff=Eff * k, VOM=(VOM + UranTax / Eff) / k, FOM=FOM / k,
                sigmaID_EH=sigmaID_EH)


def check_chp(db, plants):
    """Raise if an EX plant did not land in the model with its heat area, Cm, Cv and HCap."""
    from pyGRE.makeModels.kf26_all_years import values_of
    ids = [p.id for p in plants if p.pt == "EX"]
    if not ids:
        return
    ha = values_of(db, "id2HA").to_frame(index=False).astype(str)
    missing = {"id2HA": sorted(set(ids) - set(ha["id"]))}
    for sym in ("Cm", "Cv", "HCap"):
        have = set(values_of(db, sym).reset_index()["id"].astype(str))
        missing[sym] = sorted(set(ids) - have)
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise RuntimeError(f"EX nuclear plants incomplete in the model database: {missing}")


def check_cuts(db_base, cuts):
    """Raise if a cut exceeds the baseline capacity of its technology in its area in any year."""
    from pyGRE.makeModels.kf26_all_years import values_of
    pt = values_of(db_base, "id2Pt").to_frame(index=False).astype(str)
    ea = values_of(db_base, "id2EA").to_frame(index=False).astype(str)
    ecap = values_of(db_base, "ECap").rename("ECap").reset_index()
    ecap["id"] = ecap["id"].astype(str)
    ecap["t"] = ecap["t"].astype(int)
    ecap = ecap.merge(pt, on="id").merge(ea, on="id")
    model_years = sorted(ecap["t"].unique())
    for c in cuts:
        end = model_years[-1] if c.end_year is None else int(c.end_year)
        yrs = [y for y in model_years if int(c.start_year) <= y <= end]
        have = (ecap[(ecap["Pt"] == c.pt) & (ecap["EA"] == c.ea) & ecap["t"].isin(yrs)]
                .groupby("t")["ECap"].sum().reindex(yrs, fill_value=0.0))
        short = have[have < c.gw]
        if not short.empty:
            raise ValueError(f"Cut of {c.gw} GW {c.pt} in {c.ea} exceeds baseline capacity:\n{short.to_string()}")
        

def solve_nuc_shock(run, db_base, plants, cuts, years, steps=(0.25, 0.5, 0.75, 1.0), price_tol=0.01):
    """Add the plants, solve the zero-capacity control, then the full shock (walked in over
    `steps` if the full step is not accepted). Capacities and wind cuts are written onto run.db."""
    from pyGRE.makeModels.kf26_added_plants import (
        add_plants, capacity_text, apply_capacity, reduce_capacity, apply_levels,
    )
    from pyGRE.makeModels.kf26_all_years import values_of
    if steps[-1] != 1.0:
        raise ValueError("The last step must be 1.0.")
    check_cuts(db_base, cuts)
    add_plants(run.db, plants, years)
    check_chp(run.db, plants)
    run.absorb_new_keys = True

    control = run.solve_shock("", label="control, new plants at zero capacity")
    if not run.last_shock_accepted or not control["cleared"].all():
        raise RuntimeError("The control solve with the new plants at zero capacity did not clear.")
    base_pe = pd.to_numeric(values_of(db_base, "pE"), errors="coerce")
    ctrl_pe = pd.to_numeric(values_of(run.db, "pE"), errors="coerce").reindex(base_pe.index)
    gap = float((ctrl_pe - base_pe).abs().max())
    if not np.isfinite(gap) or gap > price_tol:
        raise RuntimeError(f"The control solve moved prices by {gap} DKK/MWh.")

    def text(step):
        wind_text, _ = reduce_capacity(db_base, cuts, step=step)
        return capacity_text(plants, step) + wind_text

    walked = []
    report = run.solve_shock(text(1.0), label="nuclear, full step")
    if run.last_shock_accepted and not report["cleared"].all():
        raise RuntimeError("The full step was accepted but did not clear.")
    if not run.last_shock_accepted:
        for step in steps:
            report = run.solve_shock(text(step), label=f"nuclear, step {step}")
            if not (run.last_shock_accepted and report["cleared"].all()):
                raise RuntimeError(f"Step {step} was not accepted.")
            walked.append(step)

    apply_capacity(run.db, plants, step=1.0)
    _, wind_levels = reduce_capacity(db_base, cuts, step=1.0)
    apply_levels(run.db, wind_levels)
    return dict(report=report, control=control, price_gap=gap, walked=walked)

