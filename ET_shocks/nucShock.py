"""Nuclear power shock for the KF26 stacked model, built on DREAM's kf26_added_plants."""
import numpy as np
import pandas as pd

from ET_pylib.groups import NUC_ID_BY_EA as NUC_IDS
from ET_pylib.costs import nuc_wind_swap  # noqa: F401  re-exported for 2_Analysis


def nuc_plants(gw, start_year, *, Eff, VOM, UranTax, FOM, sigmaID,
               ft="Ft21", bft="URA", end_year=None):
    """New condensing nuclear plants, one per area in `gw` ({EA: GWe}).
    VOM in DKK/MWh el, UranTax in DKK/MWh uranium input, FOM in mio. DKK/GW/year."""
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
    plants = [AddedPlant(id=NUC_IDS[ea], ea=ea, pt="CD", ecap=v, start_year=start_year,
                         ft=ft, bft=bft, inputs=dict(inputs), **extra)
              for ea, v in gw.items() if v > 0]
    if not plants:
        raise ValueError("No nuclear capacity given.")
    return plants


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

