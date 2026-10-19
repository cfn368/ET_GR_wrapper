"""Nuclear power shock for the KF26 stacked model, built on DREAM's kf26_added_plants."""
import numpy as np
import pandas as pd

NUC_IDS = {"DK1": "idNUC_DK1", "DK2": "idNUC_DK2"}


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


def nuc_wind_swap(baseline, shock, yr, dk=("DK1", "DK2"), wind=("WS",), verbose=True, nuc_ids=None):
    """Nuclear capacity added vs. offshore wind displaced under a shock.
    Returns (nuc, cut): per-id DataFrames."""
    nuc_ids = list(NUC_IDS.values()) if nuc_ids is None else list(nuc_ids)

    def xs(db, sym):
        v = db(sym).xs(yr, level="t").astype("float64")
        v.index = v.index.astype(str)
        return v

    def idmap(db, sym):
        v = db(sym)
        return pd.Series(v.get_level_values(1), index=v.get_level_values(0).astype(str))

    b, s, hcap = xs(baseline, "ECap"), xs(shock, "ECap"), xs(shock, "HCap")
    pt, ea = idmap(shock, "id2Pt"), idmap(shock, "id2EA")

    nuc = pd.DataFrame({"EA": ea.reindex(nuc_ids).values, "Pt": pt.reindex(nuc_ids).values,
                        "GW_base": b.reindex(nuc_ids, fill_value=0.0).values,
                        "GW_added": s.reindex(nuc_ids, fill_value=0.0).values,
                        "GW_heat": hcap.reindex(nuc_ids, fill_value=0.0).values}, index=nuc_ids)
    nuc = nuc[nuc["EA"].isin(dk)]

    w = [i for i in pt.index[pt.isin(wind)] if i not in nuc_ids]
    base_w, shock_w = b.reindex(w, fill_value=0.0), s.reindex(w, fill_value=0.0)
    cut = pd.DataFrame({"EA": ea.reindex(w).values, "Pt": pt.reindex(w).values,
                        "GW_base": base_w.values, "GW_cut": (base_w - shock_w).values},
                       index=w)
    cut = cut[cut["EA"].isin(dk)]

    if verbose:
        _print_swap(nuc, cut, yr, wind)
    return nuc, cut


def _print_swap(nuc, cut, yr, wind=("WS",)):
    from ET_pylib.groups import dk_num, col_names

    def f(v, signed=False):
        return dk_num(v, 2, signed)

    nuc_label = col_names.get("Nuclear", "Kernekraft")
    rows = [(f"  {nuc_label} {r.EA}", r.GW_base, r.GW_added)
            for _, r in nuc.sort_values("EA").iterrows()]
    agg = cut.groupby(["EA", "Pt"])[["GW_base", "GW_cut"]].sum()
    rows += [(f"  {col_names.get(p, p)} {a}", r.GW_base, r.GW_base - r.GW_cut)
             for (a, p), r in agg.iterrows()]

    lw = max([len(r[0]) for r in rows] + [28]) + 2
    rule = "─" * (lw + 30)
    print(f"Kapacitetsbytte, {yr} (GW)\n{rule}")
    print(f"{'':<{lw}}{'basis':>10}{'stød':>10}{'Δ':>10}\n{rule}")

    if nuc.GW_added.abs().sum() == 0:
        print(f"  Ingen kernekraft i {yr}.")
    for label, base, shock in rows:
        print(f"{label:<{lw}}{f(base):>10}{f(shock):>10}{f(shock - base, True):>10}")
    print(rule)

    d_nuc = float((nuc.GW_added - nuc.GW_base).sum())
    d_wind = float(-cut.GW_cut.sum())
    wind_label = col_names.get(wind[0], "Vind") if len(wind) == 1 else "Vind"
    print(f"{f'  {nuc_label} i alt':<{lw + 20}}{f(d_nuc, True):>10}")
    print(f"{f'  {wind_label} i alt':<{lw + 20}}{f(d_wind, True):>10}")
    if d_nuc > 0:
        print(f"{f'  {wind_label} fjernet pr. GW {nuc_label.lower()}':<{lw + 20}}"
              f"{dk_num(-d_wind / d_nuc, 2):>10}")
    if nuc.GW_heat.abs().sum() > 0:
        print(f"{f'  {nuc_label}, varme (GW-th)':<{lw + 20}}{f(nuc.GW_heat.sum(), True):>10}")
    print(rule)