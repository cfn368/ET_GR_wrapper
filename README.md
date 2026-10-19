# ET_GR_wrapper

Erhvervslivets Tænketanks efterbehandling af stød i GreenREFORM (KF26). Modellen og
baselines er DREAMs. Koden her læser løste forløb, sammenligner basis med stød og laver
tabeller og figurer. Alt kører lokalt fra pickles: der skal hverken GAMS-licens eller
DREAMs `pyGRE` til.

## Analysen

Spørgsmålet er, hvad det er værd for Danmark at dække en del af elforsyningen med
kernekraft frem for havvind. Kernekraftstødet tilføjer nye anlæg oven på DREAMs
KF26-baseline, og effekten opgøres som en partiel velfærdsdekomponering i ét år, 2040:

ΔW = ΔCS + ΔPS + Δflaskehalsindtægt − annuitet

Opgørelsen dækker det danske og nordiske elsystem. Kernekraftens annuitet indeholder byggerenter, og fordi GR giver kernekraft
urealistisk fleksibilitet, korrigeres velfærden med en minimumslast på 30 %.
Resultaterne skal på sigt kunne kobles til et CGE-resultat.

## Indhold

```
0_ET_num_analysis.ipynb   tabeller: priser, handel, fangstpris, kapacitetsfaktor,
                          CS, PS, flaskehalsindtægt, fleksibilitet, velfærd, LCOE
0_ET_vis_analysis.ipynb   figurer: timebalance, priser, handel, værdifaktor, velfærd, produktion
ET_pylib/                 analysebiblioteket
ET_shocks/nucShock.py     kernekraftstødet (køres i ETT's kopi af DREAMs 2_Analysis.ipynb)
ET_intermediate/<kørsel>/ løste forløb: baseline.pkl, shock_*.pkl (ikke i git)
```

## Kom i gang

Python 3.13 med `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl` samt de to pakker,
der skal til for at unpickle GpyDB-objekterne:

```powershell
python -m pip install pyDatabases==0.2.1 gamsapi==54.2.2
python -m ipykernel install --user --name gr-post --display-name "Python 3.13 (GR post)"
```

En advarsel om GAMS- mod API-version kan ignoreres. Læg en kørsel i
`ET_intermediate/<kørsel>/`, vælg kernelen "Python 3.13 (GR post)" og start med

```python
from ET_pylib.setup import *
setup_notebook()

baseline_output, shock_output = load_run("disaggEA_2040_kf25om", "shock_nuc_2040.pkl")
ctx_el = build_el_balance_data(baseline_output, shock_output, years=2040)
print_trade_summary(ctx_el, countries="DK", years=2040)
```

Genstart kernelen, hvis moduler er omdøbt eller flyttet.

## ET_pylib

Modulerne står i lagrækkefølge, og imports går kun nedad:

```
ET_pylib/
├── groups.py         zoner, aggregater, kernekraft-id'er, omdøbninger
├── style.py          danske labels, farver, FIG_TEXT, dk_num, print_table
├── _util.py          db-adgang, årsfilter, id-maps, aktive linjer, prisbins
├── balance.py        timebalancer pr. zone, Danmark og fjernvarme (ElBalanceContext)
├── market.py         priser, knaphed, handel, bytteforhold, fangstpris, kapacitetsfaktor
├── costs.py          kapacitetsbytte, byggerenter, annuitet, systemomkostning, LCOE
├── welfare.py        ΔCS, ΔPS, flaskehalsindtægt, fleksibilitet, velfærdsblok
├── plot_base.py      fælles figurmøblement
├── effect_figs.py    søjlefigurer pr. region
├── balance_figs.py   timesorterede balance- og nettoeksportfigurer
├── price_figs.py     prisændring pr. time og ΔCS pr. prisbin
└── setup.py          preamble, stier, load_run
```

## Konventioner

- **Δ er altid stød minus basis**; tabelkolonner hedder `basis | stød | Δ`.
- **Alt printes via `style.print_table`** (danske labels og talformat). Notebooks printer
  ikke DataFrames selv.
- **Én eller flere.** Flerværdi-argumenter (`regions`, `countries`, `areas` osv.) tager
  også én værdi, og et område kan være et aggregat (`"DK"`) eller en zone (`"DK1"`).
- **Fejl højlydt** ved ukendte områder, manglende mappings og tomme årsudsnit.
- **Enheder.** `SEh`, `lineExport` og `qDemandE_DCIIA` er allerede summeret over
  klyngetimer, så gang ikke med `hWeight`. `trCap` er i TW. FOM ligger i `UP`, så
  annuiteten er CAPEX alene.
- **Kernekraft-id'er findes kun i stødforløbet**: brug `id2Pt` og `id2EA` fra `shock_output`.