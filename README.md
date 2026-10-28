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

Opgørelsen dækker det danske og nordiske elsystem. Kernekraftens annuitet indeholder
byggerenter, og fordi GR giver kernekraft urealistisk fleksibilitet, korrigeres velfærden
med en minimumslast på 30 %. Resultaterne skal på sigt kunne kobles til et CGE-resultat.

**Hovedscenariet** er kernekraft som kondensværk, der kun producerer el. Som **alternativt
scenarie** kan kernekraften køres som udtagsværk, der også leverer fjernvarme (se nedenfor).

## Indhold

```
0_ET_num_analysis.ipynb   tabeller: priser, handel, fangstpris, kapacitetsfaktor, CS, PS,
                          flaskehalsindtægt, fleksibilitet, velfærd, LCOE; fjernvarmedel
0_ET_vis_analysis.ipynb   figurer: timebalance, priser, handel, værdifaktor, velfærd,
                          produktion; fjernvarmedel
ET_pylib/                 analysebiblioteket
ET_shocks/nucShock.py     kernekraftstødet (køres i ETT's kopi af DREAMs 2_Analysis.ipynb)
ET_intermediate/<kørsel>/ løste forløb: baseline.pkl, shock_*.pkl (ikke i git)
```

Begge notebooks har en eldel (`## Elmarkedet`) og en fjernvarmedel
(`## Fjernvarmemarkedet`). Stødet vælges med `SHOCK` i afsnit 1.

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

Genstart kernelen, når moduler er kopieret ind, omdøbt eller flyttet. Autoreload
genindlæser ikke alle afhængigheder, så fx listen over kernekraft-id'er kan blive forældet.

## Alternativt scenarie: kernekraft med fjernvarme

Kernekraften modelleres som udtagsværk (`EX`). Det kan køre som rent elværk eller afgive
damp til fjernvarme, så det ikke er bundet af varmebehovet om sommeren. Det er ikke
hovedscenariet: anlægsomkostningerne til varmesiden (udtag og transmissionsledninger)
indgår ikke i velfærden.

**Antagelser** (argumentet `chp` i `nucShock.nuc_plants`):

| Parameter | Værdi | Betydning |
|---|---|---|
| `x` | 0,15 | andel af elkapaciteten, der kan afgives til fjernvarme |
| `f` | 4,0 | MWh fjernvarme pr. MWh el, der afgives (Ea Energianalyse) |
| `loss` | 0,2 | varmetab i transmission fra værk til fjernvarmenet |
| `sigmaID_EH` | 10 | udglatning mellem modtryk og kondens, som GR's øvrige kraftvarmeværker |

Med 1 GWe kondenskapacitet giver det 0,85 GWe i modtryk og 0,48 GW fjernvarme netto.
I GR's parametre er `Cv = 1/(f(1 − loss))` og `Cm = (1 − x)/(f(1 − loss)x)`.

**Placering.** Fordelingen på DK1 og DK2 er som i hovedscenariet. Inden for hver zone
fordeles kapaciteten på det centrale og det decentrale fjernvarmeområde efter
fjernvarmeforbruget (`heat_shares`). Det giver fire værker: `idNUC_DK1_C`, `idNUC_DK1_D`,
`idNUC_DK2_C` og `idNUC_DK2_D`.

**Stødforhold.** Havvind skæres energiækvivalent: den tabte havvindproduktion svarer til
kernekraftens elproduktion (`energy_cuts`). Grundlaget er hardcodet i `groups.py`
(`WS_CF_2040`, `NUC_TWH_CHP_2040`). Hvis Δ havvind og Δ kernekraft afviger efter en
kørsel, opdateres `NUC_TWH_CHP_2040` med kernekraftens nye produktion, og stødet køres igen.

**Kørsel ved modellen** (i cellen med kernekraftværker og vindskæring i `2_Analysis.ipynb`):

```python
chp = dict(x=0.15, f=4.0, loss=0.2, sigmaID_EH=10,
           shares=ns.heat_shares(db_dict["baseline"], NUC_START))
plants = ns.nuc_plants({"DK1": 0.65, "DK2": 0.35}, NUC_START, Eff=0.33, VOM=20,
                       UranTax=15, FOM=1000, sigmaID=30, chp=chp)
cuts = ns.energy_cuts({"DK1": 0.65, "DK2": 0.35}, start_year=NUC_START)
```

Gem stødet som `shock_nuc_chp_2040.pkl`, så hovedscenariet ikke overskrives.

**Analyse her.** Sæt `SHOCK = "shock_nuc_chp_2040.pkl"` i afsnit 1 i begge notebooks.
Fjernvarmedelen bygger konteksten med `market="H"`; funktioner, der læser modeloutput
direkte, tager selv `market="H"`:

```python
ctx_dh = build_el_balance_data(baseline_output, shock_output, years=2040, market="H")
plot_price_levels(ctx_dh, years=2040)
plot_generation_delta(baseline_output, shock_output, id2Pt=id2Pt, id2EA=id2EA,
                      year=2040, regions="DK", market="H")
welfare_block_by_rate(baseline_output, shock_output, {"FM": "fm"}, year=2040,
                      regions="DK", markets=("E", "H"))
```

Velfærden medtager fjernvarmeforbrugernes overskud med `markets=("E", "H")`. Kernekraften
prissættes på sin kondenskapacitet (1 GWe), og varmesiden har ingen anlægsomkostning.

## ET_pylib

Modulerne står i lagrækkefølge, og imports går kun nedad:

```
ET_pylib/
├── groups.py         zoner, varmeområder, aggregater, kernekraft-id'er, markeder (E/H)
├── style.py          danske labels, farver, FIG_TEXT, MARKET_TEXT, dk_num, print_table
├── _util.py          db-adgang, årsfilter, id-maps, aktive linjer, prisbins
├── balance.py        timebalancer pr. område og for Danmark, el og fjernvarme
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
- **Marked.** `"E"` er el (zoner), `"H"` er fjernvarme (varmeområder). På varmesiden er
  regionerne `DK1`, `DK2` og `DK`, hver dækkende de centrale og decentrale varmeområder.
  Handel og flaskehalsindtægt findes kun på elmarkedet.
- **Fejl højlydt** ved ukendte områder, manglende mappings, tomme årsudsnit og en
  kontekst fra det forkerte marked.
- **Enheder.** `SEh`, `SHh`, `lineExport` og `qDemandE_DCIIA` er allerede summeret over
  klyngetimer, så gang ikke med `hWeight`. `trCap` er i TW. FOM ligger i `UP`, så
  annuiteten er CAPEX alene.
- **Kernekraft-id'er findes kun i stødforløbet**: brug `id2Pt` og `id2EA` fra `shock_output`.
