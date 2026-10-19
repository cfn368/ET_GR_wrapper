# ET_GR_wrapper

Erhvervslivets Tænketanks efterbehandling af stød i GreenREFORM (KF26). Modellen og
baselines er DREAMs. Her ligger ETT's kode til at læse de løste forløb, sammenligne
basisforløb med stødforløb og lave tabeller og figurer. Den nuværende analyse er
kernekraft mod havvind i det danske og nordiske elsystem med velfærdsåret 2040.

Modellen køres ikke herfra. Stød løses i ETT's kopi af DREAMs `2_Analysis.ipynb` med
`ET_shocks/` som undermappe. De løste forløb gemmes som pickles og analyseres her.
Det kræver hverken GAMS-licens eller DREAMs `pyGRE`.

## Indhold

```
0_ET_num_analysis.ipynb   tabeller: priser, handel, fangstpris, kapacitetsfaktor, CS, PS,
                          flaskehalsindtægt, fleksibilitet, velfærd, LCOE
0_ET_vis_analysis.ipynb   figurer: timebalance, priser, handel, værdifaktor, velfærd, produktion
ET_pylib/                 analysebiblioteket (se nedenfor)
ET_shocks/nucShock.py     kernekraftstødet til 2_Analysis (bruger DREAMs pyGRE, kører ikke lokalt)
ET_intermediate/<kørsel>/ løste forløb: baseline.pkl, shock_*.pkl (ikke i git)
```

## Kom i gang

**Miljø.** Python 3.13 (Anaconda) med `pandas`, `numpy`, `scipy`, `matplotlib`,
`openpyxl` samt `pyDatabases==0.2.1` og `gamsapi==54.2.2`. De to sidste skal til for at
unpickle GpyDB-objekterne.

```powershell
python -m pip install pyDatabases==0.2.1 gamsapi==54.2.2
python -m ipykernel install --user --name gr-post --display-name "Python 3.13 (GR post)"
```

En advarsel om at GAMS-versionen afviger fra API-versionen kan ignoreres, så længe der
kun læses pickles.

**Data.** Læg en kørsel i `ET_intermediate/<kørsel>/` med `baseline.pkl` og en
stødfil, fx `ET_intermediate/disaggEA_2040_kf25om/shock_nuc_2040.pkl`. Pickles,
gdx- og g00-filer er i `.gitignore`.

**Notebooks.** Vælg kernelen "Python 3.13 (GR post)". Første celle er altid

```python
from ET_pylib.setup import *
setup_notebook()
```

`setup` sætter autoreload og husstilen, definerer stierne (`ROOT`, `INTER_DIR`) og
importerer hele det offentlige API. Et forløb indlæses og bygges sådan:

```python
baseline_output, shock_output = load_run("disaggEA_2040_kf25om", "shock_nuc_2040.pkl")
ctx_el = build_el_balance_data(baseline_output, shock_output, years=2040)
print_trade_summary(ctx_el, countries="DK", years=2040)
```

Er der omdøbt eller flyttet moduler, skal kernelen genstartes. Autoreload kan ikke følge med.

## ET_pylib

Modulerne ligger i lag, og imports går kun nedad:

| lag | modul | indhold |
|---|---|---|
| konventioner | `groups.py` | zoner, landeaggregater, kernekraft-id'er, omdøbninger, kolonnekonventioner |
| | `style.py` | danske labels, farver, `FIG_TEXT`, `set_aej`, `dk_num`, `print_table` |
| hjælpere | `_util.py` | db-adgang, årsfilter, id-maps, `relabel_nuclear`, aktive linjer, prisbins |
| data | `balance.py` | timebalancer pr. zone, for Danmark samlet og for fjernvarme (`ElBalanceContext`) |
| | `market.py` | priser, knaphed, handel, bytteforhold, fangstpris, kapacitetsfaktor |
| | `costs.py` | kapacitetsbytte, renter og byggerenter, annuitet, systemomkostning, LCOE |
| | `welfare.py` | ΔCS, ΔPS med dekomponering, flaskehalsindtægt, fleksibilitet, velfærdsblok |
| figurer | `plot_base.py` | fælles figurmøblement |
| | `effect_figs.py` | søjlefigurer pr. region |
| | `balance_figs.py` | timesorterede balance- og nettoeksportfigurer |
| | `price_figs.py` | prisændring pr. time og ΔCS pr. prisbin |
| notebook | `setup.py` | preamble, stier, `load_run` |

## Konventioner

- **Basis og stød.** Hver analyse sammenligner `baseline` og `shock`. I tabellerne
  hedder kolonnerne `basis | stød | Δ`, og Δ er altid stød minus basis.
- **Alt printet går gennem `style.print_table`.** Det giver samme layout overalt, danske
  labels og dansk talformat. Notebook-celler printer ikke DataFrames selv. De kalder et
  `print_*`.
- **Én eller flere.** Alle argumenter, der tager flere værdier (`regions`, `countries`,
  `eas`, `areas`, `u_mins`, `partners`, `markets`, `stats`, `pts`), tager også én
  værdi. Et område kan være et aggregat (`"DK"`, `"Scandi"`) eller en zone (`"DK1"`),
  så `regions="DK1"` og `areas="DK"` virker begge.
- **Fejl højlydt.** Ukendte områder, manglende mappings og tomme årsudsnit giver en
  fejl i stedet for et tomt eller stille forkert resultat.
- **Enheder.** `SEh`, `lineExport` og `qDemandE_DCIIA` er allerede summeret over
  klyngetimer, så en almindelig sum giver årstotaler. Gang ikke med `hWeight`.
  `trCap` er i TW. FOM er nettet ind i `UP`, så annuiteten i velfærden er CAPEX
  alene (`col="annuitet"`).
- **Kernekraft-id'er findes kun i stødforløbet.** Brug `id2Pt` og `id2EA` fra
  `shock_output`, når stødets anlæg skal med.
- **Danmark** er DK1 + DK2 + DK3 (Energiø Bornholm).
