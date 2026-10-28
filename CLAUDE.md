# CLAUDE.md — ET_GR_wrapper

Post-processing of GreenREFORM (KF26) shocks for Erhvervslivets Tænketank (ET). Read
`README.md` first: it covers the analysis, environment, data layout, module layers and
the district-heat scenario. This file holds what the code does not say.

## Scope

- **The model never runs here.** Shocks are solved in ET's copy of DREAM's
  `2_Analysis.ipynb` next to the model. This repo only reads the solved pickles in
  `ET_intermediate/<run>/`. No GAMS licence, no `pyGRE`. Don't add code paths that need either.
- **DREAM's code stays out.** No `pyGRE`, DREAM notebooks or model files in the repo.
  `ET_shocks/nucShock.py` imports `pyGRE` only inside its functions, so it imports
  cleanly here. Keep it that way.
- **`ET_shocks/` and `ET_pylib/` run next to the model too.** `nucShock.py` imports names
  from `ET_pylib.groups` (`NUC_ID_BY_EA`, `NUC_HA_BY_EA`, `NUC_ID_BY_HA`, `WS_CF_2040`,
  `NUC_TWH_CHP_2040`) and `ET_pylib.costs`. The copy next to the model is updated by
  copying whole files, so after a change list the files that must be copied again.
- DREAM's fixed parameters (`sigmaID`, `sigmaCEt`, `sigmaCHt`, `pE_min`) are not touched.

## Scenarios

- **Main scenario:** nuclear as condensing plants (`CD`), ids `idNUC_DK1` / `idNUC_DK2`,
  pickle `shock_nuc_2040.pkl`. Defaults (notebook `SHOCK`, function defaults) stay on it.
- **Alternative:** nuclear as extraction CHP (`EX`), one unit per Danish heat area,
  ids `idNUC_<HA>`, pickle `shock_nuc_chp_2040.pkl`. Opt-in only: `chp=` in
  `nuc_plants`, `market="H"` in the wrapper. Present it as an alternative: the heat side
  (extraction, transmission pipes) carries no CAPEX.

## Environment

- Python 3.13 with the packages listed in `README.md`; notebooks run on the Jupyter kernel
  registered there (`gr-post`). Use the interpreter behind that kernel for scripts too.
  The Python that ships with GAMS is not the working environment.
- On Windows, set `PYTHONIOENCODING=utf-8` for any Python command whose output contains
  Danish text or `─`, `Δ`, `—`. Otherwise the console encoding can make it crash.
- Write multi-line Python scripts to a file and run the file. Shell heredocs can mangle
  backslashes and quotes, which breaks LaTeX, escape sequences and string edits.
- The notebooks may be open and edited in an editor while you work. Before overwriting a
  notebook, check `git status` and diff the sources for the user's edits. Build on them,
  never drop them.
- Notebooks use autoreload. Don't edit modules while a notebook executes, and restart the
  kernel after modules are copied in: a stale module keeps the old `NUCLEAR_IDS`.

## Architecture rules

- **Layers, imports one way:** `groups` → `style` → `_util` → `balance` →
  `market` / `costs` → `welfare` → `plot_base` → `effect_figs` / `balance_figs` /
  `price_figs`. `setup.py` imports from all of them. No lazy imports to dodge a cycle:
  a cycle means the function sits in the wrong module.
- **Each thing once.** Year filter (`filter_years`), symbol access (`sym_frame`,
  `sym_year`), id maps (`id_map`), nuclear relabel (`relabel_nuclear`), active lines,
  equal-hour bins, the trade guard (`require_el`) and figure furniture each have one home.
  Reuse them.
- **Where things go:**
  - Constants, sets, ids and market definitions go in `groups.py`.
  - Anything rendered (labels, colours, `FIG_TEXT`, `MARKET_TEXT`) goes in `style.py`. Plot
    and print code holds no string literals or colours of its own.
  - New public functions are exported from `setup.py`.
- **Markets.** Everything that differs between electricity and district heat lives in
  `groups.MARKETS` (`market_of("E" | "H")`). A context (`ElBalanceContext`) carries
  `.market`; functions that take a context follow it and raise on a mismatch. Functions
  that read model output directly take `market=`. Text that differs goes in `MARKET_TEXT`.
- **Printing.** Every printed table goes through `style.print_table`: title (unit),
  optional group header, `basis | stød | Δ`, Danish labels via `label_of`, Danish number
  format, `I alt` row. Δ columns are signed (`"+,.1f"`). Notebook cells call `print_*`;
  they never print DataFrames. Library functions print nothing unless they are a
  `print_*` or have `verbose=True`.
- **One or many.** Every multi-value argument also takes one value. Normalise with
  `groups.as_list`. An area can be an aggregate or a single area: use
  `areas_of(region, market)` / `expand_areas(areas, market)`, never `c_aggs[...]` or
  `HA_AGGS[...]` directly in library code.
- **Fail loudly.** Raise on unknown areas, unmapped ids, empty year cuts, merge holes and
  wrong-market contexts. No `try/except` fallbacks that degrade silently, no compatibility
  shims.
- Code, identifiers, docstrings and commits are in English. Everything rendered is Danish.
  Each module opens with a docstring (title, `=` underline, one paragraph); sections are
  marked `# ==================== ...` followed by `# N. name`; function docstrings are two
  to four lines, without `Args:`/`Returns:` blocks.

## Model conventions (easy to get wrong)

- `SEh`, `SHh`, `lineExport`, `qDemandE_DCIIA` are already summed over cluster hours: a
  plain sum is an annual total. Multiplying by `hWeight` overcounts about 8760×. `trCap`
  is in TW, `lineExport` in TWh.
- FOM is netted into `UP`, so the welfare annuity is CAPEX only (`col="annuitet"`).
  `col="total"` adds FOM a second time and is for standalone cost tables only.
- Nuclear ids exist only in the shock database. Take `id2Pt` / `id2EA` / `id2HA` from
  `shock_output` (or the union) when shock plants must be included. `NUCLEAR_IDS` lists
  both variants (six ids); functions use the ones the database carries.
- Denmark is DK1 + DK2 + DK3 (Energiø Bornholm). DK3 has no final demand
  (`AREAS_WITHOUT_DCIIA`) and no heat area. The heat market has four areas: `DK1_C`,
  `DK1_D`, `DK2_C`, `DK2_D`; heat regions are `DK1`, `DK2`, `DK`.
- `welfare_block` raises if a non-default `r=` is combined with a ready-made `annuity=`
  (the rate would be ignored). `nuc_u_min=0.3` is the default and adds the flexibility
  correction to ΔPS.
- An unpickled GpyDB is read with `db(sym)` or `db[sym].vals`; both return the values
  Series (checked on the KF26 pickles). A dict of databases is indexed first:
  `db_dict["baseline"](sym)`.

### CHP (`EX`) units

- `ECap` is electricity at the back-pressure point; the condensing nameplate is
  `ECap · (1 + Cv/Cm)` and `HCap = ECap / Cm`. Cost CAPEX, capacity factors and the
  minimum load on the condensing nameplate (`GW_cond` in `nuc_wind_swap`).
- DREAM's `AddedPlant` takes the **condensing** capacity as `ecap` for an EX plant and sets
  `ECap = ecap / (1 + Cv/Cm)` itself. `cm` and `cv` are plant fields, not `inputs`.
- `Eff`, `VOM` and `FOM` in GR are per MWh el / per GW `ECap` at the back-pressure point,
  so `ex_inputs` scales the condensing values by `1/(1 − x)`; fuel and FOM totals are
  unchanged. GR fixes `Cv = 1` on BB plants; on EX it is the electricity lost per MWh heat.
- EX ids have no `MC`. `MC_BP` / `MC_EX` are per composite unit; per MWh el in
  condensing mode the marginal cost is `MC_EX / (1 + Cv/Cm)`.
- The wind cut of the CHP scenario is energy-equivalent (`energy_cuts`) on hardcoded
  `WS_CF_2040` and `NUC_TWH_CHP_2040`. After a rerun, check Δ wind ≈ −Δ nuclear and
  update `NUC_TWH_CHP_2040` if not.

## Verifying a change

Restructuring must not move numbers. Figures are deterministic, so both text and PNGs
can be compared byte for byte:

1. Before the change, execute both notebooks headless with `nbclient` on kernel `gr-post`
   (cwd = repo root). Save every code cell's text output and an md5 of each `image/png`.
2. After the change, run again and diff per code cell. Any difference must be explained:
   an intended change, a moved warning line or one of the user's notebook edits.
3. Functions no notebook calls need their own before/after check on frames.
4. A change to the heat path is checked on the CHP pickle too: heat supply in the DK
   aggregate equals `qProdH_t` summed over the four heat areas, and ΔCS by price bin sums
   to the market's ΔCS.
5. Lint with `python -m pyflakes ET_pylib/*.py ET_shocks/*.py` (unused imports in
   `setup.py` and the re-export in `nucShock.py` are intended).

Both notebooks together take about 2 minutes. Run them in the background.

## Notebooks

- First cell is always `from ET_pylib.setup import *` / `setup_notebook()`.
- Minimal markdown: one H1 with a short intro, two part headers (`## Elmarkedet`,
  `## Fjernvarmemarkedet`), `*\`n. label\`*` sub-sections numbered through both parts,
  one italic descriptor line before each code cell, and a matching `# n. label` comment.
- End plot-only cells with `;` so figure reprs don't print.
- Keep executed output in the notebooks.

## Git

- Single branch: work and commit directly on `main`, in small described steps.
  Push only when asked.
- Pickles, gdx and g00 files stay out of git (`.gitignore`).
