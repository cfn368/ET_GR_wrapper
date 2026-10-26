# CLAUDE.md — ET_GR_wrapper

Post-processing of GreenREFORM (KF26) shocks for Erhvervslivets Tænketank (ET). Read
`README.md` first: it covers the analysis, environment, data layout and module layers.
This file holds what the code does not say.

## Scope

- **The model never runs here.** Shocks are solved in ETT's copy of DREAM's
  `2_Analysis.ipynb` on the shared model drive. This repo only reads the solved pickles in
  `ET_intermediate/<run>/`. No GAMS licence, no `pyGRE`. Don't add code paths that need either.
- **DREAM's code stays out.** No `pyGRE`, DREAM notebooks or model files in the repo.
  `ET_shocks/nucShock.py` imports `pyGRE` only inside its solve functions, so it imports
  cleanly here. Keep it that way.
- **`ET_shocks/` runs next to the model.** `nucShock.py` imports `ET_pylib.groups` and
  `ET_pylib.costs`. Any change to those names breaks the copy on the model drive unless
  `ET_pylib` is synced with it. Say so when a change touches them.
- DREAM's fixed parameters (`sigmaID`, `sigmaCEt`, `sigmaCHt`, `pE_min`) are not touched.

## Environment

- Python 3.13 with the packages listed in `README.md`; notebooks run on the Jupyter kernel
  registered there (`gr-post`). Use the interpreter behind that kernel for scripts too.
  The Python that ships with GAMS is not the working environment.
- On Windows, set `PYTHONIOENCODING=utf-8` for any Python command whose output contains
  Danish text or `─`, `Δ`, `—`. Otherwise the console encoding can make it crash.
- Write multi-line Python scripts to a file and run the file. Shell heredocs can mangle
  backslashes, which breaks LaTeX and escape sequences.
- The notebooks may be open and edited in an editor while you work. Before overwriting a
  notebook, check `git status` and diff the sources for the user's edits. Build on them,
  never drop them.

## Architecture rules

- **Layers, imports one way:** `groups` → `style` → `_util` → `balance` →
  `market` / `costs` → `welfare` → `plot_base` → `effect_figs` / `balance_figs` /
  `price_figs`. `setup.py` imports from all of them. No lazy imports to dodge a cycle:
  a cycle means the function sits in the wrong module.
- **Each thing once.** Year filter (`filter_years`), symbol access (`sym_frame`,
  `sym_year`), id maps (`id_map`), nuclear relabel (`relabel_nuclear`), active lines,
  equal-hour bins and figure furniture each have one home. Reuse them.
- **Where things go:**
  - Constants, sets and ids go in `groups.py`.
  - Anything rendered (labels, colours, `FIG_TEXT`) goes in `style.py`. Plot and print code
    holds no string literals or colours of its own.
  - New public functions are exported from `setup.py`.
- **Printing.** Every printed table goes through `style.print_table`: title (unit),
  optional group header, `basis | stød | Δ`, Danish labels via `label_of`, Danish number
  format, `I alt` row. Δ columns are signed (`"+,.1f"`). Notebook cells call `print_*`;
  they never print DataFrames. Library functions print nothing unless they are a
  `print_*` or have `verbose=True`.
- **One or many.** Every multi-value argument also takes one value. Normalise with
  `groups.as_list`. An area can be an aggregate or a zone: use `areas_of(region)` /
  `expand_areas(areas)`, never `c_aggs[region]` directly.
- **Fail loudly.** Raise on unknown areas, unmapped ids, empty year cuts and merge
  holes. No `try/except` fallbacks that degrade silently, no compatibility shims.
- Code, identifiers, docstrings and commits are in English. Everything rendered is Danish.
  Each module opens with a docstring (title, `=` underline, one paragraph); sections are
  marked `# ==================== ...` followed by `# N. name`; function docstrings are two
  to four lines, without `Args:`/`Returns:` blocks.

## Model conventions (easy to get wrong)

- `SEh`, `lineExport`, `qDemandE_DCIIA` are already summed over cluster hours: a plain sum
  is an annual total. Multiplying by `hWeight` overcounts about 8760×. `trCap` is in TW,
  `lineExport` in TWh.
- FOM is netted into `UP`, so the welfare annuity is CAPEX only (`col="annuitet"`).
- Nuclear ids (`idNUC_DK1`, `idNUC_DK2`) exist only in the shock database. Take `id2Pt` /
  `id2EA` from `shock_output` (or the union) when shock plants must be included.
- `welfare_block` raises if a non-default `r=` is combined with a ready-made `annuity=`
  (the rate would be ignored). `nuc_u_min=0.3` is the default and adds the flexibility
  correction to ΔPS.
- Denmark is DK1 + DK2 + DK3 (Energiø Bornholm). DK3 has no final demand
  (`AREAS_WITHOUT_DCIIA`).
- An unpickled GpyDB is read with `db(sym)` or `db[sym].vals`; both return the values
  Series (checked on the KF26 pickles). A dict of databases is indexed first:
  `db_dict["baseline"](sym)`.

## Verifying a change

Restructuring must not move numbers. Figures are deterministic, so both text and PNGs
can be compared byte for byte:

1. Before the change, execute both notebooks headless with `nbclient` on kernel `gr-post`
   (cwd = repo root). Save every code cell's text output and an md5 of each `image/png`.
2. After the change, run again and diff per code cell. Any difference must be explained:
   an intended change or one of the user's notebook edits.
3. Functions no notebook calls (the district-heat path: `build_dh_balance_data`,
   `heat_*`, `plot_energy_balance_dh`) need their own before/after check on frames.
4. Lint with `python -m pyflakes ET_pylib/*.py` (unused imports in `setup.py` are intended).

Both notebooks together take about 1.5 minutes. Run them in the background.

## Notebooks

- First cell is always `from ET_pylib.setup import *` / `setup_notebook()`.
- Minimal markdown: one H1 with a short intro, `*\`n. label\`*`
  sub-sections, one italic descriptor line before each code cell, and a matching
  `# n. label` comment.
- End plot-only cells with `;` so figure reprs don't print.
- Keep executed output in the notebooks.

## Git

- Single branch: work and commit directly on `main`, in small described steps.
  Push only when asked.
- Pickles, gdx and g00 files stay out of git (`.gitignore`).
