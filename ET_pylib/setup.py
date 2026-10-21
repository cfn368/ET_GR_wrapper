"""
Notebook setup — one import cell for the analysis notebooks
============================================================

`from ET_pylib.setup import *` followed by `setup_notebook()` gives a notebook
the standard stack, the repo paths, `load_run` and the public ET_pylib API.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ET_pylib.groups import (
    ANALYSIS_YEAR, NUCLEAR_IDS, NUCLEAR_IDS_BY_GEO, use_geography,
    c_aggs, c_aggs_countries, EA_TO_COUNTRY, STATES,
)
from ET_pylib.style import set_aej, col_names, area_colors, FIG_TEXT, print_table
from ET_pylib.balance import (
    ElBalanceContext, build_el_balance_data, build_el_balance_data_country, build_dh_balance_data,
)
from ET_pylib.market import (
    price_summary, scarcity_summary, full_summary, trade_summary, full_trade_summary,
    terms_of_trade_partner, print_terms_of_trade_partner, print_price_summary,
    print_trade_summary,
    capture_summary, capacity_factor_summary, print_capture, print_capacity_factor,
    heat_capture_summary, heat_capacity_factor_summary,
    print_heat_capture, print_heat_capacity_factor,
)
from ET_pylib.costs import (
    nuc_wind_swap, print_swap, swap_deltas, plant_cost, annuity_by_tech, annuity_factor,
    lcoe_inputs, lcoe_table, print_lcoe,
)
from ET_pylib.welfare import (
    consumer_surplus, print_cs_table, producer_profit, producer_generation,
    profit_decomposition, print_producer_profit, print_producer_qty,
    ci_by_country, print_ci, nuc_flex_value, print_flex, welfare_levels, cs_by_price_bin,
    welfare_block, welfare_block_by_rate, print_welfare_blocks, welfare_by_syscost_gap,
)
from ET_pylib.balance_figs import (
    build_supply_shares, plot_energy_balance_el, plot_energy_balance_dh, plot_net_export,
    plot_net_export_by_pt, plot_net_export_by_pt_agg,
)
from ET_pylib.effect_figs import (
    plot_price_levels, plot_price_delta, plot_trade_levels, plot_trade_delta_by_partner,
    plot_nx_delta_by_pt, plot_capture_delta_by_pt, plot_welfare_levels,
    plot_generation_stack, plot_profit_stack, plot_generation_delta, plot_profit_delta,
    plot_cs_by_price_bin, plot_welfare_by_syscost_gap,
)
from ET_pylib.price_figs import plot_price_change_and_cs, plot_price_change_scatter


# ==================== ==================== ==================== ====================
# 1. paths

ROOT      = Path(__file__).resolve().parents[1]
INTER_DIR = ROOT / "ET_intermediate"


def load_run(run, shock="shock_nuc_2040.pkl", *, baseline="baseline.pkl"):
    """(baseline_output, shock_output) from ET_intermediate/<run>/."""
    folder = INTER_DIR / run
    if not folder.is_dir():
        have = sorted(p.name for p in INTER_DIR.iterdir() if p.is_dir())
        raise FileNotFoundError(f"no run folder {folder}; ET_intermediate holds {have}")
    out = []
    for name in (baseline, shock):
        with open(folder / name, "rb") as f:
            out.append(pickle.load(f))
    return tuple(out)


# ==================== ==================== ==================== ====================
# 2. notebook environment

def enable_autoreload(mode: int = 2) -> None:
    try:
        from IPython import get_ipython
    except Exception:
        return
    ip = get_ipython()
    if ip is None:
        return
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", str(int(mode)))


def setup_notebook(*, autoreload: int = 2, aej: bool = True) -> None:
    enable_autoreload(autoreload)
    if aej:
        set_aej()
