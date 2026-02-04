#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pandas as pd
import numpy as np


REQUIRED_COLS = ["scenario", "kind", "group", "rank", "technology", "value", "share [%]"]


def _load_all(parquets: list[str]) -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=REQUIRED_COLS)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in concatenated parquet: {missing}")

    # sanitize dtypes
    df["scenario"] = df["scenario"].astype(str)
    df["kind"] = df["kind"].astype(str)
    df["group"] = df["group"].astype(str)
    df["technology"] = df["technology"].astype(str)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    df["share [%]"] = pd.to_numeric(df["share [%]"], errors="coerce").fillna(0.0)

    return df


def _levels_wide(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    d = df[df["kind"] == kind].copy()

    if d.empty:
        return pd.DataFrame(columns=["group", "technology"])

    # pivot value/share/rank to wide
    base_idx = ["group", "technology"]

    wide_value = d.pivot_table(index=base_idx, columns="scenario", values="value", aggfunc="sum", fill_value=0.0)
    wide_share = d.pivot_table(index=base_idx, columns="scenario", values="share [%]", aggfunc="sum", fill_value=0.0)
    wide_rank  = d.pivot_table(index=base_idx, columns="scenario", values="rank", aggfunc="min")

    # flatten column names
    wide_value.columns = [f"value__{c}" for c in wide_value.columns]
    wide_share.columns = [f"share__{c}" for c in wide_share.columns]
    wide_rank.columns  = [f"rank__{c}" for c in wide_rank.columns]

    out = pd.concat([wide_rank, wide_value, wide_share], axis=1).reset_index()

    # sort by max value across scenarios
    vcols = [c for c in out.columns if c.startswith("value__")]
    out["_max_value"] = out[vcols].max(axis=1) if vcols else 0.0
    out = out.sort_values(["group", "_max_value"], ascending=[True, False]).drop(columns=["_max_value"])

    return out


def _delta_vs_base(levels: pd.DataFrame, base: str) -> pd.DataFrame:
    out = levels.copy()
    eps = 1e-12

    base_v = f"value__{base}"
    base_s = f"share__{base}"

    if base_v not in out.columns or base_s not in out.columns:
        raise ValueError(
            f"Base scenario '{base}' not present. "
            f"Have value cols: {[c for c in out.columns if c.startswith('value__')]}"
        )

    scenarios = sorted({c.split("__", 1)[1] for c in out.columns if c.startswith("value__")})

    for sc in scenarios:
        v = f"value__{sc}"
        s = f"share__{sc}"

        out[f"delta_value__{sc}"] = out[v] - out[base_v]
        out[f"delta_share__{sc}"] = out[s] - out[base_s]

        denom_v = np.maximum(np.abs(out[base_v].to_numpy()), eps)
        denom_s = np.maximum(np.abs(out[base_s].to_numpy()), eps)

        out[f"relchg_value__{sc}"] = (out[v] - out[base_v]) / denom_v
        out[f"relchg_share__{sc}"] = (out[s] - out[base_s]) / denom_s

    # compact column order
    front = ["group", "technology", base_v, base_s]
    delta_cols = [c for c in out.columns if c.startswith("delta_")]
    rel_cols = [c for c in out.columns if c.startswith("relchg_")]

    keep = [c for c in front + delta_cols + rel_cols if c in out.columns]
    return out[keep]


def main():
    parquets = list(snakemake.input.parquets)
    output = snakemake.output.excel
    base = snakemake.params["base_scenario"] 

    df = _load_all(parquets)

    supply_levels = _levels_wide(df, "Supply")
    cons_levels   = _levels_wide(df, "Consumption")

    supply_delta = _delta_vs_base(supply_levels, base)
    cons_delta   = _delta_vs_base(cons_levels, base)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        supply_levels.to_excel(writer, sheet_name="Supply_levels", index=False)
        cons_levels.to_excel(writer, sheet_name="Consumption_levels", index=False)
        supply_delta.to_excel(writer, sheet_name="Supply_vs_base", index=False)
        cons_delta.to_excel(writer, sheet_name="Consumption_vs_base", index=False)

    print(f"✔ Wrote single consolidated Excel to {output}")
    print(f"✔ Base scenario: {base}")


main()
