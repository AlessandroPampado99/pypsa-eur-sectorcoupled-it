#!/usr/bin/env python3

import pandas as pd


# =====================================================
# INPUT
# =====================================================

excel1 = snakemake.input.excel1
excel2 = snakemake.input.excel2

output = snakemake.output.excel

name1 = excel1.split("/")[-1].replace("_analysis.xlsx", "")
name2 = excel2.split("/")[-1].replace("_analysis.xlsx", "")


# =====================================================
# LOAD DATA
# =====================================================

supply_1 = pd.read_excel(excel1, sheet_name="Supply")
cons_1   = pd.read_excel(excel1, sheet_name="Consumption")

supply_2 = pd.read_excel(excel2, sheet_name="Supply")
cons_2   = pd.read_excel(excel2, sheet_name="Consumption")


# =====================================================
# GENERIC COMPARISON FUNCTION
# =====================================================

def compare(df1, df2):

    df1 = df1.copy()
    df2 = df2.copy()

    df1 = df1.rename(columns={
        "rank": "rank_1",
        "value": "value_1",
        "share [%]": "share_1"
    })

    df2 = df2.rename(columns={
        "rank": "rank_2",
        "value": "value_2",
        "share [%]": "share_2"
    })

    # Merge on group + technology
    merged = pd.merge(
        df1,
        df2,
        on=["group", "technology"],
        how="outer"
    )

    # Fill NaNs with 0 for values and shares
    for col in ["value_1", "value_2", "share_1", "share_2"]:
        merged[col] = merged[col].fillna(0.0)

    # Rank stays NaN if missing
    merged["rank_1"] = merged["rank_1"]
    merged["rank_2"] = merged["rank_2"]

    # Relative differences
    merged["rel_diff_value"] = (
        (merged["value_2"] - merged["value_1"]) / merged["value_2"]
    )

    merged["rel_diff_share"] = (
        (merged["share_2"] - merged["share_1"]) / merged["share_2"]
    )

    # Clean infinities (division by zero)
    merged.replace([float("inf"), -float("inf")], pd.NA, inplace=True)

    # Order columns
    merged = merged[[
        "group",
        "technology",
        "rank_1",
        "rank_2",
        "value_1",
        "value_2",
        "share_1",
        "share_2",
        "rel_diff_value",
        "rel_diff_share",
    ]]

    # Sort nicely
    merged = merged.sort_values(
        ["group", "value_2"],
        ascending=[True, False]
    )

    return merged


# =====================================================
# BUILD COMPARISON TABLES
# =====================================================

supply_comp = compare(supply_1, supply_2)
cons_comp   = compare(cons_1, cons_2)


# =====================================================
# WRITE EXCEL
# =====================================================

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    supply_comp.to_excel(writer, sheet_name="Supply", index=False)
    cons_comp.to_excel(writer, sheet_name="Consumption", index=False)

print(f"✔ Comparison written to {output}")
