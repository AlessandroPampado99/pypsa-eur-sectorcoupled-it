#!/usr/bin/env python3

import pypsa
import pandas as pd

from pypsa.statistics import get_transmission_carriers
from scripts.add_electricity import sanitize_carriers


# =====================================================
# LOAD NETWORK (Snakemake-native)
# =====================================================

n = pypsa.Network(snakemake.input.network)

# EXACT same preprocessing as plot_balance_map
sanitize_carriers(n, snakemake.config)

pypsa.options.params.statistics.nice_names = False
pypsa.options.params.statistics.drop_zero = True

# tech_colors is used ONLY to filter technologies
tech_colors = snakemake.config["plotting"]["tech_colors"]


# =====================================================
# FUNCTIONS – EXACT LOGIC FROM plot_balance_map.py
# =====================================================

def get_supply_consumption_map(n, bus_carrier, tech_colors):

    eb = n.statistics.energy_balance(
        bus_carrier=bus_carrier,
        groupby=["bus", "carrier"]
    )

    if eb.empty:
        return set(), set(), pd.Series(dtype=float)

    bus_sizes = (
        eb
        .groupby(level=["bus", "carrier"])
        .sum()
    )

    # Remove transmission carriers
    transmission = get_transmission_carriers(
        n, bus_carrier=bus_carrier
    ).rename({"name": "carrier"})

    transmission_carriers = set(transmission.unique("carrier"))

    bus_sizes = bus_sizes[
        ~bus_sizes.index.get_level_values("carrier").isin(transmission_carriers)
    ]

    # Remove the bus carrier itself
    bus_sizes = bus_sizes[
        bus_sizes.index.get_level_values("carrier") != bus_carrier
    ]

    # Keep only technologies shown in balance maps
    bus_sizes = bus_sizes[
        bus_sizes.index.get_level_values("carrier").isin(tech_colors)
    ]

    if bus_sizes.empty:
        return set(), set(), bus_sizes

    pos_carriers = bus_sizes[bus_sizes > 0].index.unique("carrier")
    neg_carriers = bus_sizes[bus_sizes < 0].index.unique("carrier")

    common = pos_carriers.intersection(neg_carriers)

    def total_abs(carrier, sign):
        vals = bus_sizes.loc[:, carrier]
        return vals[vals * sign > 0].abs().sum()

    supply = set(pos_carriers) - set(common)
    consumption = set(neg_carriers) - set(common)

    for c in common:
        if total_abs(c, +1) >= total_abs(c, -1):
            supply.add(c)
        else:
            consumption.add(c)

    return supply, consumption, bus_sizes


# =====================================================
# BUILD RECORDS (ONE ROW = ONE LEGEND ENTRY)
# =====================================================

records = []

bus_carriers = n.buses.carrier.unique()

for group in bus_carriers:

    supply_carriers, cons_carriers, bus_sizes = \
        get_supply_consumption_map(n, group, tech_colors)

    if bus_sizes.empty:
        continue

    # Supply
    for tech in supply_carriers:
        values = bus_sizes.loc[:, tech]
        value = values[values > 0].sum()
        if value == 0:
            continue

        records.append({
            "group": group,
            "kind": "Supply",
            "technology": tech,
            "value": value,
        })

    # Consumption
    for tech in cons_carriers:
        values = bus_sizes.loc[:, tech]
        value = (-values[values < 0]).sum()
        if value == 0:
            continue

        records.append({
            "group": group,
            "kind": "Consumption",
            "technology": tech,
            "value": value,
        })

df = pd.DataFrame(records)


# =====================================================
# FINALIZE (NO RANK / SHARE YET)
# =====================================================

def finalize(df):

    if df.empty:
        return pd.DataFrame(
            columns=["group", "rank", "technology", "value", "share [%]"]
        )

    out = []

    for group, g in df.groupby("group"):

        g = (
            g.groupby("technology", as_index=False)
            .value.sum()
            .sort_values("value", ascending=False)
            .reset_index(drop=True)
        )

        total = g.value.sum()
        if total == 0:
            continue

        g["group"] = group
        out.append(g)

    if not out:
        return pd.DataFrame(
            columns=["group", "rank", "technology", "value", "share [%]"]
        )

    df = pd.concat(out, ignore_index=True)
    df["rank"] = None
    df["share [%]"] = None

    return df[["group", "rank", "technology", "value", "share [%]"]]


supply = finalize(df[df.kind == "Supply"])
consumption = finalize(df[df.kind == "Consumption"])


# =====================================================
# ADD "OTHERS" + RECOMPUTE RANK & SHARE
# =====================================================

def add_others_and_recompute_shares(supply, consumption, tol=1e-6):

    supply = supply.copy()
    consumption = consumption.copy()

    groups = sorted(
        set(supply.group.unique()) | set(consumption.group.unique())
    )

    # ---- add "others" to close balance ----
    new_supply = []
    new_consumption = []

    for group in groups:

        s = supply.loc[supply.group == group, "value"].sum()
        c = consumption.loc[consumption.group == group, "value"].sum()

        diff = s - c

        if abs(diff) <= tol:
            continue

        if diff > 0:
            new_consumption.append({
                "group": group,
                "technology": "others",
                "value": diff,
            })
        else:
            new_supply.append({
                "group": group,
                "technology": "others",
                "value": -diff,
            })

    if new_supply:
        supply = pd.concat(
            [supply, pd.DataFrame(new_supply)],
            ignore_index=True
        )

    if new_consumption:
        consumption = pd.concat(
            [consumption, pd.DataFrame(new_consumption)],
            ignore_index=True
        )

    # ---- recompute rank and share including others ----
    def recompute(df):

        out = []

        for group, g in df.groupby("group"):

            g = (
                g.sort_values("value", ascending=False)
                .reset_index(drop=True)
            )

            total = g.value.sum()
            if total == 0:
                continue

            g["rank"] = g.index + 1
            g["share [%]"] = 100 * g.value / total
            g["group"] = group

            out.append(g)

        if not out:
            return pd.DataFrame(
                columns=["group", "rank", "technology", "value", "share [%]"]
            )

        return pd.concat(out, ignore_index=True)[
            ["group", "rank", "technology", "value", "share [%]"]
        ]

    return recompute(supply), recompute(consumption)


supply, consumption = add_others_and_recompute_shares(
    supply, consumption
)


# =====================================================
# CONSISTENCY CHECK (Excel == balance maps)
# =====================================================

def check_consistency(df, n, tech_colors, tol=1e-4):

    errors = []

    for group in df.group.unique():

        # ⬅️ EXCLUDE "others" from Excel side
        excel_total = df.loc[
            (df.group == group) & (df.technology != "others"),
            "value"
        ].sum()

        supply_carriers, cons_carriers, bus_sizes = \
            get_supply_consumption_map(n, group, tech_colors)

        if bus_sizes.empty:
            continue

        network_total = 0.0

        for tech in supply_carriers:
            vals = bus_sizes.loc[:, tech]
            network_total += vals[vals > 0].sum()

        for tech in cons_carriers:
            vals = bus_sizes.loc[:, tech]
            network_total += (-vals[vals < 0]).sum()

        if abs(excel_total - network_total) > tol:
            errors.append(
                f"[{group}] Excel={excel_total:.3e}, "
                f"BalanceMap={network_total:.3e}"
            )

    if errors:
        raise AssertionError(
            "❌ Balance-map consistency check FAILED:\n"
            + "\n".join(errors)
        )

    print("✔ Balance-map consistency check passed")

check_consistency(
    pd.concat([supply, consumption], ignore_index=True),
    n,
    tech_colors
)


# =====================================================
# WRITE PARQUET (LONG FORMAT, ONE FILE PER NETWORK)
# =====================================================

out_rows = []

for kind, df_kind in [("Supply", supply), ("Consumption", consumption)]:
    if df_kind.empty:
        continue
    tmp = df_kind.copy()
    tmp["kind"] = kind
    out_rows.append(tmp)

out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(
    columns=["group", "rank", "technology", "value", "share [%]", "kind"]
)

# IMPORTANT: make scenario unique across subfolders
scenario = getattr(snakemake.wildcards, "scenario", None)
if scenario is None:
    scenario = "__BASE__"
out["scenario"] = scenario


out = out[["scenario", "kind", "group", "rank", "technology", "value", "share [%]"]]
out.to_parquet(snakemake.output.parquet, index=False)

print(f"✔ Analysis written to {snakemake.output.parquet}")

