#!/usr/bin/env python3

from pypsa.statistics import get_transmission_carriers
from scripts.add_electricity import sanitize_carriers
import pypsa
import pandas as pd

# =====================================================
# LOAD NETWORK (Snakemake-native)
# =====================================================

n = pypsa.Network(snakemake.input.network)

# EXACT same preprocessing as plot_balance_map
sanitize_carriers(n, snakemake.config)

pypsa.options.params.statistics.nice_names = False
pypsa.options.params.statistics.drop_zero = True

print(
    sorted(
        n.statistics.energy_balance(
            bus_carrier="AC",
            groupby=["bus", "carrier"]
        ).index.get_level_values("carrier").unique()
    )
)

# tech_colors is used ONLY to filter technologies
tech_colors = snakemake.config["plotting"]["tech_colors"]

# =====================================================
# FUNCTIONS – EXACT LOGIC FROM plot_balance_map.py
# =====================================================

from pypsa.statistics import get_transmission_carriers


def get_supply_consumption_map(n, bus_carrier, tech_colors):
    """
    Replicates EXACTLY the supply/consumption logic of plot_balance_map.py
    for a given bus carrier, but adapted to an already aggregated
    energy_balance (no component level).

    Returns
    -------
    supply_carriers : set
    consumption_carriers : set
    bus_sizes : pd.Series
        Indexed by (bus, carrier), values are energy balances.
    """

    # --- energy balance (same call as plot_balance_map) ---
    eb = n.statistics.energy_balance(
        bus_carrier=bus_carrier,
        groupby=["bus", "carrier"]
    )

    if eb.empty:
        return set(), set(), pd.Series(dtype=float)

    # aggregate over buses (same as plot)
    bus_sizes = (
        eb
        .groupby(level=["bus", "carrier"])
        .sum()
    )

    # =====================================================
    # FILTERING (logical equivalent of plot_balance_map)
    # =====================================================

    # 1. Remove transmission carriers (lines, links, transformers)
    transmission = get_transmission_carriers(
        n, bus_carrier=bus_carrier
    ).rename({"name": "carrier"})

    transmission_carriers = set(transmission.unique("carrier"))

    bus_sizes = bus_sizes[
        ~bus_sizes.index.get_level_values("carrier").isin(transmission_carriers)
    ]

    # 2. Remove the bus carrier itself (AC, DC, co2, ...)
    bus_sizes = bus_sizes[
        bus_sizes.index.get_level_values("carrier") != bus_carrier
    ]

    # 3. Keep only technologies shown in balance maps
    bus_sizes = bus_sizes[
        bus_sizes.index.get_level_values("carrier").isin(tech_colors)
    ]

    if bus_sizes.empty:
        return set(), set(), bus_sizes

    # =====================================================
    # SUPPLY / CONSUMPTION CLASSIFICATION (exact logic)
    # =====================================================

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
# FINAL TABLES
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

        g["rank"] = g.index + 1
        g["share [%]"] = 100 * g.value / total
        g["group"] = group

        out.append(g)

    if not out:
        return pd.DataFrame(
            columns=["group", "rank", "technology", "value", "share [%]"]
        )

    return pd.concat(out)[["group", "rank", "technology", "value", "share [%]"]]


supply = finalize(df[df.kind == "Supply"])
consumption = finalize(df[df.kind == "Consumption"])

# =====================================================
# CONSISTENCY CHECK (Excel == balance maps)
# =====================================================

def check_consistency(df, n, tech_colors, tol=1e-4):

    errors = []

    for group in df.group.unique():

        # Excel total
        excel_total = df.loc[df.group == group, "value"].sum()

        # Recompute balance-map logic
        supply_carriers, cons_carriers, bus_sizes = \
            get_supply_consumption_map(n, group, tech_colors)

        if bus_sizes.empty:
            continue

        network_total = 0.0

        # Supply part
        for tech in supply_carriers:
            if tech not in bus_sizes.index.get_level_values("carrier"):
                continue
            vals = bus_sizes.loc[:, tech]
            network_total += vals[vals > 0].sum()

        # Consumption part
        for tech in cons_carriers:
            if tech not in bus_sizes.index.get_level_values("carrier"):
                continue
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
# WRITE EXCEL
# =====================================================

with pd.ExcelWriter(snakemake.output.excel, engine="openpyxl") as writer:
    supply.to_excel(writer, sheet_name="Supply", index=False)
    consumption.to_excel(writer, sheet_name="Consumption", index=False)

print(f"✔ Analysis written to {snakemake.output.excel}")