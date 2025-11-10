# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Creates plots from summary CSV files.
"""

import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import re
import os

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # points to /dati/pampado/pypsa-eur
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import configure_logging, rename_techs, set_scenario_config
from scripts.prepare_sector_network import co2_emissions_year

logger = logging.getLogger(__name__)
plt.style.use("bmh")


# consolidate and rename

preferred_order = pd.Index(
    [
        "transmission lines",
        "hydroelectricity",
        "hydro reservoir",
        "run of river",
        "pumped hydro storage",
        "solid biomass",
        "biogas",
        "onshore wind",
        "offshore wind",
        "offshore wind (AC)",
        "offshore wind (DC)",
        "solar PV",
        "solar thermal",
        "solar rooftop",
        "solar",
        "building retrofitting",
        "ground heat pump",
        "air heat pump",
        "heat pump",
        "resistive heater",
        "power-to-heat",
        "gas-to-power/heat",
        "CHP",
        "OCGT",
        "gas boiler",
        "gas",
        "natural gas",
        "methanation",
        "ammonia",
        "hydrogen storage",
        "power-to-gas",
        "power-to-liquid",
        "battery storage",
        "hot water storage",
        "CO2 sequestration",
    ]
)

def _find_header_row(csv_path: str, startswith_tokens=("cost,", "component,")) -> int:
    """Return 0-based line index where the real header starts (after meta rows)."""
    pat = re.compile(rf"^({'|'.join(map(re.escape, startswith_tokens))})", re.IGNORECASE)
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if pat.match(line.strip()):
                return i
    # Fallback: assume no meta rows
    return 0

def load_nodal(csv_path: str, index_names: list[str]) -> pd.DataFrame:
    """
    Generic loader for nodal CSVs with meta-rows + a header line listing index columns
    (ending with a trailing comma), followed by rows with the numeric value as last field.
    Returns a DataFrame with MultiIndex=index_names and one numeric column 'value'.
    """
    hdr = _find_header_row(csv_path, startswith_tokens=(f"{index_names[0]}," , "cost,", "component,"))
    cols = index_names + ["value"]
    df = pd.read_csv(csv_path, skiprows=hdr + 1, header=None, names=cols)
    # coerce numeric and drop junk rows
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=index_names, how="any")
    return df.set_index(index_names)[["value"]]

def add_country_level(df: pd.DataFrame, level_name: str = "location") -> pd.DataFrame:
    """Append a 'country' level parsed from the first two letters of the 'location' level."""
    loc = df.index.get_level_values(level_name).astype(str)
    country = loc.str.extract(r"^([A-Z]{2})", expand=False).fillna("EU")
    df = df.set_index(pd.Index(country, name="country"), append=True)
    desired = ["country", *[n for n in df.index.names if n != "country"]]
    return df.reorder_levels(desired).sort_index()


def plot_costs_regional(outdir, tech_colors, threshold_billion=0.5):
    base_dir = REGIONAL_BASE

    df = load_nodal(snakemake.input.nodal_costs,
                    index_names=["cost","component","location","carrier"])
    df = add_country_level(df, level_name="location")            # add 'country'
    
    # total (CAPEX+OPEX) per country & carrier
    df_tot = df.groupby(["country","carrier"]).sum()
    df_tot = (df_tot / 1e9).rename_axis(index=["country","carrier"])  # EUR→bn

    df_tot = df_tot.rename(index=rename_techs, level="carrier")
    df_tot = df_tot.groupby(level=["country","carrier"]).sum()

    # drop small techs (threshold per carrier, across countries)
    to_drop = df_tot.groupby(level="carrier")["value"].max()
    to_drop = to_drop[to_drop < threshold_billion].index
    if len(to_drop) > 0:
        mask = df_tot.index.get_level_values("carrier").isin(to_drop)
        df_tot = df_tot[~mask]

    # plot per country
    for country, df_c in df_tot.groupby(level="country"):
        s = df_c.droplevel("country")["value"]
        if s.empty: 
            continue
        # colonne=una sola (scenario) → trasformiamo in Series→DataFrame per barre
        s = s.sum(axis=1) if isinstance(s, pd.DataFrame) else s
        
        # order preferito
        new_index = preferred_order.intersection(s.index).append(s.index.difference(preferred_order))

        country_dir = os.path.join(base_dir, country)
        os.makedirs(country_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        s = s.reindex(new_index)
        s.plot(kind="bar", ax=ax, color=[tech_colors.get(i, "grey") for i in s.index])
        ax.set_ylabel("System Cost [EUR billion per year]")
        ax.set_xlabel("")
        ax.grid(axis="x")
        fig.savefig(os.path.join(country_dir, f"costs_{country}.svg"), bbox_inches="tight")
        plt.close(fig)

def plot_capacities_regional(tech_colors, threshold_GW=0.5):
    """
    Make 3 per-country capacity plots:
      1) Generators
      2) Stores + StorageUnits
      3) Lines + Links
    Input: nodal_capacities.csv with index ['component','location','carrier'] and 'value' in MW.
    Saves per-country charts under dirname(snakemake.output.capacities).
    Also writes a system-wide summary to snakemake.output.capacities (as before).
    """
    base_dir = REGIONAL_BASE

    # 1) load nodal capacities (MW)
    df = load_nodal(
        snakemake.input.nodal_capacities,
        index_names=["component", "location", "carrier"],
    )
    # 2) add country
    df = add_country_level(df, level_name="location")

    # --- helper to aggregate, rename, threshold and plot per-country ---
    def _make_group_plots(name, components, ylabel="Capacity [GW]"):
        sub = df[df.index.get_level_values("component").isin(components)]
        if sub.empty:
            return

        # aggregate to [country, carrier] and convert MW -> GW
        tot = sub.groupby(["country", "carrier"]).sum()
        tot["value"] = tot["value"] / 1e3

        # rename techs on 'carrier' and re-aggregate
        tot = tot.rename(index=rename_techs, level="carrier")
        tot = tot.groupby(level=["country", "carrier"]).sum()

        # drop small techs (threshold per carrier across countries)
        to_drop = tot.groupby(level="carrier")["value"].max()
        to_drop = to_drop[to_drop < threshold_GW].index
        if len(to_drop) > 0:
            mask = tot.index.get_level_values("carrier").isin(to_drop)
            tot = tot[~mask]

        # per-country plots
        for country, df_c in tot.groupby(level="country"):
            s = df_c.droplevel("country")["value"]
            if s.empty:
                continue
            # preferred order
            new_index = preferred_order.intersection(s.index).append(
                s.index.difference(preferred_order)
            )
            s = s.reindex(new_index)

            country_dir = os.path.join(base_dir, country)
            os.makedirs(country_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            s.plot(kind="bar", ax=ax,
                   color=[tech_colors.get(i, "grey") for i in s.index])
            
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("")
            ax.grid(axis="x")
            # file name includes group tag
            fname = f"capacities_{name}_{country}.svg"
            fig.savefig(os.path.join(country_dir, fname), bbox_inches="tight")
            plt.close(fig)

        return tot  # return for optional system summary

    # 3) build the three groups
    sys_gen   = _make_group_plots("generators", {"Generator"})
    sys_store = _make_group_plots("storage", {"Store", "StorageUnit"})
    sys_grid  = _make_group_plots("network", {"Line", "Link"})


def plot_capacity_factors_regional(tech_colors, threshold_pct=1.0, agg="mean"):
    """
    Plot regional capacity factors aggregated by country and carrier, split in 3 sub-groups:
      1) Generators
      2) Stores + StorageUnits
      3) Lines + Links
    Input: nodal_capacity_factors.csv with index ['component','location','carrier'] and 'value' in p.u. (0..1)
    Output: per-country SVGs under dirname(snakemake.output.capacity_factors)
    """
    base_dir = REGIONAL_BASE

    # 1) Load nodal CF (p.u.)
    df = load_nodal(
        snakemake.input.nodal_capacity_factors,
        index_names=["component", "location", "carrier"],
    )
    # 2) Add country level
    df = add_country_level(df, level_name="location")

    # --- helper to handle each component group ---
    def _make_cf_group(name, components, ylabel="Capacity factor [%]"):
        sub = df[df.index.get_level_values("component").isin(components)]
        if sub.empty:
            return

        # aggregate to [country, carrier]
        if agg == "median":
            tot = sub.groupby(["country", "carrier"]).median(numeric_only=True)
        else:
            tot = sub.groupby(["country", "carrier"]).mean(numeric_only=True)

        # convert to percentage
        tot["value"] = tot["value"] * 100.0

        # rename and regroup
        tot = tot.rename(index=rename_techs, level="carrier")
        tot = tot.groupby(level=["country", "carrier"]).mean(numeric_only=True)

        # filter small CFs
        to_drop = tot.groupby(level="carrier")["value"].max()
        to_drop = to_drop[to_drop < threshold_pct].index
        if len(to_drop) > 0:
            mask = tot.index.get_level_values("carrier").isin(to_drop)
            tot = tot[~mask]

        # per-country plots
        for country, df_c in tot.groupby(level="country"):
            s = df_c.droplevel("country")["value"]
            if s.empty:
                continue

            # order carriers
            new_index = preferred_order.intersection(s.index).append(
                s.index.difference(preferred_order)
            )
            s = s.reindex(new_index)

            # ensure subdir and plot
            country_dir = os.path.join(base_dir, country)
            os.makedirs(country_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            s.plot(kind="bar", ax=ax, color=[tech_colors.get(i, "grey") for i in s.index])

            ax.set_ylabel(ylabel)
            ax.set_xlabel("")
            ax.grid(axis="x")

            fname = f"capacity_factors_{name}_{country}.svg"
            fig.savefig(os.path.join(country_dir, fname), bbox_inches="tight")
            plt.close(fig)

    # 3) Apply to the three groups
    _make_cf_group("generators", {"Generator"})
    _make_cf_group("storage", {"Store", "StorageUnit"})
    _make_cf_group("network", {"Line", "Link"})



def plot_balances_regional():
    """
    Read nodal energy balance (aggregated CSV), aggregate by country and bus_carrier,
    and create per-country bar charts (annual, not time-series).
    """
    co2_carriers = ["co2", "co2 stored", "process emissions"]

    base_dir = REGIONAL_BASE

    # 1) load nodal energy balance: index ['component','carrier','location','bus_carrier'] + 'value'
    df = load_nodal(
        snakemake.input.nodal_balances,
        index_names=["component", "carrier", "location", "bus_carrier"],
    )
    # 2) add 'country' from 'location'
    df = add_country_level(df, level_name="location")

    # 3) split dict per bus_carrier + 'energy' (tutti i carrier di bus messi insieme)
    by_bc = {bc: sub for bc, sub in df.groupby(level="bus_carrier")}
    by_bc["energy"] = df.groupby(level=["component", "carrier", "country"]).sum()

    # 4) loop per ciascun bus_carrier (o 'energy')
    for bus_carrier, df_bc in by_bc.items():
        # safety: assicurati che 'country' sia un livello
        if "country" not in df_bc.index.names:
            df_bc = add_country_level(df_bc, "location")

        # 5) collassa a [country, carrier] sommand(o component/location)
        df_bc = df_bc.groupby(level=["country", "carrier"]).sum()  # DataFrame con colonna 'value'

        # 6) MWh -> TWh
        df_bc["value"] = df_bc["value"] / 1e6

        # 7) rinomina tecnologie sul livello 'carrier' e ri-aggrego (perché alcune si accorpano)
        df_bc = df_bc.rename(index=rename_techs, level="carrier")
        df_bc = df_bc.groupby(level=["country", "carrier"]).sum()

        # 8) drop tech piccole (soglia come nel summary: threshold/10)
        thr = float(snakemake.params.plotting["energy_threshold"]) / 10.0
        # massimo assoluto per carrier (across countries)
        max_per_car = df_bc.groupby(level="carrier")["value"].max().abs()
        small = max_per_car[max_per_car < thr].index
        if len(small) > 0:
            mask = df_bc.index.get_level_values("carrier").isin(small)
            df_bc = df_bc[~mask]

        # 9) se vuoto, next
        if df_bc.empty:
            continue

        # 10) ordine preferito carrier
        all_carriers = df_bc.index.get_level_values("carrier").unique()
        order = preferred_order.intersection(all_carriers).append(
            all_carriers.difference(preferred_order)
        )

        # 11) serie “per country” e salvataggio in sottocartelle
        for country, df_c in df_bc.groupby(level="country"):
            s = df_c.droplevel("country")["value"]  # Series indicizzata per 'carrier'
            if s.empty:
                continue
            s = s.reindex([c for c in order if c in s.index])

            cdir = os.path.join(base_dir, country)
            os.makedirs(cdir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(12, 8))
            s.plot(
                kind="bar",
                ax=ax,
                color=[snakemake.params.plotting["tech_colors"].get(i, "grey") for i in s.index],
            )

            units = "MtCO2/a" if bus_carrier in co2_carriers else "TWh/a"
            ax.set_ylabel(f"Energy [{units}]")
            ax.set_xlabel("")
            ax.grid(axis="x")

            # legenda non necessaria (una barra per tecnologia); se vuoi mantenere coerenza con altri plot, puoi ometterla
            fig.savefig(os.path.join(cdir, f"balances_{bus_carrier}.svg"), bbox_inches="tight")
            plt.close(fig)



def historical_emissions(countries):
    """
    Read historical emissions to add them to the carbon budget plot.
    """
    # https://www.eea.europa.eu/data-and-maps/data/national-emissions-reported-to-the-unfccc-and-to-the-eu-greenhouse-gas-monitoring-mechanism-16
    # downloaded 201228 (modified by EEA last on 201221)
    df = pd.read_csv(snakemake.input.co2, encoding="latin-1", low_memory=False)
    df.loc[df["Year"] == "1985-1987", "Year"] = 1986
    df["Year"] = df["Year"].astype(int)
    df = df.set_index(
        ["Year", "Sector_name", "Country_code", "Pollutant_name"]
    ).sort_index()

    e = pd.Series()
    e["electricity"] = "1.A.1.a - Public Electricity and Heat Production"
    e["residential non-elec"] = "1.A.4.b - Residential"
    e["services non-elec"] = "1.A.4.a - Commercial/Institutional"
    e["rail non-elec"] = "1.A.3.c - Railways"
    e["road non-elec"] = "1.A.3.b - Road Transportation"
    e["domestic navigation"] = "1.A.3.d - Domestic Navigation"
    e["international navigation"] = "1.D.1.b - International Navigation"
    e["domestic aviation"] = "1.A.3.a - Domestic Aviation"
    e["international aviation"] = "1.D.1.a - International Aviation"
    e["total energy"] = "1 - Energy"
    e["industrial processes"] = "2 - Industrial Processes and Product Use"
    e["agriculture"] = "3 - Agriculture"
    e["LULUCF"] = "4 - Land Use, Land-Use Change and Forestry"
    e["waste management"] = "5 - Waste management"
    e["other"] = "6 - Other Sector"
    e["indirect"] = "ind_CO2 - Indirect CO2"
    e["other LULUCF"] = "4.H - Other LULUCF"

    pol = ["CO2"]  # ["All greenhouse gases - (CO2 equivalent)"]
    if "GB" in countries:
        countries.remove("GB")
        countries.append("UK")

    year = df.index.levels[0][df.index.levels[0] >= 1990]

    missing = pd.Index(countries).difference(df.index.levels[2])
    if not missing.empty:
        logger.warning(
            f"The following countries are missing and not considered when plotting historic CO2 emissions: {missing}"
        )
        countries = pd.Index(df.index.levels[2]).intersection(countries)

    idx = pd.IndexSlice
    co2_totals = (
        df.loc[idx[year, e.values, countries, pol], "emissions"]
        .unstack("Year")
        .rename(index=pd.Series(e.index, e.values))
    )

    co2_totals = (1 / 1e6) * co2_totals.groupby(level=0, axis=0).sum()  # Gton CO2

    co2_totals.loc["industrial non-elec"] = (
        co2_totals.loc["total energy"]
        - co2_totals.loc[
            [
                "electricity",
                "services non-elec",
                "residential non-elec",
                "road non-elec",
                "rail non-elec",
                "domestic aviation",
                "international aviation",
                "domestic navigation",
                "international navigation",
            ]
        ].sum()
    )

    emissions = co2_totals.loc["electricity"]
    if options["transport"]:
        emissions += co2_totals.loc[[i + " non-elec" for i in ["rail", "road"]]].sum()
    if options["heating"]:
        emissions += co2_totals.loc[
            [i + " non-elec" for i in ["residential", "services"]]
        ].sum()
    if options["industry"]:
        emissions += co2_totals.loc[
            [
                "industrial non-elec",
                "industrial processes",
                "domestic aviation",
                "international aviation",
                "domestic navigation",
                "international navigation",
            ]
        ].sum()
    return emissions


def plot_carbon_budget_distribution(input_eurostat, options):
    """
    Plot historical carbon emissions in the EU and decarbonization path.
    """
    import seaborn as sns

    sns.set()
    sns.set_style("ticks")
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20

    emissions_scope = snakemake.params.emissions_scope
    input_co2 = snakemake.input.co2

    # historic emissions
    countries = snakemake.params.countries
    e_1990 = co2_emissions_year(
        countries,
        input_eurostat,
        options,
        emissions_scope,
        input_co2,
        year=1990,
    )
    emissions = historical_emissions(countries)
    # add other years https://sdi.eea.europa.eu/data/0569441f-2853-4664-a7cd-db969ef54de0
    emissions.loc[2019] = 3.414362
    emissions.loc[2020] = 3.092434
    emissions.loc[2021] = 3.290418
    emissions.loc[2022] = 3.213025

    if snakemake.config["foresight"] == "myopic":
        path_cb = "results/" + snakemake.params.RDIR + "/csvs/"
        co2_cap = pd.read_csv(path_cb + "carbon_budget_distribution.csv", index_col=0)[
            ["cb"]
        ]
        co2_cap *= e_1990
    else:
        supply_energy = pd.read_csv(
            snakemake.input.balances, index_col=[0, 1, 2], header=[0, 1, 2, 3]
        )
        co2_cap = (
            supply_energy.loc["co2"].droplevel(0).drop("co2").sum().unstack().T / 1e9
        )
        co2_cap.rename(index=lambda x: int(x), inplace=True)

    plt.figure(figsize=(10, 7))
    gs1 = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs1[0, 0])
    ax1.set_ylabel("CO$_2$ emissions \n [Gt per year]", fontsize=22)
    # ax1.set_ylim([0, 5])
    ax1.set_xlim([1990, snakemake.params.planning_horizons[-1] + 1])

    ax1.plot(emissions, color="black", linewidth=3, label=None)

    # plot committed and under-discussion targets
    # (notice that historical emissions include all countries in the
    # network, but targets refer to EU)
    ax1.plot(
        [2020],
        [0.8 * emissions[1990]],
        marker="*",
        markersize=12,
        markerfacecolor="black",
        markeredgecolor="black",
    )

    ax1.plot(
        [2030],
        [0.45 * emissions[1990]],
        marker="*",
        markersize=12,
        markerfacecolor="black",
        markeredgecolor="black",
    )

    ax1.plot(
        [2030],
        [0.6 * emissions[1990]],
        marker="*",
        markersize=12,
        markerfacecolor="black",
        markeredgecolor="black",
    )

    ax1.plot(
        [2050, 2050],
        [x * emissions[1990] for x in [0.2, 0.05]],
        color="gray",
        linewidth=2,
        marker="_",
        alpha=0.5,
    )

    ax1.plot(
        [2050],
        [0.0 * emissions[1990]],
        marker="*",
        markersize=12,
        markerfacecolor="black",
        markeredgecolor="black",
        label="EU committed target",
    )

    for col in co2_cap.columns:
        ax1.plot(co2_cap[col], linewidth=3, label=col)

    ax1.legend(
        fancybox=True, fontsize=18, loc=(0.01, 0.01), facecolor="white", frameon=True
    )

    plt.grid(axis="y")
    path = snakemake.output.balances.split("balances")[0] + "carbon_budget.svg"
    plt.savefig(path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("plot_summary_regional",
                                   configfiles=["config/sector-coupled-test/config_validation.yaml"],
                                   run="italy__nuts3_pp",)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    REGIONAL_BASE = getattr(snakemake.output, "regional_dir", None)

    n_header = 3

    tech_colors = snakemake.params.plotting["tech_colors"]
    outdir = snakemake.output[0]

    plot_costs_regional(outdir, tech_colors, threshold_billion=snakemake.params.plotting["costs_threshold"])
    plot_capacities_regional(tech_colors,
                         threshold_GW=snakemake.params.plotting.get("capacities_threshold", 0.05))

    plot_capacity_factors_regional(tech_colors,
                               threshold_pct=snakemake.params.plotting.get("cf_threshold", 0.1),
                               agg=snakemake.params.plotting.get("cf_agg", "mean"))
    plot_balances_regional()

    co2_budget = snakemake.params["co2_budget"]
    if (
        isinstance(co2_budget, str) and co2_budget.startswith("cb")
    ) or snakemake.params["foresight"] == "perfect":
        options = snakemake.params.sector
        plot_carbon_budget_distribution(snakemake.input.eurostat, options)
