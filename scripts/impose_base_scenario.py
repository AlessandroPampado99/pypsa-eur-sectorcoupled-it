import pypsa

def impose_totalcapacity(n, technology, target_capacity, country):

    bus_country = n.buses.country

    # ---------- generators ----------
    gen_mask = (
        n.generators.carrier.str.contains(technology, case=False) &
        (n.generators.bus.map(bus_country) == country)
    )

    generators = n.generators.index[gen_mask]

    if len(generators) > 0:

        current_capacity = n.generators.loc[generators, "p_nom"].sum()

        scaling = target_capacity / current_capacity

        new_capacity = n.generators.loc[generators, "p_nom"] * scaling

        n.generators.loc[generators, "p_nom"] = new_capacity
        n.generators.loc[generators, "p_nom_min"] = new_capacity
        n.generators.loc[generators, "p_nom_max"] = new_capacity

        return


    # ---------- links ----------
    link_mask = (
        n.links.carrier.str.contains(technology, case=False) &
        (n.links.bus0.map(bus_country) == country)
    )

    links = n.links.index[link_mask]

    if len(links) > 0:

        current_capacity = n.links.loc[links, "p_nom"].sum()

        scaling = target_capacity / current_capacity

        new_capacity = n.links.loc[links, "p_nom"] * scaling

        n.links.loc[links, "p_nom"] = new_capacity
        n.links.loc[links, "p_nom_min"] = new_capacity
        n.links.loc[links, "p_nom_max"] = new_capacity

        return


    raise ValueError(f"No generators or links found for {technology} in {country}")

def store_energy_constraints(n, constraint_type, key, value):

    if "energy_constraints" not in n.meta:
        n.meta["energy_constraints"] = {}

    if constraint_type not in n.meta["energy_constraints"]:
        n.meta["energy_constraints"][constraint_type] = {}

    n.meta["energy_constraints"][constraint_type][key] = value


def apply_base_scenario(n, params):

    country = params["country"]
    n.meta["country"] = country

    for param, value in params.items():

        # -----------------
        # TOTAL CAPACITY
        # -----------------

        if param.endswith("_totalcapacity"):

            technology = param.replace("_totalcapacity","").replace("_"," ")

            impose_totalcapacity(
                n,
                technology,
                value,
                country
            )

        # ELECTRICITY EXCHANGE
        if param.endswith("_totalexchange"):

            country_out = param.replace("_totalexchange","")

            store_energy_constraints(
                n,
                "electricity_exchange",
                country_out,
                value
            )

        # GAS
        if param.endswith("_gas"):

            constraint = param.replace("_gas","")

            store_energy_constraints(
                n,
                "gas",
                constraint,
                value
            )

        # H2
        if param.endswith("_H2"):

            constraint = param.replace("_H2","")

            store_energy_constraints(
                n,
                "hydrogen",
                constraint,
                value
            )

# -----------------

n = pypsa.Network(snakemake.input.network)

scenario = snakemake.params.scenariobase
all_params = snakemake.params.scenario_params

if scenario != "none":

    params = all_params.get(scenario, {})

    apply_base_scenario(n, params)

n.export_to_netcdf(snakemake.output.network)