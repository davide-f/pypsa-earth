# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Creates electric demand profile csv.

Experimental copy with demand filling; current rules use the original.
To integrate later, change both the Snakefile import of get_load_paths_gegis
and the rule script path to this module, so reference-country files become
explicit input dependencies. DemandCast already downloads a global file.

Example configuration (totals from the archived NG run's 37 GADM shapes)::

    load_options:
      fill_missing:
        base_country: "NG"
        gdp: 1194938546688.0
        pop: 206119089.90968263

GDP and population must use the same datasets and units as the target shapes.
The fill_missing mapping is required. A country with no non-null demand at
the target snapshots uses the reference profile scaled by the weighted GDP and
population ratios. With the default weights, the multiplier is
0.6 * target_gdp / reference_gdp + 0.4 * target_pop / reference_pop.
Target totals sum the administrative shapes for each country. Bus allocation
and load scaling also apply to the synthesized profile. Timestamp alignment
and gap repair take place later in upsample().

Relevant Settings
-----------------

.. code:: yaml

    load_options:
        source:
        scale:
        weather_year:
        prediction_year:
        fill_missing:
            base_country:
            gdp:
            pop:

Inputs
------

- ``networks/base.nc``: confer :ref:`base`, a base PyPSA Network
- ``resources/bus_regions/regions_onshore.geojson``: confer :mod:`build_bus_regions`
- ``load_data_paths``: paths to load profiles, e.g. hourly country load profiles produced by GEGIS
- ``resources/shapes/gadm_shapes.geojson``: confer :ref:`shapes`, file containing the gadm shapes

Outputs
-------

- ``resources/demand_profiles.csv``: the content of the file is the electric demand profile associated to each bus. The file has the snapshots as rows and the buses of the network as columns.

Description
-----------

The rule :mod:`build_demand_profiles` creates electricity demand profiles for
network buses. Demand is read from GEGIS or DemandCast for the selected
countries and the reference country. Missing country profiles are synthesized
from the reference profile using country GDP and population totals. The load
scale is then applied, and country demand is aligned to network snapshots and
allocated to bus regions using GDP and population weights.
"""

import os
import os.path
from itertools import product

import country_converter as coco
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import scipy.sparse as sparse
import xarray as xr
from _helpers import (
    BASE_DIR,
    add_transform_iso3,
    configure_logging,
    create_logger,
    read_csv_nafix,
    read_osm_config,
)
from shapely.prepared import prep
from shapely.validation import make_valid

logger = create_logger(__name__)


# Default weights for country demand filling and allocation to bus regions.
WEIGHTS = {"gdp": 0.6, "pop": 0.4}

def normed(s):
    if s.sum() == 0:
        return s
    return s / s.sum()


def get_gegis_regions(countries):
    """
    Find the configured GEGIS regions covering the requested countries.

    Parameters
    ----------
    countries : list[str]
        Country codes for which demand data is required.

    Returns
    -------
    list[str]
        Matching GEGIS region names, without duplicates.
    """
    gegis_dict, world_iso = read_osm_config("gegis_regions", "world_iso")

    regions = []

    for d_region in [gegis_dict, world_iso]:
        for key, value in d_region.items():
            # ignore if the key is already in the regions list
            if key not in regions:
                # if a country is in the regions values, then load it
                cintersect = set(countries).intersection(set(value.keys()))
                if cintersect:
                    regions.append(key)
    return regions


def get_load_countries(countries, fill_missing=None):
    """Include the reference country without changing the modeled countries."""
    return list(countries) + [fill_missing["base_country"]]


def get_load_paths_gegis(ssp_parentfolder, config):
    """
    Create load paths for GEGIS outputs.

    Paths cover the selected countries and the reference country, using the
    configured weather year and prediction year with the legacy SSP2-2.6 scenario.

    Example
    -------
    ["/data/ssp2-2.6/2030/era5_2013/Africa.nc", "/data/ssp2-2.6/2030/era5_2013/Africa.nc"]
    """
    countries = get_load_countries(
        config.get("countries"), config["load_options"]["fill_missing"]
    )
    region_load = get_gegis_regions(countries)
    weather_year = config.get("load_options")["weather_year"]
    prediction_year = config.get("load_options")["prediction_year"]
    # legacy option
    ssp = "ssp2-2.6"

    scenario_path = os.path.join(ssp_parentfolder, ssp)

    load_paths = []
    load_dir = os.path.join(
        ssp_parentfolder,
        str(ssp),
        str(prediction_year),
        "era5_" + str(weather_year),
    )

    file_names = []
    for continent in region_load:
        sel_ext = ".nc"
        for ext in [".nc", ".csv"]:
            load_path = os.path.join(BASE_DIR, str(load_dir), str(continent) + str(ext))
            if os.path.exists(load_path):
                sel_ext = ext
                break
        file_name = str(continent) + str(sel_ext)
        load_path = os.path.join(str(load_dir), file_name)
        load_paths.append(load_path)
        file_names.append(file_name)

    logger.info(
        f"Demand data folder: {load_dir}, load path is {load_paths}.\n"
        + f"Expected files: "
        + "; ".join(file_names)
    )

    return load_paths


def shapes_to_shapes(orig, dest):
    """
    Adopted from vresutils.transfer.Shapes2Shapes()
    """
    orig = list(orig)
    dest = list(dest)
    orig_prepped = list(map(prep, orig))
    transfer = sparse.lil_matrix((len(dest), len(orig)), dtype=float)

    for i, j in product(range(len(dest)), range(len(orig))):
        if orig_prepped[j].intersects(dest[i]):
            area = orig[j].intersection(dest[i]).area
            transfer[i, j] = area / dest[i].area

    return transfer


def load_demand_csv(path):
    df = read_csv_nafix(path, sep=";")
    df.time = pd.to_datetime(df.time, format="%Y-%m-%d %H:%M:%S")
    load_regions = {c: n for c, n in zip(df.region_code, df.region_name)}

    gegis_load = df.set_index(["region_code", "time"]).to_xarray()
    gegis_load = gegis_load.assign_coords(
        {
            "region_name": (
                "region_code",
                [name for (code, name) in load_regions.items()],
            )
        }
    )
    return gegis_load


def compose_gegis_load(
    load_paths: str | list[str],
    countries: str | list[str],
) -> pd.DataFrame:
    """
    Read and merge GEGIS electricity demand data from multiple input files.

    Parameters
    ----------
    load_paths : str or list[str]
        Paths to demand input files.
    countries : str or list[str]
        Region codes used to look for the demand data.

    Returns
    -------
    gegis_load : pd.DataFrame
        Electricity load with ``time`` index, and containing the columns
        ``region_code``, ``region_name``, and ``Electricity demand``.
    """

    gegis_load_list = []

    for path in load_paths:
        if str(path).endswith(".csv"):
            gegis_load_xr = load_demand_csv(path)
        else:
            # Merge load .nc files: https://stackoverflow.com/questions/47226429/join-merge-multiple-netcdf-files-using-xarray
            gegis_load_xr = xr.open_mfdataset(path, combine="nested")
        gegis_load_list.append(gegis_load_xr)

    logger.info(f"Merging demand data from paths {load_paths} into the load data frame")
    gegis_load = xr.merge(gegis_load_list)
    gegis_load = gegis_load.to_dataframe().reset_index().set_index("time")

    gegis_load = gegis_load.loc[gegis_load.region_code.isin(countries)]

    return gegis_load


def read_demcast_load(
    load_paths: str, weather_year: int, countries: str | list[str]
) -> pd.DataFrame:
    """
    Load electricity demand data from DemandCast dataset
    for selected countries and a given weather year.

    Parameters
    ----------
    load_paths : str
        Path to the parquet file with Demcast demand data.
    weather_year : int
        Weather year for which demand profile should be extracted.
    countries : str or list
        Country name or list of country names to subset the demand dataset.

    Returns
    -------
    demcast_load : pd.DataFrame
        Electricity load with ``time`` index, and containing the columns
        ``region_code``, ``region_name``, and ``Electricity demand``.

    References
    ----------
    Kevin Steijn, Vamsi Priya Goli, Enrico Antonini (2025)
    "DemandCast: Global hourly electricity demand forecasting"
    https://arxiv.org/abs/2510.08000
    """
    countries_iso3 = coco.convert(names=countries, to="ISO3")
    if isinstance(countries_iso3, str):
        countries_iso3 = [countries_iso3]

    # Filter by country while reading to avoid loading the whole dataset
    demcast_full_load = pd.read_parquet(
        load_paths,
        columns=["Time (UTC)", "Entity code", "Forecast load (MW)"],
        filters=[("Entity code", "in", countries_iso3)],
    )

    demcast_full_load["time"] = pd.to_datetime(demcast_full_load["Time (UTC)"])
    demcast_load = demcast_full_load[demcast_full_load["time"].dt.year == weather_year]
    demcast_load = demcast_load.rename(
        columns={"Forecast load (MW)": "Electricity demand"}
    )

    demcast_load = add_transform_iso3(
        demcast_load,
        source="Entity code",
        target="ISO2",
        output="region_code",
    )
    demcast_load = add_transform_iso3(
        demcast_load,
        source="Entity code",
        target="name_short",
        output="region_name",
    )

    demcast_load = demcast_load[
        ["region_code", "time", "region_name", "Electricity demand"]
    ]

    demcast_load = demcast_load.set_index("time")

    return demcast_load


def align_demand_profile(demand, snapshots, country, max_gap=48):
    """Align country demand, collapsing identical duplicates and logging repairs.

    Fill gaps of at most ``max_gap`` snapshots by time interpolation internally
    and nearest values at endpoints; reject conflicts and larger gaps.
    """
    snapshots = pd.DatetimeIndex(snapshots)
    if (
        snapshots.empty
        or snapshots.hasnans
        or not snapshots.is_unique
        or not snapshots.is_monotonic_increasing
    ):
        raise ValueError("Target demand snapshots must be nonempty, unique and sorted.")
    if not isinstance(demand.index, pd.DatetimeIndex) or demand.index.hasnans:
        raise ValueError(f"{country}: demand must have valid datetime timestamps.")
    if demand.index.tz != snapshots.tz:
        raise ValueError(f"{country}: demand and snapshot time zones do not match.")
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1.")

    demand = demand.loc[demand.index.isin(snapshots)]
    duplicates = int(demand.index.duplicated().sum())
    logger.info(
        "%s demand audit: %d missing timestamps, %d duplicate records, %d null values.",
        country,
        len(snapshots.difference(demand.index)),
        duplicates,
        int(demand.isna().sum()),
    )
    if duplicates:
        counts = demand.groupby(level=0).nunique(dropna=False)
        conflicts = counts.index[counts > 1]
        if len(conflicts):
            raise ValueError(
                f"{country}: conflicting demand values at duplicate timestamps "
                f"{conflicts[:5].tolist()}. Resolve the source timestamps first."
            )
    demand = demand.loc[~demand.index.duplicated()].reindex(snapshots)
    missing = demand.isna()
    if missing.all():
        raise ValueError(f"{country}: no valid demand values at target snapshots.")
    longest_gap = missing.groupby(missing.ne(missing.shift()).cumsum()).sum().max()
    if longest_gap > max_gap:
        raise ValueError(
            f"{country}: demand gap spans {longest_gap} consecutive snapshots "
            f"(maximum allowed: {max_gap}). Check source coverage."
        )

    interpolated = demand.interpolate(method="time", limit_area="inside")
    result = interpolated.ffill().bfill()
    if not np.isfinite(result).all():
        raise ValueError(f"{country}: aligned demand contains invalid values.")
    if duplicates or missing.any():
        logger.warning(
            "%s demand repair: %d duplicate records removed, %d values interpolated, "
            "%d endpoint values filled using nearest demand.",
            country,
            duplicates,
            int((missing & interpolated.notna()).sum()),
            int(interpolated.isna().sum()),
        )
    return result


def fill_missing_demand(
    el_load, countries, shapes, missing_opts, snapshots, *, weights=WEIGHTS
):
    """
    Fill countries with no non-null demand at the target snapshots.

    Reference timestamps and missing values are preserved for alignment during
    bus allocation. Profiles with a non-null value at a target snapshot are kept.

    Parameters
    ----------
    el_load : pd.DataFrame
        Electricity load with ``time`` index, and containing the columns
        ``region_code``, ``region_name``, and ``Electricity demand``.
    countries : list
        List of countries that is config input.
    shapes : gpd.GeoDataFrame
        Contains subregional gdp, population and shape data.
    missing_opts : dict
        Reference base_country and its positive gdp/pop totals, in the same
        units as the administrative shape data.
    snapshots : pd.DatetimeIndex
        Target network snapshots used to check country demand availability.
    weights : dict, optional
        Relative weights for scaling demand between countries. Expected keys
        are ``"gdp"`` and ``"pop"``, with non-negative values summing to 1.

    Returns
    -------
    pd.DataFrame
        Input demand plus synthesized country profiles, before load scaling
        and timestamp alignment.
    """
    available = set(
        el_load.loc[el_load.index.isin(snapshots)]
        .dropna(subset=["Electricity demand"])
        .region_code
    )
    missing = [country for country in countries if country not in available]
    if not missing:
        return el_load

    reference = missing_opts["base_country"]
    if any(
        not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0
        for value in [missing_opts["gdp"], missing_opts["pop"]]
    ):
        raise ValueError("fill_missing.gdp and pop must be positive reference totals.")
    if reference not in available:
        raise ValueError(f"{reference}: no reference demand at target snapshots.")
    
    base = el_load.loc[el_load.region_code == reference, "Electricity demand"]
    filled = []
    for country in missing:
        # Sum administrative GDP and population totals for the missing country.
        country_shapes = shapes.loc[shapes.country == country]
        missing_counts = country_shapes[["gdp", "pop"]].isna().sum()
        if missing_counts.any():
            logger.warning(
                "%s: skipping %d missing GDP and %d missing population values "
                "when summing country totals; demand may be underestimated.",
                country,
                missing_counts["gdp"],
                missing_counts["pop"],
            )
        target_totals = country_shapes[["gdp", "pop"]].sum(axis=0)
        if not np.isfinite(target_totals.values).all() or (target_totals <= 0).any():
            raise ValueError(f"{country}: demand filling needs positive GDP/population.")
        # Weight the target-to-reference ratios using the supplied weights.
        factor = sum(
            target_totals[key] / missing_opts[key] * weight
            for key, weight in weights.items()
        )
        filled.append(
            pd.DataFrame(
                {
                    "region_code": country,
                    "region_name": country,
                    "Electricity demand": base * factor,
                },
                index=base.index,
            )
        )
        logger.warning(
            "Filling %s demand from %s with GDP/population multiplier %.6g.",
            country,
            reference,
            factor,
        )
    return pd.concat([el_load.loc[~el_load.region_code.isin(missing)], *filled])


def build_demand_profiles(
    n,
    load_source,
    load_paths,
    regions,
    admin_shapes,
    countries,
    scale,
    weather_year,
    out_path,
    fill_missing,
    weights=WEIGHTS,
):
    """
    Create csv file of electric demand time series.

    Parameters
    ----------
    n : pypsa network
        Provides the buses and target demand snapshots.
    load_source : str
        Type of data source to be used for electricity demand
    load_paths: paths of the load files
    regions : .geojson
        Contains bus_id of low voltage substations and
        bus region shapes (voronoi cells)
    admin_shapes : .geojson
        contains subregional gdp, population and shape data
    countries : list
        List of countries that is config input
    scale : float or dict
        Demand multiplier, or country-specific multipliers with an optional
        DEFAULT value (1.3 = 30% more load).
    weather_year : int
        Weather year selected from the DemandCast dataset.
    out_path : str
        Destination CSV path.
    fill_missing : dict
        Reference base_country and its positive gdp/pop totals, in the same
        units as the administrative shape data.

    weights : dict, optional
        GDP and population weights used for allocation to bus regions.

    Returns
    -------
    None
        Writes the bus demand time series to out_path.
    """
    substation_lv_i = n.buses.index[n.buses["substation_lv"]]
    regions = gpd.read_file(regions).set_index("name").reindex(substation_lv_i)

    load_countries = get_load_countries(countries, fill_missing)
    if (load_source == "gegis") or (load_source == "ssp"):
        gegis_load = compose_gegis_load(load_paths=load_paths, countries=load_countries)
        el_load = gegis_load
    else:
        if not os.path.exists(load_paths):
            logger.error(
                f"DemandCast dataset is not available via {load_paths}."
                " Please download it from the source. "
                "The specification available under `demandcast_full`"
                "in configs/bundle_config.yaml"
            )
            raise Exception(f"Stopping execution since no demand data available")

        demcast_load = read_demcast_load(
            load_paths=load_paths,
            weather_year=weather_year,
            countries=load_countries,
        )
        el_load = demcast_load

    shapes = gpd.read_file(admin_shapes).set_index("GADM_ID")
    shapes["geometry"] = shapes["geometry"].apply(lambda x: make_valid(x))

    el_load = fill_missing_demand(
        el_load, countries, shapes, fill_missing, n.snapshots, weights=weights
    )

    # filter load for analysed countries
    el_load = el_load.loc[el_load.region_code.isin(countries)].copy()

    if isinstance(scale, dict):
        logger.info(f"Using custom scaling factor for load data.")
        DEFAULT_VAL = scale.get("DEFAULT", 1.0)
        for country in countries:
            scale.setdefault(country, DEFAULT_VAL)

        for country, scale_country in scale.items():
            el_load.loc[
                el_load.region_code == country, "Electricity demand"
            ] *= scale_country

    elif isinstance(scale, (int, float)):
        logger.info(f"Load data scaled with scaling factor {scale}.")
        el_load["Electricity demand"] *= scale

    def upsample(cntry, group, *, weights=weights):
        """
        Align country demand to snapshots and allocate it to bus regions.
        """
        l = el_load.loc[el_load.region_code == cntry]["Electricity demand"]
        l = align_demand_profile(l, n.snapshots, cntry)
        if len(group) == 1:
            return pd.DataFrame({group.index[0]: l})
        else:
            shapes_cntry = shapes.loc[shapes.country == cntry]
            transfer = shapes_to_shapes(group, shapes_cntry.geometry).T.tocsr()
            gdp_n = pd.Series(
                transfer.dot(shapes_cntry["gdp"].fillna(1.0).values), index=group.index
            )
            pop_n = pd.Series(
                transfer.dot(shapes_cntry["pop"].fillna(1.0).values), index=group.index
            )

            # Default weights (0.6 GDP, 0.4 population) come from a regression
            # on European country load data; see vresutils.load._upsampling_weights.
            # TODO: assess whether these defaults need adjustment for Africa.
            factors = normed(weights["gdp"] * normed(gdp_n) + weights["pop"] * normed(pop_n))
            if factors.sum() == 0:
                logger.warning(
                    f"Upsampling factors for {cntry} are all zero, returning uniform distribution across {len(factors)} shapes."
                )
                factors = pd.Series(
                    np.ones(len(factors)) / len(factors), index=factors.index
                )
            return pd.DataFrame(
                factors.values * l.values[:, np.newaxis],
                index=l.index,
                columns=factors.index,
            )

    demand_profiles = pd.concat(
        [
            upsample(cntry, group)
            for cntry, group in regions.geometry.groupby(regions.country)
        ],
        axis=1,
    )

    demand_profiles = demand_profiles.reindex(n.snapshots)
    if not np.isfinite(demand_profiles.to_numpy()).all():
        raise ValueError("Distributed demand profiles contain invalid values.")
    demand_profiles.to_csv(out_path, header=True)

    logger.info(f"Demand_profiles csv file created for the corresponding snapshots.")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_demand_profiles")

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)

    # Snakemake imports:
    regions = snakemake.input.regions
    load_paths = snakemake.input["load"]
    countries = snakemake.params.countries
    admin_shapes = snakemake.input.gadm_shapes
    scale = snakemake.params.load_options.get("scale", 1.0)
    out_path = snakemake.output[0]

    load_source = snakemake.params.load_options.get("source", "gegis")
    weather_year = snakemake.params.load_options["weather_year"]
    fill_missing = snakemake.params.load_options["fill_missing"]

    if load_source == "ssp":
        logger.warning(
            f"Configuration option 'load_options::ssp' is deprecated. Use 'load_options::source' instead"
        )

    build_demand_profiles(
        n,
        load_source,
        load_paths,
        regions,
        admin_shapes,
        countries,
        scale,
        weather_year,
        out_path,
        fill_missing=fill_missing,
    )
