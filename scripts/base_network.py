# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Creates the network topology from OpenStreetMap and the corresponding bus regions.

Relevant Settings
-----------------

.. code:: yaml

    snapshots:

    countries:

    crs:

    cluster_options:

    electricity:
        voltages:

    lines:
        types:
        s_max_pu:
        under_construction:

    links:
        p_max_pu:
        p_nom_max:
        under_construction:

    transformers:
        x:
        s_nom:
        type:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`snapshots_cf`, :ref:`toplevel_cf`, :ref:`electricity_cf`, :ref:`load_options_cf`,
    :ref:`lines_cf`, :ref:`links_cf`, :ref:`transformers_cf`

Inputs
------

- ``resources/base_network/all_buses_build_network.csv``: OSM buses
- ``resources/base_network/all_lines_build_network.csv``: OSM HVAC and HVDC lines
- ``resources/base_network/all_converters_build_network.csv``: OSM converters
- ``resources/base_network/all_transformers_build_network.csv``: OSM transformers
- ``resources/shapes/country_shapes.geojson``: confer :ref:`shapes`
- ``resources/shapes/offshore_shapes.geojson``: confer :ref:`shapes`
- ``resources/shapes/gadm_shapes.geojson``: administrative shapes for alternative clustering

Outputs
-------

- ``networks/base.nc``:

    .. image:: /img/base.png
        :width: 33 %

- ``resources/bus_regions/regions_onshore.geojson``:

    .. image:: /img/regions_onshore.png
        :width: 33 %

- ``resources/bus_regions/regions_offshore.geojson``:

    .. image:: /img/regions_offshore.png
        :width: 33 %

Description
-----------
"""
import os
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pypsa
import scipy as sp
import shapely.prepared
import shapely.wkt
from _helpers import (
    REGION_COLS,
    configure_logging,
    create_logger,
    nearest_shape,
    read_csv_nafix,
)
from shapely.geometry import Polygon
from shapely.ops import unary_union

logger = create_logger(__name__)


def _get_oid(df):
    if "tags" in df.columns:
        return df.tags.str.extract('"oid"=>"(\\d+)"', expand=False)
    else:
        return pd.Series(np.nan, df.index)


def get_country(df):
    if "tags" in df.columns:
        return df.tags.str.extract('"country"=>"([A-Z]{2})"', expand=False)
    else:
        return pd.Series(np.nan, df.index)


def _find_closest_links(links, new_links, distance_upper_bound=1.5):
    treecoords = np.asarray(
        [np.asarray(shapely.wkt.loads(s))[[0, -1]].flatten() for s in links.geometry]
    )
    querycoords = np.vstack(
        [new_links[["x1", "y1", "x2", "y2"]], new_links[["x2", "y2", "x1", "y1"]]]
    )
    tree = sp.spatial.KDTree(treecoords)
    dist, ind = tree.query(querycoords, distance_upper_bound=distance_upper_bound)
    found_b = ind < len(links)
    found_i = np.arange(len(new_links) * 2)[found_b] % len(new_links)

    return (
        pd.DataFrame(
            dict(D=dist[found_b], i=links.index[ind[found_b] % len(links)]),
            index=new_links.index[found_i],
        )
        .sort_values(by="D")[lambda ds: ~ds.index.duplicated(keep="first")]
        .sort_index()["i"]
    )


def _load_buses_from_osm(fp_buses):
    buses = (
        read_csv_nafix(fp_buses, dtype=dict(bus_id="str", voltage="float"))
        .set_index("bus_id")
        .drop(["station_id"], axis=1)
        .rename(columns=dict(voltage="v_nom"))
    )

    buses = buses.loc[:, ~buses.columns.str.contains("^Unnamed")]
    buses["v_nom"] /= 1e3
    buses["carrier"] = buses.pop("dc").map({True: "DC", False: "AC"})
    buses["under_construction"] = buses["under_construction"].fillna(False).astype(bool)
    buses["x"] = buses["lon"]
    buses["y"] = buses["lat"]
    # TODO: Drop NAN maybe somewhere else?
    buses = buses.dropna(axis="index", subset=["x", "y", "country"])

    return buses


def add_underwater_links(n, fp_offshore_shapes):
    if not hasattr(n.links, "geometry"):
        n.links["underwater_fraction"] = 0.0
    else:
        offshore_shape = gpd.read_file(fp_offshore_shapes).unary_union
        if offshore_shape is None or offshore_shape.is_empty:
            n.links["underwater_fraction"] = 0.0
        else:
            links = gpd.GeoSeries(n.links.geometry.dropna().map(shapely.wkt.loads))
            n.links["underwater_fraction"] = (
                links.intersection(offshore_shape).length / links.length
            )


def _set_dc_underwater_fraction(lines_or_links, fp_offshore_shapes):
    # HVDC part always has some links as converters
    # excluding probably purely DC networks which are currently somewhat exotic
    if lines_or_links.empty:
        return

    if lines_or_links.loc[lines_or_links.carrier == "DC"].empty:
        # Add "underwater_fraction" both to lines and links
        lines_or_links["underwater_fraction"] = 0.0
        return

    if not hasattr(lines_or_links, "geometry"):
        lines_or_links["underwater_fraction"] = 0.0
    else:
        offshore_shape = gpd.read_file(fp_offshore_shapes).unary_union
        if offshore_shape is None or offshore_shape.is_empty:
            lines_or_links["underwater_fraction"] = 0.0
        else:
            branches = gpd.GeoSeries(
                lines_or_links.geometry.dropna().map(shapely.wkt.loads)
            )
            # fix to avoid NaN for links during augmentation
            if branches.empty:
                lines_or_links["underwater_fraction"] = 0
            else:
                lines_or_links["underwater_fraction"] = (
                    # TODO Check assumption that all underwater lines are DC
                    branches.intersection(offshore_shape).length
                    / branches.length
                )


def _load_lines_from_osm(fp_osm_lines):
    lines = (
        read_csv_nafix(
            fp_osm_lines,
            dtype=dict(
                line_id="str",
                bus0="str",
                bus1="str",
                underground="bool",
                under_construction="bool",
                voltage="float",
                circuits="float",
            ),
        )
        .set_index("line_id")
        .rename(columns=dict(voltage="v_nom", circuits="num_parallel"))
    )

    lines["length"] /= 1e3  # m to km conversion
    lines["v_nom"] /= 1e3  # V to kV conversion
    lines = lines.loc[:, ~lines.columns.str.contains("^Unnamed")]  # remove unnamed col
    # lines = _remove_dangling_branches(lines, buses)  # TODO: add dangling branch removal?

    return lines


# TODO Seems to be not needed anymore
def _load_links_from_osm(fp_osm_converters, base_network_config, voltages_config):
    # the links file can be empty
    if os.path.getsize(fp_osm_converters) == 0:
        links = pd.DataFrame()
        return links

    links = (
        read_csv_nafix(
            fp_osm_converters,
            dtype=dict(
                line_id="str",
                bus0="str",
                bus1="str",
                underground="bool",
                under_construction="bool",
            ),
        )
        .set_index("line_id")
        .rename(columns=dict(voltage="v_nom", circuits="num_parallel"))
    )

    links["length"] /= 1e3  # m to km conversion
    links["v_nom"] /= 1e3  # V to kV conversion
    links = links.loc[:, ~links.columns.str.contains("^Unnamed")]  # remove unnamed col
    # links = _remove_dangling_branches(links, buses)  # TODO: add dangling branch removal?

    return links


def _load_converters_from_osm(fp_osm_converters, buses):
    # the links file can be empty
    if os.path.getsize(fp_osm_converters) == 0:
        converters = pd.DataFrame()
        return converters

    converters = read_csv_nafix(
        fp_osm_converters,
        dtype=dict(converter_id="str", bus0="str", bus1="str"),
    ).set_index("converter_id")

    # converters = _remove_dangling_branches(converters, buses)

    converters["carrier"] = "B2B"
    converters["dc"] = True

    return converters


def _load_transformers_from_osm(fp_osm_transformers, buses):
    transformers = (
        read_csv_nafix(
            fp_osm_transformers,
            dtype=dict(transformer_id="str", bus0="str", bus1="str"),
        )
        .rename(columns=dict(line_id="transformer_id"))
        .set_index("transformer_id")
    )
    # transformers = _remove_dangling_branches(transformers, buses)  # TODO: add dangling branch removal?

    return transformers


def _get_linetypes_config(line_types, voltages):
    """
    Return the dictionary of linetypes for selected voltages. The dictionary is
    a subset of the dictionary line_types, whose keys match the selected
    voltages.

    Parameters
    ----------
    line_types : dict
        Dictionary of linetypes: keys are nominal voltages and values are linetypes.
    voltages : list
        List of selected voltages.

    Returns
    -------
        Dictionary of linetypes for selected voltages.
    """
    # get voltages value that are not available in the line types
    vnoms_diff = set(voltages).symmetric_difference(set(line_types.keys()))
    if vnoms_diff:
        logger.warning(
            f"Voltages {vnoms_diff} not in the {line_types} or {voltages} list."
        )
    return {k: v for k, v in line_types.items() if k in voltages}


def _get_linetype_by_voltage(v_nom, d_linetypes):
    """
    Return the linetype of a specific line based on its voltage v_nom.

    Parameters
    ----------
    v_nom : float
        The voltage of the line.
    d_linetypes : dict
        Dictionary of linetypes: keys are nominal voltages and values are linetypes.

    Returns
    -------
        The linetype of the line whose nominal voltage is closest to the line voltage.
    """
    v_nom_min, line_type_min = min(
        d_linetypes.items(),
        key=lambda x: abs(x[0] - v_nom),
    )
    return line_type_min


def _set_electrical_parameters_lines(lines_config, voltages, lines):
    if lines.empty:
        lines["type"] = []
        return lines

    linetypes = _get_linetypes_config(lines_config["ac_types"], voltages)

    lines["carrier"] = "AC"
    lines["dc"] = False

    lines.loc[:, "type"] = lines.v_nom.apply(
        lambda x: _get_linetype_by_voltage(x, linetypes)
    )

    lines["s_max_pu"] = lines_config["s_max_pu"]

    return lines


def _set_electrical_parameters_dc_lines(lines_config, voltages, lines):
    if lines.empty:
        lines["type"] = []
        return lines

    linetypes = _get_linetypes_config(lines_config["dc_types"], voltages)

    lines["carrier"] = "DC"
    lines["dc"] = True
    lines.loc[:, "type"] = lines.v_nom.apply(
        lambda x: _get_linetype_by_voltage(x, linetypes)
    )

    lines["s_max_pu"] = lines_config["s_max_pu"]

    return lines


def _set_electrical_parameters_links(links_config, links):
    if links.empty:
        return links

    p_max_pu = links_config.get("p_max_pu", 1.0)
    links["p_max_pu"] = p_max_pu
    links["p_min_pu"] = -p_max_pu

    links["carrier"] = "DC"
    links["dc"] = True

    return links


def _set_electrical_parameters_transformers(transformers_config, transformers):
    config = transformers_config

    # Add transformer parameters
    transformers["x"] = config.get("x", 0.1)
    transformers["s_nom"] = config.get("s_nom", 2000)
    transformers["type"] = config.get("type", "")

    return transformers


def _set_electrical_parameters_converters(links_config, converters):
    p_max_pu = links_config.get("p_max_pu", 1.0)
    converters["p_max_pu"] = p_max_pu
    converters["p_min_pu"] = -p_max_pu

    converters["p_nom"] = 2000  # [MW]?

    # Converters are combined with links
    converters["under_construction"] = False
    converters["underground"] = False

    return converters


def _set_lines_s_nom_from_linetypes(n):
    # Info: n.line_types is a lineregister from pypsa/pandapowers
    n.lines["s_nom"] = (
        np.sqrt(3)
        * n.lines["type"].map(n.line_types.i_nom)
        * n.lines.eval("v_nom * num_parallel")
    )
    # Re-define s_nom for DC lines
    n.lines.loc[n.lines["carrier"] == "DC", "s_nom"] = n.lines["type"].map(
        n.line_types.i_nom
    ) * n.lines.eval("v_nom * num_parallel")


def _remove_dangling_branches(branches, buses):
    return pd.DataFrame(
        branches.loc[branches.bus0.isin(buses.index) & branches.bus1.isin(buses.index)]
    )


def _set_countries_and_substations(inputs, base_network_config, countries_config, n):
    countries = countries_config
    country_shapes = gpd.read_file(inputs.country_shapes).set_index("name")["geometry"]

    offshore_shapes = unary_union(gpd.read_file(inputs.offshore_shapes)["geometry"])

    buses = n.buses
    bus_locations = buses
    bus_locations = gpd.GeoDataFrame(
        bus_locations,
        geometry=gpd.points_from_xy(bus_locations.x, bus_locations.y),
        crs=country_shapes.crs,  # the workflow sets the the same crs for buses and shapes
    )
    # Check if bus is in shape
    offshore_b = bus_locations.within(offshore_shapes)

    # Assumption that HV-bus qualifies as potential offshore bus. Offshore bus is empty otherwise.
    offshore_hvb = (
        buses["v_nom"] >= base_network_config["min_voltage_substation_offshore"] / 1000
    )
    # Compares two lists & makes list value true if at least one is true
    buses["substation_off"] = offshore_b | offshore_hvb

    # Buses without country tag are removed OR get a country tag if close to country
    c_nan_b = buses.country.isnull()
    if c_nan_b.sum() > 0:
        c_tag = get_country(buses.loc[c_nan_b])
        c_tag.loc[~c_tag.isin(countries)] = np.nan
        n.buses.loc[c_nan_b, "country"] = c_tag

        c_tag_nan_b = n.buses.country.isnull()

        # Nearest country in path length defines country of still homeless buses
        # Work-around until commit 705119 lands in pypsa release
        # pypsa-earth comment: Important to connect 'homeless' offshore assets
        # Otherwise
        n.transformers["length"] = 0.0
        graph = n.graph(weight="length")
        n.transformers.drop("length", axis=1, inplace=True)

        for b in n.buses.index[c_tag_nan_b]:
            df = (
                pd.DataFrame(
                    dict(
                        pathlength=nx.single_source_dijkstra_path_length(
                            graph, b, cutoff=200
                        )
                    )
                )
                .join(n.buses.country)
                .dropna()
            )
            assert (
                not df.empty
            ), "No buses with defined country within 200km of bus `{}`".format(b)
            n.buses.at[b, "country"] = df.loc[df.pathlength.idxmin(), "country"]

        logger.warning(
            "{} buses are not in any country or offshore shape,"
            " {} have been assigned from the tag of the entsoe map,"
            " the rest from the next bus in terms of pathlength.".format(
                c_nan_b.sum(), c_nan_b.sum() - c_tag_nan_b.sum()
            )
        )

    return buses


def voronoi(
    points: pd.DataFrame,
    outline: Polygon,
    geo_crs: str = "EPSG:4326",
) -> gpd.GeoSeries:
    """
    Create Voronoi polygons from a set of points within an outline.

    Parameters
    ----------
    points : pd.DataFrame
         DataFrame containing the coordinates of the points with columns ["x", "y"] and index
    outline : Polygon
        Shapely Polygon defining the outline within which to compute the Voronoi partition.
    geo_crs : str
        CRS used for geographic projection, passed to GeoPandas (e.g. "EPSG:4326")

    Returns
    -------
    gpd.GeoSeries
        GeoSeries of Voronoi polygons corresponding to each point in `points`,
        clipped to the `outline` polygon.
    """

    pts = gpd.GeoSeries(
        gpd.points_from_xy(points.x, points.y),
        index=points.index,
        crs=geo_crs,
    )
    voronoi = pts.voronoi_polygons(extend_to=outline).clip(outline)

    # can be removed with shapely 2.1 where order is preserved
    # https://github.com/shapely/shapely/issues/2020
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        pts = gpd.GeoDataFrame(geometry=pts)
        voronoi = gpd.GeoDataFrame(geometry=voronoi)
        joined = gpd.sjoin_nearest(pts, voronoi, how="right")

    gdf = joined.dissolve(by=points.index.name).reindex(points.index).squeeze()

    return gdf


def get_gadm_shape(
    onshore_buses: pd.DataFrame,
    gadm_shapes: gpd.GeoDataFrame,
    geo_crs: str = "EPSG:4326",
    metric_crs: str = "EPSG:3857",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the nearest GADM shape for each bus.

    Parameters
    ----------
    onshore_buses: pd.DataFrame
        DataFrame containing the onshore buses with columns ["x", "y"].
    gadm_shapes: gpd.GeoDataFrame
        GeoDataFrame containing the GADM shapes with a geometry column.
    geo_crs : str
        CRS used for geographic projection, passed to GeoPandas (e.g. "EPSG:4326").
    metric_crs : str
        CRS used for distance projection, passed to GeoPandas (e.g. "EPSG:3857").

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the matching geometries and their GADM shape IDs.
    """
    geo_regions = gpd.GeoDataFrame(
        onshore_buses[["x", "y"]],
        geometry=gpd.points_from_xy(onshore_buses["x"], onshore_buses["y"]),
        crs=geo_crs,
    ).to_crs(metric_crs)

    join_geos = gpd.sjoin_nearest(
        geo_regions, gadm_shapes.to_crs(metric_crs), how="left"
    )

    # when duplicates, keep only the first entry
    join_geos = join_geos[~join_geos.index.duplicated()]

    gadm_sel = gadm_shapes.loc[join_geos[gadm_shapes.index.name].values]

    return gadm_sel.geometry.values, gadm_sel.index.values


def _get_optional_input(inputs, key, default=None):
    return (
        inputs.get(key, default)
        if hasattr(inputs, "get")
        else getattr(inputs, key, default)
    )


def build_bus_regions(
    n: pypsa.Network,
    inputs,
    countries: list[str],
    crs_config: dict,
    alternative_clustering: bool,
    config: dict | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Create onshore and offshore bus regions for the base network.

    The regions are produced from the same in-memory base network that is
    exported by this rule. A network copy is used because subregion handling
    temporarily rewrites bus country labels for region generation.
    """
    n = n.copy()
    config = config or {}

    country_shapes_fn = (
        _get_optional_input(inputs, "subregion_shapes") or inputs.country_shapes
    )
    offshore_shapes_fn = (
        _get_optional_input(inputs, "subregion_offshore") or inputs.offshore_shapes
    )

    geo_crs = crs_config["geo_crs"]
    area_crs = crs_config["area_crs"]
    metric_crs = crs_config["distance_crs"]

    country_shapes = gpd.read_file(country_shapes_fn).set_index("name")["geometry"]

    offshore_shapes = gpd.read_file(offshore_shapes_fn)
    offshore_shapes = offshore_shapes.reindex(columns=REGION_COLS).set_index("name")[
        "geometry"
    ]

    subregion_shapes = _get_optional_input(inputs, "subregion_shapes")
    if subregion_shapes:
        crs = {"geo_crs": geo_crs, "distance_crs": metric_crs}
        tolerance = config.get("subregion", {}).get("tolerance", 100)
        n = nearest_shape(n, country_shapes_fn, crs, tolerance=tolerance)

        countries = list(country_shapes.index)

    gadm_shapes = gpd.read_file(inputs.gadm_shapes).set_index("GADM_ID")

    onshore_regions = []
    offshore_regions = []

    for country in countries:
        c_b = n.buses.country == country
        if n.buses.loc[c_b & n.buses.substation_lv, ["x", "y"]].empty:
            logger.warning(f"No low voltage buses found for {country}!")
            continue

        onshore_shape = country_shapes[country]
        onshore_locs = n.buses.loc[c_b & n.buses.substation_lv, ["x", "y"]]
        gadm_country = gadm_shapes[gadm_shapes.country == country]
        if alternative_clustering:
            onshore_geometry, shape_id = get_gadm_shape(
                onshore_locs,
                gadm_country,
                geo_crs,
                metric_crs,
            )
        else:
            onshore_geometry = voronoi(onshore_locs, onshore_shape)
            shape_id = 0  # Not used

        temp_region = gpd.GeoDataFrame(
            {
                "name": onshore_locs.index,
                "x": onshore_locs["x"],
                "y": onshore_locs["y"],
                "geometry": onshore_geometry,
                "country": country,
                "shape_id": shape_id,
            },
            crs=geo_crs,
        )
        temp_region = temp_region[
            temp_region.geometry.is_valid & ~temp_region.geometry.is_empty
        ]
        onshore_regions.append(temp_region)

        if country not in offshore_shapes.index:
            logger.warning(f"No off-shore shapes for {country}")
            continue

        offshore_shape = offshore_shapes[country]

        if n.buses.loc[c_b & n.buses.substation_off, ["x", "y"]].empty:
            logger.warning(f"No off-shore substations found for {country}")
            continue

        offshore_locs = n.buses.loc[c_b & n.buses.substation_off, ["x", "y"]]
        shape_id = 0  # Not used
        offshore_geometry = voronoi(offshore_locs, offshore_shape)
        offshore_regions_c = gpd.GeoDataFrame(
            {
                "name": offshore_locs.index,
                "x": offshore_locs["x"],
                "y": offshore_locs["y"],
                "geometry": offshore_geometry,
                "country": country,
                "shape_id": shape_id,
            },
            crs=country_shapes.crs,
        )
        offshore_regions_c = offshore_regions_c.loc[
            offshore_regions_c.to_crs(area_crs).area > 1e-2
        ]
        offshore_regions_c = offshore_regions_c[
            offshore_regions_c.geometry.is_valid & ~offshore_regions_c.geometry.is_empty
        ]
        offshore_regions.append(offshore_regions_c)

    onshore_regions = gpd.GeoDataFrame(
        pd.concat(onshore_regions, ignore_index=True),
        crs=country_shapes.crs,
    ).dropna(axis="index", subset=["geometry"])

    if alternative_clustering:
        # determine isolated buses
        n.determine_network_topology()
        non_isolated_buses = n.buses.duplicated(subset=["sub_network"], keep=False)
        isolated_buses = n.buses[~non_isolated_buses].index
        non_isolated_regions = onshore_regions[
            ~onshore_regions.name.isin(isolated_buses)
        ]
        isolated_regions = onshore_regions[onshore_regions.name.isin(isolated_buses)]

        # Combine regions while prioritizing non-isolated ones
        onshore_regions = pd.concat(
            [non_isolated_regions, isolated_regions]
        ).drop_duplicates("shape_id", keep="first")

        gadm_region_count = len(gadm_shapes[gadm_shapes.country.isin(countries)])
        if len(onshore_regions) < gadm_region_count:
            logger.warning(
                "The number of remaining buses is less than the number of "
                "administrative clusters suggested!"
            )

    if subregion_shapes:
        logger.info("Deactivate subregion classification")
        original_shapes = inputs.original_shapes
        n = nearest_shape(n, original_shapes, crs, tolerance=tolerance)

        onshore_regions["country"] = onshore_regions.name.map(n.buses.country)
        for offshore_region in offshore_regions:
            offshore_region["country"] = offshore_region.name.map(n.buses.country)

    if offshore_regions:
        offshore_regions = gpd.GeoDataFrame(
            pd.concat(offshore_regions, ignore_index=True),
            crs=country_shapes.crs,
        )
    else:
        offshore_regions = offshore_shapes.to_frame()

    return onshore_regions, offshore_regions


def base_network(
    inputs,
    base_network_config,
    countries_config,
    hvdc_as_lines_config,
    lines_config,
    links_config,
    snapshots_config,
    transformers_config,
    voltages_config,
):
    buses = _load_buses_from_osm(inputs.osm_buses).reset_index(drop=True)
    lines = _load_lines_from_osm(inputs.osm_lines).reset_index(drop=True)
    transformers = _load_transformers_from_osm(inputs.osm_transformers, buses)
    converters = _load_converters_from_osm(inputs.osm_converters, buses)

    lines_ac = lines[~lines.dc].copy()
    lines_dc = lines[lines.dc].copy()
    lines_ac = _set_electrical_parameters_lines(lines_config, voltages_config, lines_ac)

    lines_dc = _set_electrical_parameters_dc_lines(
        lines_config, voltages_config, lines_dc
    )

    transformers = _set_electrical_parameters_transformers(
        transformers_config, transformers
    )
    converters = _set_electrical_parameters_converters(links_config, converters)

    n = pypsa.Network()
    n.name = "PyPSA-Earth"

    n.set_snapshots(pd.date_range(freq="h", **snapshots_config))
    n.snapshot_weightings[:] *= 8760.0 / n.snapshot_weightings.sum()

    n.import_components_from_dataframe(buses, "Bus")

    if hvdc_as_lines_config:
        lines = pd.concat([lines_ac, lines_dc])
        n.import_components_from_dataframe(lines, "Line")
    else:
        lines_dc = _set_electrical_parameters_links(links_config, lines_dc)
        # parse line information into p_nom required for converters
        lines_dc["p_nom"] = lines_dc.apply(
            lambda x: x["v_nom"] * n.line_types.i_nom[x["type"]],
            axis=1,
            result_type="reduce",
        )
        n.import_components_from_dataframe(lines_ac, "Line")
        n.import_components_from_dataframe(lines_dc, "Link")

    n.import_components_from_dataframe(transformers, "Transformer")
    n.import_components_from_dataframe(converters, "Link")

    # greenfield capacity expansion is represented with null capacity using num_parallel==0
    n.lines["num_parallel"] = n.lines["num_parallel"].where(
        ~n.lines["under_construction"], 0.0
    )
    n.lines.drop(columns="under_construction", inplace=True, errors="ignore")

    _set_lines_s_nom_from_linetypes(n)

    _set_countries_and_substations(inputs, base_network_config, countries_config, n)

    _set_dc_underwater_fraction(n.lines, inputs.offshore_shapes)
    _set_dc_underwater_fraction(n.links, inputs.offshore_shapes)

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("base_network")

    configure_logging(snakemake)

    inputs = snakemake.input

    # Snakemake imports:
    alternative_clustering = snakemake.params.alternative_clustering
    base_network_config = snakemake.params.base_network
    countries = snakemake.params.countries
    crs = snakemake.params.crs
    hvdc_as_lines = snakemake.params.hvdc_as_lines
    lines = snakemake.params.lines
    links = snakemake.params.links
    snapshots = snakemake.params.snapshots
    transformers = snakemake.params.transformers
    voltages = snakemake.params.voltages

    n = base_network(
        inputs,
        base_network_config,
        countries,
        hvdc_as_lines,
        lines,
        links,
        snapshots,
        transformers,
        voltages,
    )

    onshore_regions, offshore_regions = build_bus_regions(
        n,
        inputs,
        countries,
        crs,
        alternative_clustering,
        snakemake.config,
    )

    n.buses = pd.DataFrame(n.buses.drop(columns="geometry", errors="ignore"))
    n.meta = snakemake.config
    n.export_to_netcdf(snakemake.output.network)
    onshore_regions.to_file(snakemake.output.regions_onshore)
    offshore_regions.to_file(snakemake.output.regions_offshore)
