# shape.py

import geopandas as gpd
from shapely import affinity, geometry


def similar_geog_different_centroid(reference_geog: gpd.GeoDataFrame,
                                    centroid_y: float,
                                    centroid_x: float,
                                   ):
    """
    Creates polygon with similar shape and area using a different centroid.
    Input:
      reference_geog - reference geopandas polygon
      centroid_y - centroid y-location.
      centroid_x - centroid x-location.
    Output:
      new_geog - new geopandas polygon
    """
    # Suppose your new desired center is (new_x, new_y)
    new_center = (centroid_x, centroid_y)

    # Extract the original geometry (assuming it's one polygon)
    old_geom = reference_geog.geometry.iloc[0]

    # Get the centroid of the original geometry
    old_center = old_geom.centroid

    # Compute shift offsets
    dx = new_center[0] - old_center.x
    dy = new_center[1] - old_center.y

    # Translate the geometry to the new center
    new_geom = affinity.translate(old_geom, xoff=dx, yoff=dy)

    # Create a new GeoDataFrame with the same CRS
    new_geog_gdf = gpd.GeoDataFrame(geometry=[new_geom], crs=reference_geog.crs)
    return new_geog_gdf