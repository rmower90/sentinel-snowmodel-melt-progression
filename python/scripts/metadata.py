"""
Utilities for loading project metadata and retrieving configured paths.
"""

from __future__ import annotations

from pathlib import Path
import yaml


FT_TO_M = 0.3048


def _resolve_path_tokens(obj, roots: dict):
    """
    Recursively replace path tokens in strings contained within a YAML object.

    Supported tokens
    ----------------
    {repo_root}
    {data_root}

    Parameters
    ----------
    obj
        YAML-loaded object. May be a dict, list, string, or scalar.

    roots : dict
        Dictionary containing root path definitions.

    Returns
    -------
    object
        Object with path tokens expanded.
    """

    if isinstance(obj, dict):
        return {
            key: _resolve_path_tokens(value, roots)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            _resolve_path_tokens(value, roots)
            for value in obj
        ]

    if isinstance(obj, str):
        return (
            obj
            .replace("{repo_root}", roots["repo_root"])
            .replace("{data_root}", roots["data_root"])
        )

    return obj


def load_yaml(path: Path | str) -> dict:
    """
    Load a project YAML configuration file.

    For domain configuration files, path placeholders such as
    ``{repo_root}`` and ``{data_root}`` are automatically expanded using
    ``configs/paths.yaml``.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration with root path placeholders resolved.
    """

    path = Path(path)

    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    # paths.yaml defines the roots itself and therefore should not
    # recursively attempt to resolve another paths.yaml.
    if path.name == "paths.yaml":
        return cfg

    # Assume paths.yaml lives beside the domain YAML file.
    paths_yaml = path.parent / "paths.yaml"

    if not paths_yaml.exists():
        raise FileNotFoundError(
            f"Could not find shared path configuration: {paths_yaml}"
        )

    with paths_yaml.open("r") as f:
        roots = yaml.safe_load(f)

    required_roots = {"repo_root", "data_root"}
    missing = required_roots - roots.keys()

    if missing:
        raise KeyError(
            f"Missing required entries in {paths_yaml}: {sorted(missing)}"
        )

    return _resolve_path_tokens(cfg, roots)


def get_point_geography(cfg_region: dict, site: str):
    """
    Return geographic and projected coordinates for a point location.
    """

    lat = cfg_region["location"][site]["latitude"]
    lon = cfg_region["location"][site]["longitude"]
    x_proj = cfg_region["location"][site]["x_proj"]
    y_proj = cfg_region["location"][site]["y_proj"]
    crs = cfg_region["crs"]["epsg"]

    return lat, lon, x_proj, y_proj, crs


def get_shapefile_fpath(
    cfg_region: dict,
    key: str,
    value: str,
):
    """
    Return a configured shapefile path and CRS.
    """

    shapefile_fpath = cfg_region[key][value]
    crs = cfg_region["crs"]["epsg"]

    return shapefile_fpath, crs


def get_sentinel_fpath(
    cfg_region: dict,
    resolution: str = "100m",
):
    """
    Return a configured Sentinel-1 directory.
    """

    return cfg_region["sentinel_fpath"][resolution]


def get_snowmodel_fpath(
    cfg_region: dict,
    bias: str = "biased",
):
    """
    Return a configured SnowModel directory.

    The supplied key must exist under ``snowmodel_fpath`` in the domain
    configuration. This allows configurations such as biased, corrected,
    optimized, aso_match_biased, and aso_match_prec without hard-coding
    valid options here.
    """

    try:
        return cfg_region["snowmodel_fpath"][bias]

    except KeyError as exc:
        available = list(cfg_region.get("snowmodel_fpath", {}).keys())

        raise KeyError(
            f"Unknown SnowModel configuration '{bias}'. "
            f"Available options: {available}"
        ) from exc


def get_cuesOBS_fpath(
    cfg_region: dict,
    var: str = "swe",
):
    """
    Return a configured CUES/JDSSS observation filepath.
    """

    try:
        return cfg_region["cues_obs"][var]

    except KeyError as exc:
        available = list(cfg_region.get("cues_obs", {}).keys())

        raise KeyError(
            f"Unknown CUES observation '{var}'. "
            f"Available options: {available}"
        ) from exc


def get_watershed_fpath(
    cfg_region: dict,
    basin_id: str,
):
    """
    Return a configured watershed filepath.
    """

    try:
        return cfg_region["watershed_delineation"][basin_id]

    except KeyError as exc:
        available = list(
            cfg_region.get("watershed_delineation", {}).keys()
        )

        raise KeyError(
            f"Unknown watershed '{basin_id}'. "
            f"Available options: {available}"
        ) from exc