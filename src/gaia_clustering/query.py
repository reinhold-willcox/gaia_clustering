"""Resolve a cluster or star name and pull a Gaia DR3 neighbourhood.

Gaia radial velocities are never used. Independent RVs can be merged
afterwards via ``attach_radial_velocities``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.utils.data import download_file, get_file_contents

SESAME_URL = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxpI/SNV?"
GAIA_TAP_SYNC = "https://gea.esac.esa.int/tap-server/tap/sync"
SIMBAD_TAP_SYNC = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"

DEFAULT_CACHE_DIR = Path.cwd() / "data" / "cache"

# SIMBAD otypes that are groups rather than individual stars.
_GROUP_OTYPES = {
    "As*", "St*", "OpC", "Cl*", "GlC", "MGr", "C?*", "*Ass",
    "Association", "OpenCluster", "Cluster", "MovingGroup",
    "ClG", "GroupG", "SuperCl", "Stream",
}

GAIA_COLUMNS = [
    "designation",
    "source_id",
    "ra",
    "dec",
    "ra_error",
    "dec_error",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "ra_dec_corr",
    "ra_parallax_corr",
    "ra_pmra_corr",
    "ra_pmdec_corr",
    "dec_parallax_corr",
    "dec_pmra_corr",
    "dec_pmdec_corr",
    "parallax_pmra_corr",
    "parallax_pmdec_corr",
    "pmra_pmdec_corr",
    "ruwe",
    "phot_g_mean_mag",
    "visibility_periods_used",
    "astrometric_sigma5d_max",
    "astrometric_gof_al",
    "parallax_over_error",
]


def _cache_path(cache_dir, prefix, payload):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{prefix}_{key}.pkl"


def _read_cache(path):
    if path is None or not Path(path).exists():
        return None
    return pd.read_pickle(path)


def _write_cache(path, obj):
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_pickle(path)
    else:
        Path(path).write_text(json.dumps(obj), encoding="utf-8")


def gaia_tap_query(adql, timeout=300, cache_dir=None):
    """Run a synchronous Gaia TAP ADQL query.

    Parameters
    ----------
    adql : str
        ADQL query string.
    timeout : float
        HTTP timeout in seconds.
    cache_dir : path-like or None
        If set, pickle results keyed by a hash of ``adql``.

    Returns
    -------
    pandas.DataFrame
        Column names lower-cased.
    """
    cache_path = None
    if cache_dir is not None:
        cache_path = _cache_path(cache_dir, "gaia", adql)
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached
    payload = urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "votable",
            "QUERY": adql,
        }
    ).encode("utf-8")
    request = Request(GAIA_TAP_SYNC, data=payload)
    with urlopen(request, timeout=timeout) as response:
        votable_bytes = response.read()
    table = Table.read(BytesIO(votable_bytes), format="votable")
    df = table.to_pandas()
    df.columns = [c.lower() for c in df.columns]
    if cache_path is not None:
        _write_cache(cache_path, df)
    return df


def _sesame_xml(name):
    url = SESAME_URL + quote(name, safe="")
    text = get_file_contents(download_file(url, cache=True, show_progress=False))
    return text.split("<!---")[0]


def _is_group_otype(otype):
    if not otype:
        return False
    o = str(otype)
    if o in _GROUP_OTYPES:
        return True
    lower = o.lower()
    return any(k in lower for k in (
        "assoc", "cluster", "moving group", "stream", "ob association",
    ))


def resolve_name(name):
    """Resolve a catalogue name with CDS Sesame (SIMBAD / NED / VizieR).

    Parameters
    ----------
    name : str
        Cluster, association, or star identifier.

    Returns
    -------
    dict
        ``input_name``, ``simbad_main_id``, ``simbad_ra``, ``simbad_dec``,
        ``otype``, ``kind`` (``"group"`` or ``"star"``), ``gaia_source_id``,
        and ``status`` (``ok``, ``no_gaia_id``, or ``unrecognized``).
    """
    result = {
        "input_name": name,
        "simbad_main_id": None,
        "simbad_ra": np.nan,
        "simbad_dec": np.nan,
        "otype": None,
        "kind": "unknown",
        "gaia_source_id": None,
        "status": "unrecognized",
    }
    try:
        root = ET.fromstring(_sesame_xml(name))
    except ET.ParseError:
        return result

    resolver = None
    for node in root.findall(".//Resolver"):
        info = " ".join((node.findtext("INFO") or "", node.findtext("info") or ""))
        if "Nothing found" in info:
            continue
        if node.find("jradeg") is not None:
            resolver = node
            break
    if resolver is None:
        return result

    result["simbad_main_id"] = resolver.findtext("oname") or name
    result["simbad_ra"] = float(resolver.findtext("jradeg"))
    result["simbad_dec"] = float(resolver.findtext("jdedeg"))
    otype = resolver.findtext("otype") or resolver.findtext("OTYPE")
    result["otype"] = otype
    result["kind"] = "group" if _is_group_otype(otype) else "star"
    # Name-based fallback if Sesame otype is missing or too generic.
    lowered = name.lower()
    if any(tok in lowered for tok in ("ob ", "ob-", "association", "cluster", "moving group")):
        result["kind"] = "group"

    aliases = [a.text.strip() for a in resolver.findall("alias") if a.text]
    gaia_dr3 = next((a for a in aliases if a.startswith("Gaia DR3 ")), None)
    if gaia_dr3 is not None:
        result["gaia_source_id"] = int(gaia_dr3.split()[-1])
        result["status"] = "ok"
    else:
        result["status"] = "no_gaia_id"
    return result


def query_gaia_by_source_ids(source_ids, cache_dir=None):
    """Fetch Gaia DR3 astrometry for a list of ``source_id`` values.

    Radial-velocity columns are not requested.
    """
    ids = sorted({int(i) for i in source_ids if i is not None and not pd.isna(i)})
    if not ids:
        return pd.DataFrame(columns=GAIA_COLUMNS)
    cols = ", ".join(GAIA_COLUMNS)
    id_list = ", ".join(str(i) for i in ids)
    adql = f"SELECT {cols} FROM gaiadr3.gaia_source WHERE source_id IN ({id_list})"
    return gaia_tap_query(adql, cache_dir=cache_dir)


def query_gaia_cone(
    ra_deg,
    dec_deg,
    radius_deg=1.5,
    g_max=13.0,
    ruwe_max=1.4,
    plx_snr_min=5.0,
    visibility_min=8,
    top=10000,
    cache_dir=None,
):
    """Gaia DR3 cone search with quality cuts.

    Gaia radial velocities are not retrieved.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Cone centre (ICRS, degrees).
    radius_deg : float
        Search radius in degrees.
    g_max : float
        Faint *G* magnitude limit.
    ruwe_max : float
        Maximum RUWE.
    plx_snr_min : float
        Minimum ``parallax_over_error``.
    visibility_min : int
        Minimum ``visibility_periods_used``.
    top : int
        Maximum number of rows. A warning is issued if the cap is hit.
    cache_dir : path-like or None
        Optional TAP cache directory.

    Returns
    -------
    pandas.DataFrame
    """
    cols = ", ".join(GAIA_COLUMNS)
    adql = f"""
    SELECT TOP {int(top)} {cols}
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {float(ra_deg)}, {float(dec_deg)}, {float(radius_deg)})
    )
    AND phot_g_mean_mag < {float(g_max)}
    AND parallax IS NOT NULL
    AND parallax > 0
    AND pmra IS NOT NULL
    AND pmdec IS NOT NULL
    AND parallax_error > 0
    AND pmra_error > 0
    AND pmdec_error > 0
    AND ruwe < {float(ruwe_max)}
    AND parallax_over_error > {float(plx_snr_min)}
    AND visibility_periods_used > {int(visibility_min)}
    """
    df = gaia_tap_query(adql, cache_dir=cache_dir)
    if "source_id" in df.columns and len(df):
        df["source_id"] = df["source_id"].astype("int64")
    if len(df) >= int(top):
        warnings.warn(
            "Gaia cone returned TOP={} rows; tighten --g-max/--radius or raise top".format(top),
            RuntimeWarning,
            stacklevel=2,
        )
    return df


def apply_parallax_window(df, pi0, frac_lo=0.4, frac_hi=2.5):
    """Keep stars whose parallax lies in ``[frac_lo, frac_hi] * pi0``.

    Parameters
    ----------
    df : DataFrame
        Must contain ``parallax`` (mas).
    pi0 : float
        Reference parallax (mas). Non-positive values return ``df`` unchanged.
    frac_lo, frac_hi : float
        Multiplicative window edges.

    Returns
    -------
    pandas.DataFrame
    """
    pi0 = float(pi0)
    if not np.isfinite(pi0) or pi0 <= 0:
        return df
    plx = np.asarray(df["parallax"], dtype=float)
    keep = (plx >= frac_lo * pi0) & (plx <= frac_hi * pi0)
    return df.loc[keep].copy()


def fetch_neighbourhood(
    name,
    radius_deg=1.5,
    g_max=13.0,
    ruwe_max=1.4,
    plx_snr_min=5.0,
    parallax_window=(0.4, 2.5),
    cache_dir=DEFAULT_CACHE_DIR,
    top=10000,
):
    """Resolve ``name`` and return a Gaia DR3 neighbourhood.

    Parameters
    ----------
    name : str
        Cluster/association name, or a star in the association.
    radius_deg : float
        Cone radius in degrees.
    g_max : float
        Faint *G* magnitude limit.
    ruwe_max : float
        Maximum RUWE.
    plx_snr_min : float
        Minimum ``parallax_over_error``.
    parallax_window : tuple of float or None
        Multiplicative parallax window around a resolved star's parallax
        (ignored for groups with no Gaia source). ``None`` disables the cut.
    cache_dir : path-like or None
        Directory for pickled TAP results. ``None`` disables caching.
    top : int
        Maximum cone-search rows.

    Returns
    -------
    catalog : pandas.DataFrame
        Gaia astrometry (no radial velocities).
    query_info : dict
        Resolution metadata, search parameters, and row counts.
    """
    resolved = resolve_name(name)
    if resolved["status"] == "unrecognized":
        raise ValueError("SIMBAD/Sesame did not recognize {!r}".format(name))
    if not np.isfinite(resolved["simbad_ra"]):
        raise ValueError("No coordinates for {!r}".format(name))

    seed = None
    if resolved["gaia_source_id"] is not None:
        seed = query_gaia_by_source_ids(
            [resolved["gaia_source_id"]], cache_dir=cache_dir
        )
        if seed is not None and len(seed) == 0:
            seed = None

    catalog = query_gaia_cone(
        resolved["simbad_ra"],
        resolved["simbad_dec"],
        radius_deg=radius_deg,
        g_max=g_max,
        ruwe_max=ruwe_max,
        plx_snr_min=plx_snr_min,
        top=top,
        cache_dir=cache_dir,
    )
    n_cone = len(catalog)

    pi0 = None
    if seed is not None and "parallax" in seed.columns and len(seed):
        pi0 = float(seed["parallax"].iloc[0])
    if pi0 is not None and parallax_window is not None:
        catalog = apply_parallax_window(
            catalog, pi0, frac_lo=parallax_window[0], frac_hi=parallax_window[1]
        )

    if seed is not None and len(seed) and "source_id" in catalog.columns:
        missing = ~seed["source_id"].isin(catalog["source_id"])
        if missing.any():
            catalog = pd.concat([catalog, seed.loc[missing]], ignore_index=True)

    catalog = catalog.drop_duplicates(subset=["source_id"]).reset_index(drop=True)
    # Never let a Gaia RV column sneak in if a future query adds one.
    for col in ("radial_velocity", "radial_velocity_error"):
        if col in catalog.columns:
            catalog = catalog.drop(columns=[col])

    query_info = {
        **resolved,
        "radius_deg": float(radius_deg),
        "g_max": float(g_max),
        "ruwe_max": float(ruwe_max),
        "plx_snr_min": float(plx_snr_min),
        "parallax_window": parallax_window,
        "reference_parallax": pi0,
        "n_cone": int(n_cone),
        "n_catalog": int(len(catalog)),
    }
    return catalog, query_info


def attach_radial_velocities(catalog, rv_table, how="left"):
    """Merge independently measured RVs onto a Gaia catalogue.

    Gaia radial velocities are not used. Matching is case-insensitive on
    string identifiers.

    Parameters
    ----------
    catalog : DataFrame
        Gaia astrometry.
    rv_table : DataFrame
        Must contain ``radial_velocity`` and ``radial_velocity_error``,
        plus one of ``source_id``, ``designation``, ``name``, or
        ``input_name`` to match on.
    how : str
        pandas merge how; default ``"left"``.

    Returns
    -------
    merged : pandas.DataFrame
    n_rv : int
        Number of stars with a finite merged RV.
    """
    catalog = catalog.copy()
    rv = rv_table.copy()
    for col in ("radial_velocity", "radial_velocity_error"):
        if col in catalog.columns:
            catalog = catalog.drop(columns=[col])
    if "radial_velocity" not in rv.columns or "radial_velocity_error" not in rv.columns:
        raise ValueError(
            "rv_table must contain radial_velocity and radial_velocity_error"
        )

    match_pairs = (
        ("source_id", "source_id"),
        ("designation", "designation"),
        ("name", "name"),
        ("name", "input_name"),
        ("input_name", "name"),
        ("simbad_main_id", "name"),
        ("designation", "name"),
    )
    left_key = right_key = None
    for lk, rk in match_pairs:
        if lk in catalog.columns and rk in rv.columns:
            left_key, right_key = lk, rk
            break
    if left_key is None:
        raise ValueError(
            "Could not find a shared identifier to merge RVs "
            "(need source_id, designation, or name)"
        )

    keep = ["radial_velocity", "radial_velocity_error", right_key]
    rv = rv[keep].dropna(subset=["radial_velocity", "radial_velocity_error"])
    if left_key == "source_id" or right_key == "source_id":
        catalog["_merge_key"] = pd.to_numeric(catalog[left_key], errors="coerce")
        rv["_merge_key"] = pd.to_numeric(rv[right_key], errors="coerce")
    else:
        catalog["_merge_key"] = catalog[left_key].astype(str).str.strip().str.lower()
        rv["_merge_key"] = rv[right_key].astype(str).str.strip().str.lower()

    rv = rv.drop_duplicates(subset=["_merge_key"], keep="first")
    merged = catalog.merge(
        rv[["_merge_key", "radial_velocity", "radial_velocity_error"]],
        on="_merge_key",
        how=how,
        suffixes=("", "_rv"),
    )
    merged = merged.drop(columns=["_merge_key"])
    n_rv = int(merged["radial_velocity"].notna().sum()) if "radial_velocity" in merged.columns else 0
    return merged, n_rv


def load_rv_table(path):
    """Read an RV table from CSV or parquet.

    Parameters
    ----------
    path : path-like

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)
