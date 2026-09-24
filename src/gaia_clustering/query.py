"""Resolve a cluster or star name and pull a Gaia DR3 neighbourhood.

Gaia radial velocities are never used. Independent RVs can be merged
afterwards via ``attach_radial_velocities``.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils.data import download_file, get_file_contents

SESAME_URL = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxpI/SNV?"
GAIA_TAP_SYNC = "https://gea.esac.esa.int/tap-server/tap/sync"
SIMBAD_TAP_SYNC = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"


class NameResolutionError(ValueError):
    """CDS Sesame/SIMBAD could not resolve a target name."""


class GaiaQueryError(RuntimeError):
    """A Gaia TAP request failed."""


def default_cache_dir():
    """Return the default directory for pickled Gaia TAP results.

    Honours ``GAIA_CLUSTERING_CACHE``, then ``$XDG_CACHE_HOME/gaia_clustering``,
    then ``~/.cache/gaia_clustering``. Callers do not need to pass a cache path.
    """
    env = os.environ.get("GAIA_CLUSTERING_CACHE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "gaia_clustering"


def _resolve_cache_dir(cache_dir):
    """``None`` → :func:`default_cache_dir`; ``False`` disables caching."""
    if cache_dir is False:
        return None
    if cache_dir is None:
        return default_cache_dir()
    return Path(cache_dir)

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
    try:
        with urlopen(request, timeout=timeout) as response:
            votable_bytes = response.read()
    except HTTPError as exc:
        if exc.code == 408:
            raise GaiaQueryError(
                "Gaia TAP timed out (HTTP 408). The cone search is probably too "
                "large or too crowded. Try a smaller position_radius_arcmin or a brighter "
                "g_max."
            ) from exc
        raise GaiaQueryError(
            "Gaia TAP request failed (HTTP {}): {}".format(exc.code, exc.reason)
        ) from exc
    except (TimeoutError, URLError) as exc:
        raise GaiaQueryError("Gaia TAP request failed: {}".format(exc)) from exc
    try:
        table = Table.read(BytesIO(votable_bytes), format="votable")
    except Exception as exc:
        raise GaiaQueryError(
            "Gaia TAP returned a response that could not be parsed as a table: "
            "{}".format(exc)
        ) from exc
    df = table.to_pandas()
    df.columns = [c.lower() for c in df.columns]
    if cache_path is not None:
        _write_cache(cache_path, df)
    return df


def _sesame_xml(name):
    url = SESAME_URL + quote(name, safe="")
    try:
        text = get_file_contents(download_file(url, cache=True, show_progress=False))
    except Exception as exc:
        raise NameResolutionError(
            "Could not resolve {!r} via CDS Sesame/SIMBAD: {}".format(name, exc)
        ) from exc
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
    except NameResolutionError:
        raise
    except ET.ParseError as exc:
        raise NameResolutionError(
            "Could not parse the CDS Sesame response for {!r}. "
            "The name may be unrecognized, or the resolver returned invalid XML.".format(name)
        ) from exc

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


def arcmin_to_deg(radius_arcmin):
    """Convert a cone radius from arcminutes to degrees."""
    return float(radius_arcmin) / 60.0


def _as_float_or_none(value):
    if value is None:
        return None
    return float(value)


def query_gaia_cone(
    ra_deg,
    dec_deg,
    radius_arcmin=10,
    g_max=20,
    ruwe_max=None,
    plx_snr_min=None,
    parallax_lo=None,
    parallax_hi=None,
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
    radius_arcmin : float
        Search radius in arcminutes.
    g_max : float or None
        Faint *G* magnitude limit. ``None`` skips the cut.
    ruwe_max : float or None
        Maximum RUWE. ``None`` (default) skips the cut.
    plx_snr_min : float or None
        Minimum ``parallax_over_error``. ``None`` (default) skips the cut.
    parallax_lo, parallax_hi : float or None
        Inclusive parallax limits in mas for the TAP query. ``None``
        skips that bound (aside from ``parallax > 0``).
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
    radius_deg = arcmin_to_deg(radius_arcmin)
    cols = ", ".join(GAIA_COLUMNS)
    where = [
        "1 = CONTAINS(POINT('ICRS', ra, dec), "
        "CIRCLE('ICRS', {ra}, {dec}, {rad}))".format(
            ra=float(ra_deg), dec=float(dec_deg), rad=float(radius_deg),
        ),
        "parallax IS NOT NULL",
        "parallax > 0",
        "pmra IS NOT NULL",
        "pmdec IS NOT NULL",
        "parallax_error > 0",
        "pmra_error > 0",
        "pmdec_error > 0",
        "visibility_periods_used > {}".format(int(visibility_min)),
    ]
    if g_max is not None:
        where.append("phot_g_mean_mag < {}".format(float(g_max)))
    if ruwe_max is not None:
        where.append("ruwe < {}".format(float(ruwe_max)))
    if plx_snr_min is not None:
        where.append("parallax_over_error > {}".format(float(plx_snr_min)))
    if parallax_lo is not None:
        where.append("parallax >= {}".format(float(parallax_lo)))
    if parallax_hi is not None:
        where.append("parallax <= {}".format(float(parallax_hi)))
    adql = (
        "SELECT TOP {top} {cols}\n"
        "FROM gaiadr3.gaia_source\n"
        "WHERE {where}\n"
    ).format(top=int(top), cols=cols, where="\n    AND ".join(where))
    df = gaia_tap_query(adql, cache_dir=cache_dir)
    if "source_id" in df.columns and len(df):
        df["source_id"] = df["source_id"].astype("int64")
    if len(df) >= int(top):
        warnings.warn(
            "Gaia cone returned TOP={} rows; tighten g_max/radius or raise top".format(top),
            RuntimeWarning,
            stacklevel=2,
        )
    return df


PARALLAX_PREVIEW_LO_FACTOR = 0.75
PARALLAX_PREVIEW_HI_FACTOR = 1.5


def _mean_positive_parallax(df):
    if df is None or len(df) == 0 or "parallax" not in getattr(df, "columns", []):
        return None
    plx = np.asarray(df["parallax"], dtype=float)
    plx = plx[np.isfinite(plx) & (plx > 0)]
    if plx.size == 0:
        return None
    return float(np.mean(plx))


def _as_cut_pair(value, name):
    if value is None:
        return None
    seq = tuple(value)
    if len(seq) != 2:
        raise ValueError("{} must be a (lo, hi) pair".format(name))
    lo, hi = float(seq[0]), float(seq[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi >= lo):
        raise ValueError("{} must satisfy lo <= hi with finite values".format(name))
    return (lo, hi)


def resolve_reference_parallax(kind, seed, in_cone):
    """Reference ϖ: the submitted star, otherwise the mean in the cone."""
    if kind != "group" and seed is not None and len(seed) and "parallax" in seed.columns:
        pi = float(seed["parallax"].iloc[0])
        if np.isfinite(pi) and pi > 0:
            return pi, "star"
    pi = _mean_positive_parallax(in_cone)
    if pi is None:
        return None, None
    return pi, "cone_mean"


def resolved_parallax_bounds(info):
    """Absolute parallax cut ``(lo, hi)`` in mas, or ``None`` if disabled."""
    for key in ("parallax_range", "parallax_cuts"):
        cuts = _as_cut_pair(info.get(key), key)
        if cuts is not None:
            return cuts
    window = info.get("parallax_window")
    pi0 = info.get("reference_parallax")
    if window is None or pi0 is None:
        return None
    lo, hi = float(window[0]), float(window[1])
    pi0 = float(pi0)
    if not (np.isfinite(pi0) and pi0 > 0):
        return None
    return (lo * pi0, hi * pi0)


def preview_parallax_bounds(
    bounds,
    lo_fac=PARALLAX_PREVIEW_LO_FACTOR,
    hi_fac=PARALLAX_PREVIEW_HI_FACTOR,
):
    """Widen accepted parallax cuts for the Gaia search / preview sample."""
    if bounds is None:
        return None
    lo, hi = bounds
    return (float(lo_fac) * float(lo), float(hi_fac) * float(hi))


def parallax_bounds_mask(df, bounds):
    """Boolean mask for stars inside absolute parallax bounds (mas)."""
    if bounds is None:
        return np.ones(len(df), dtype=bool)
    lo, hi = bounds
    plx = np.asarray(df["parallax"], dtype=float)
    return (plx >= lo) & (plx <= hi)


def parse_position_center(position_center):
    """Parse an ICRS RA/Dec into degrees.

    Accepts a :class:`~astropy.coordinates.SkyCoord`, an ``(ra, dec)``
    pair in degrees, or a string in the usual Gaia / Astropy forms
    (decimal degrees, ``hh:mm:ss ±dd:mm:ss``, ``XXhYYmZZs ±AAdBBmCCs``).

    Returns
    -------
    ra, dec : float
        ICRS right ascension and declination in degrees.
    """
    if position_center is None:
        raise ValueError("position_center is empty")
    if isinstance(position_center, SkyCoord):
        c = position_center.icrs
        return float(c.ra.deg), float(c.dec.deg)
    if isinstance(position_center, (tuple, list, np.ndarray)):
        if len(position_center) != 2:
            raise ValueError("position_center must be RA and Dec")
        ra_i, dec_i = position_center
        try:
            ra, dec = float(ra_i), float(dec_i)
        except (TypeError, ValueError):
            return parse_position_center("{} {}".format(ra_i, dec_i))
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec
        raise ValueError("position_center RA/Dec must be finite")
    s = " ".join(str(position_center).replace(",", " ").split())
    if not s:
        raise ValueError("position_center is empty")
    low = s.lower()
    looks_sexagesimal = (
        "h" in low or ":" in s or ("d" in low and "m" in low)
    )
    attempts = []
    if looks_sexagesimal:
        attempts.extend((
            lambda: SkyCoord(s),
            lambda: SkyCoord(s, unit=(u.hourangle, u.deg)),
            lambda: SkyCoord(s, unit=u.deg),
        ))
    else:
        attempts.extend((
            lambda: SkyCoord(s, unit=u.deg),
            lambda: SkyCoord(s),
            lambda: SkyCoord(s, unit=(u.hourangle, u.deg)),
        ))
    last_err = None
    for build in attempts:
        try:
            c = build().icrs
            ra, dec = float(c.ra.deg), float(c.dec.deg)
        except Exception as exc:
            last_err = exc
            continue
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec
    raise ValueError(
        "Could not parse position_center={!r} as ICRS RA/Dec. "
        "Try decimal degrees ('308.11 +41.23'), sexagesimal "
        "('20h32m25.8s +41d18m31s'), or a 2-tuple of degrees.".format(
            position_center,
        )
    ) from last_err


def search_center(info):
    """RA/Dec (degrees) of the cone used for the Gaia search."""
    ra = info.get("center_ra", info.get("simbad_ra"))
    dec = info.get("center_dec", info.get("simbad_dec"))
    if ra is None or dec is None or not np.isfinite(ra) or not np.isfinite(dec):
        raise ValueError("query metadata is missing a cone center")
    return float(ra), float(dec)


def name_position(info):
    """RA/Dec (degrees) resolved from the target name (SIMBAD/Sesame)."""
    ra = info.get("simbad_ra")
    dec = info.get("simbad_dec")
    if ra is None or dec is None or not np.isfinite(ra) or not np.isfinite(dec):
        return None
    return float(ra), float(dec)


def apply_query_cuts(extended, info, seed=None):
    """Apply the selected sky, proper-motion, and parallax cuts; update ``info``.

    Accepted stars lie inside the on-sky cone, the proper-motion cone
    (when ``pm_radius`` is set), *and* the parallax range (when set).
    The reference parallax is the submitted star when available;
    otherwise the mean of stars in the on-sky cone.
    """
    ra0, dec0 = search_center(info)
    in_cone = subset_cone(extended, ra0, dec0, info["radius_deg"])
    if seed is None:
        sid = info.get("gaia_source_id")
        if sid is not None and "source_id" in extended.columns:
            match = extended.loc[extended["source_id"] == sid]
            if len(match):
                seed = match
    if info.get("pm_center") is None:
        info["pm_center"] = default_pm_center(extended, info, seed=seed)
    if info.get("reference_parallax_source") != "star":
        pi0, source = resolve_reference_parallax(info.get("kind"), seed, in_cone)
        info["reference_parallax"] = pi0
        info["reference_parallax_source"] = source
    bounds = resolved_parallax_bounds(info)
    if bounds is not None:
        keep = parallax_bounds_mask(in_cone, bounds)
        info["parallax_lo_mas"] = float(bounds[0])
        info["parallax_hi_mas"] = float(bounds[1])
        preview = preview_parallax_bounds(bounds)
        info["parallax_preview_lo_mas"] = float(preview[0])
        info["parallax_preview_hi_mas"] = float(preview[1])
    else:
        keep = np.ones(len(in_cone), dtype=bool)
        info["parallax_lo_mas"] = None
        info["parallax_hi_mas"] = None
        info["parallax_preview_lo_mas"] = None
        info["parallax_preview_hi_mas"] = None
    keep = np.asarray(keep, dtype=bool) & proper_motion_mask(in_cone, info)
    catalog = in_cone.iloc[np.flatnonzero(keep)].copy()
    info["n_cone"] = int(len(in_cone))
    info["n_pm"] = int(proper_motion_mask(extended, info).sum()) if len(extended) else 0
    info["n_catalog"] = int(len(catalog))
    return catalog


def parse_pm_center(pm_center):
    """Parse ``(pmra, pmdec)`` in mas/yr."""
    if pm_center is None:
        raise ValueError("pm_center is empty")
    if not isinstance(pm_center, (tuple, list, np.ndarray)) or len(pm_center) != 2:
        raise ValueError("pm_center must be (pmra, pmdec) in mas/yr")
    pmra, pmdec = float(pm_center[0]), float(pm_center[1])
    if not (np.isfinite(pmra) and np.isfinite(pmdec)):
        raise ValueError("pm_center must be finite")
    return pmra, pmdec


def default_pm_center(df, info, seed=None):
    """Seed-star proper motion, else the median of the position cone."""
    if seed is not None and len(seed) and "pmra" in seed.columns:
        row = seed.iloc[0]
        if np.isfinite(row["pmra"]) and np.isfinite(row["pmdec"]):
            return float(row["pmra"]), float(row["pmdec"])
    sid = info.get("gaia_source_id")
    if sid is not None and df is not None and "source_id" in df.columns:
        match = df.loc[df["source_id"] == sid]
        if len(match):
            row = match.iloc[0]
            if np.isfinite(row.get("pmra", np.nan)) and np.isfinite(row.get("pmdec", np.nan)):
                return float(row["pmra"]), float(row["pmdec"])
    if df is None or len(df) == 0 or "pmra" not in df.columns:
        return None
    try:
        ra0, dec0 = search_center(info)
        cone = subset_cone(df, ra0, dec0, info.get("radius_deg") or 0.0)
    except (KeyError, ValueError):
        cone = df
    pmra = np.asarray(cone["pmra"], dtype=float)
    pmdec = np.asarray(cone["pmdec"], dtype=float)
    ok = np.isfinite(pmra) & np.isfinite(pmdec)
    if not np.any(ok):
        return None
    return float(np.median(pmra[ok])), float(np.median(pmdec[ok]))


def proper_motion_mask(df, info):
    """Boolean mask for the proper-motion cone, or all-True if unset."""
    radius = info.get("pm_radius")
    center = info.get("pm_center")
    if (
        df is None or len(df) == 0 or radius is None or center is None
        or not np.isfinite(radius) or float(radius) <= 0
        or "pmra" not in df.columns or "pmdec" not in df.columns
    ):
        return np.ones(0 if df is None else len(df), dtype=bool)
    pmra = np.asarray(df["pmra"], dtype=float)
    pmdec = np.asarray(df["pmdec"], dtype=float)
    d = np.hypot(pmra - float(center[0]), pmdec - float(center[1]))
    return np.isfinite(d) & (d <= float(radius))


def wrap_ra_deg(ra, ra0):
    """Shift right ascension so it is continuous around ``ra0`` (degrees)."""
    ra = np.asarray(ra, dtype=float)
    ra0 = float(ra0)
    return ra0 + ((ra - ra0 + 180.0) % 360.0 - 180.0)


def on_sky_separation_deg(ra, dec, ra0, dec0):
    """Great-circle separation in degrees from ``(ra0, dec0)``."""
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if ra.size == 0:
        return np.empty(0, dtype=float)
    coords = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    center = SkyCoord(float(ra0) * u.deg, float(dec0) * u.deg, frame="icrs")
    return np.asarray(coords.separation(center).deg, dtype=float)


def subset_cone(df, ra0, dec0, radius_deg):
    """Keep rows whose on-sky position lies inside the cone."""
    if df is None or len(df) == 0:
        return df.copy() if df is not None else df
    sep = on_sky_separation_deg(df["ra"], df["dec"], ra0, dec0)
    return df.loc[sep <= float(radius_deg)].copy()


def sky_circle_radec(ra0, dec0, radius_deg, n=256):
    """ICRS ``(ra, dec)`` vertices of a spherical cone boundary."""
    center = SkyCoord(float(ra0) * u.deg, float(dec0) * u.deg, frame="icrs")
    pa = np.linspace(0.0, 360.0, int(n)) * u.deg
    circ = center.directional_offset_by(pa, float(radius_deg) * u.deg)
    return np.asarray(circ.ra.deg, dtype=float), np.asarray(circ.dec.deg, dtype=float)


def cone_mask(df, ra0, dec0, radius_deg):
    """Boolean mask for :func:`subset_cone`."""
    if df is None or len(df) == 0:
        return np.ones(0, dtype=bool)
    sep = on_sky_separation_deg(df["ra"], df["dec"], ra0, dec0)
    return sep <= float(radius_deg)


def _drop_gaia_rv_columns(df):
    for col in ("radial_velocity", "radial_velocity_error"):
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def _merge_seed(df, seed):
    if seed is None or len(seed) == 0 or "source_id" not in df.columns:
        return df
    missing = ~seed["source_id"].isin(df["source_id"])
    if missing.any():
        df = pd.concat([df, seed.loc[missing]], ignore_index=True)
    return df


def _accepted_bounds_before_tap(kind, seed, parallax_window, parallax_cuts):
    """Accepted ϖ cuts if they can be known before the cone search."""
    info = {
        "kind": kind,
        "parallax_window": None if parallax_window is None else tuple(parallax_window),
        "parallax_cuts": _as_cut_pair(parallax_cuts, "parallax_cuts"),
        "reference_parallax": None,
        "reference_parallax_source": None,
    }
    pi0, source = resolve_reference_parallax(kind, seed, None)
    if pi0 is not None:
        info["reference_parallax"] = pi0
        info["reference_parallax_source"] = source
    return resolved_parallax_bounds(info)


def _load_cone(
    resolved, seed, radius_arcmin, g_max, ruwe_max, plx_snr_min, top, cache_dir,
    parallax_lo=None, parallax_hi=None, ra=None, dec=None,
):
    if ra is None:
        ra = resolved["simbad_ra"]
    if dec is None:
        dec = resolved["simbad_dec"]
    df = query_gaia_cone(
        ra,
        dec,
        radius_arcmin=radius_arcmin,
        g_max=g_max,
        ruwe_max=ruwe_max,
        plx_snr_min=plx_snr_min,
        parallax_lo=parallax_lo,
        parallax_hi=parallax_hi,
        top=top,
        cache_dir=cache_dir,
    )
    df = _merge_seed(df, seed)
    if "source_id" in df.columns:
        df = df.drop_duplicates(subset=["source_id"])
    return _drop_gaia_rv_columns(df.reset_index(drop=True))


def fetch_neighbourhood(
    name,
    position_radius_arcmin=10,
    g_max=20,
    parallax_range=None,
    cache_dir=None,
    top=10000,
    return_extended=False,
    ruwe_max=None,
    plx_snr_min=None,
):
    """Resolve ``name`` and return a Gaia DR3 neighbourhood.

    Parameters
    ----------
    name : str
        Cluster/association name, or a star in the association.
    position_radius_arcmin : float
        TAP cone radius in arcminutes, centred on the name-resolved
        coordinates.
    g_max : float or None
        Faint *G* magnitude limit. ``None`` skips the cut.
    parallax_range : tuple of float or None
        Hard TAP parallax limits in mas, ``(lo, hi)``. ``None``
        (default) does not filter on parallax.
    cache_dir : path-like, False, or None
        Directory for pickled TAP results. ``None`` (default) uses
        :func:`default_cache_dir`. ``False`` disables caching.
    top : int
        Maximum cone-search rows.
    return_extended : bool
        If True, also return the full downloaded table (before later
        :meth:`~gaia_clustering.pipeline.GaiaQuery.refine_search` cuts).
    ruwe_max : float or None
        Maximum RUWE. ``None`` (default) skips the cut.
    plx_snr_min : float or None
        Minimum ``parallax_over_error``. ``None`` (default) skips the cut.

    Returns
    -------
    catalog : pandas.DataFrame
        Gaia astrometry (no radial velocities) inside the search cone
        and optional parallax range.
    query_info : dict
        Resolution metadata, search parameters, and row counts.
    extended : pandas.DataFrame, optional
        Full downloaded table, only if ``return_extended`` is True.
    """
    cache_dir = _resolve_cache_dir(cache_dir)
    resolved = resolve_name(name)
    if resolved["status"] == "unrecognized" or not np.isfinite(resolved["simbad_ra"]):
        raise NameResolutionError(
            "Could not resolve the target name {!r} with CDS Sesame/SIMBAD. "
            "Check the spelling, or look up a SIMBAD identifier at "
            "https://simbad.cds.unistra.fr/simbad/.".format(name)
        )
    print(
        "  resolved {!r} → {} ({})".format(
            name, resolved.get("simbad_main_id") or name, resolved.get("kind"),
        )
    )
    center_ra, center_dec = float(resolved["simbad_ra"]), float(resolved["simbad_dec"])

    position_radius_arcmin = float(position_radius_arcmin)
    radius_deg = arcmin_to_deg(position_radius_arcmin)
    plx_range = _as_cut_pair(parallax_range, "parallax_range")

    seed = None
    if resolved["gaia_source_id"] is not None:
        seed = query_gaia_by_source_ids(
            [resolved["gaia_source_id"]], cache_dir=cache_dir
        )
        if seed is not None and len(seed) == 0:
            seed = None

    extended = _load_cone(
        resolved, seed, position_radius_arcmin, g_max, ruwe_max, plx_snr_min,
        top, cache_dir,
        parallax_lo=None if plx_range is None else plx_range[0],
        parallax_hi=None if plx_range is None else plx_range[1],
        ra=center_ra, dec=center_dec,
    )

    query_info = {
        **resolved,
        "center_ra": float(center_ra),
        "center_dec": float(center_dec),
        "position_center": None,
        "center_position": None,
        "position_radius_arcmin": position_radius_arcmin,
        "radius_arcmin": position_radius_arcmin,
        "radius_deg": radius_deg,
        "search_radius_arcmin": position_radius_arcmin,
        "search_radius_deg": radius_deg,
        "preview_factor": 1.0,
        "preview_radius_arcmin": position_radius_arcmin,
        "preview_radius_deg": radius_deg,
        "g_max": _as_float_or_none(g_max),
        "ruwe_max": _as_float_or_none(ruwe_max),
        "plx_snr_min": _as_float_or_none(plx_snr_min),
        "parallax_range": None if plx_range is None else tuple(plx_range),
        "parallax_window": None,
        "parallax_cuts": None if plx_range is None else tuple(plx_range),
        "n_extended": int(len(extended)),
    }
    catalog = apply_query_cuts(extended, query_info, seed=seed)
    query_info["n_extended"] = int(len(extended))
    if return_extended:
        return catalog, query_info, extended
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
