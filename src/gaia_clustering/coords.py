"""Pack Gaia observables and convert to ICRS phase space.

Observable order and units
--------------------------
``ra``, ``dec``              deg
``parallax``                 mas
``pmra``, ``pmdec``          mas/yr  (μ_α* = μ_α cos δ, μ_δ)
``radial_velocity``          km/s, optional per star (see ``use_rv``)

Phase-space order after conversion: ``(X, Y, Z [pc], Vx, Vy, Vz [km/s])``.

Gaia ``ra_error`` / ``dec_error`` are mas and are converted to degrees
internally. Radial velocity is optional: stars that do not contribute RV
get a placeholder mean and an uninformative uncertainty so the 6D
conversion still runs; extreme deconvolution then ignores the
line-of-sight velocity for those stars.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, SkyCoord

# v [km/s] = AU_KMS * μ [mas/yr] / π [mas]
AU_KMS = 4.74047

_ASTROMETRY_MEAN_COLUMNS = (
    "ra", "dec", "parallax", "pmra", "pmdec",
)
_ASTROMETRY_ERROR_COLUMNS = (
    "ra_error", "dec_error", "parallax_error",
    "pmra_error", "pmdec_error",
)
OBS_MEAN_COLUMNS = _ASTROMETRY_MEAN_COLUMNS + ("radial_velocity",)
OBS_ERROR_COLUMNS = _ASTROMETRY_ERROR_COLUMNS + ("radial_velocity_error",)

# Placeholder RV and its uncertainty when a star does not contribute RV.
# 10^3 km/s is uninformative next to a ~25 km/s field scale, but keeps
# the 6×6 observable covariance numerically positive definite.
MISSING_RV_VALUE = 0.0
MISSING_RV_SIGMA = 1.0e3
_MAS_TO_DEG = 1.0 / 3.6e6
# Distance travelled at 1 km/s in 1 Myr, in pc.
PC_PER_KMS_MYR = (1.0 * u.km / u.s * 1.0 * u.Myr).to(u.pc).value


def resolve_use_rv(data, use_rv=None, n=None, has_cov=False):
    """Boolean mask: True if that star's RV enters the likelihood.

    Parameters
    ----------
    data : DataFrame or ndarray
        Gaia-like table or packed array (see :func:`pack_observables`).
    use_rv : None, bool, str, or array-like of bool
        ``None``
            Use RV where the value (and error, unless ``cov`` supplies it)
            is finite and positive. A boolean ``use_rv`` column, if
            present, is the explicit per-star mask. Missing RV columns
            mean no star contributes RV.
        ``True`` / ``False``
            All stars, or none.
        column name
            Boolean column in a DataFrame.
        array-like of bool, shape ``(n,)``
            Explicit per-star mask. Stars flagged True must have a
            usable RV measurement.
    n : int, optional
        Number of stars; inferred from ``data`` when omitted.
    has_cov : bool
        If True, a usable RV needs only a finite value (the covariance
        supplies the uncertainty).

    Returns
    -------
    ndarray of bool, shape (n,)
    """
    if hasattr(data, "columns"):
        n = len(data)
        vr = (
            np.asarray(data["radial_velocity"], dtype=float)
            if "radial_velocity" in data.columns
            else np.full(n, np.nan)
        )
        if "radial_velocity_error" in data.columns:
            vr_err = np.asarray(data["radial_velocity_error"], dtype=float)
        else:
            vr_err = np.full(n, np.nan)
        if has_cov:
            usable = np.isfinite(vr)
        else:
            usable = np.isfinite(vr) & np.isfinite(vr_err) & (vr_err > 0)
        if use_rv is None:
            if "use_rv" in data.columns:
                use_rv = "use_rv"
            else:
                return usable
        if isinstance(use_rv, str):
            use_rv = np.asarray(data[use_rv], dtype=bool)
    else:
        arr = np.asarray(data, dtype=float)
        if n is None:
            n = arr.shape[0]
        if arr.ndim == 2 and arr.shape[1] == 12:
            usable = np.isfinite(arr[:, 10]) & np.isfinite(arr[:, 11]) & (arr[:, 11] > 0)
        elif arr.ndim == 2 and arr.shape[1] == 6:
            usable = np.isfinite(arr[:, 5])
        else:
            usable = np.ones(n, dtype=bool)
        if use_rv is None:
            return usable

    if np.isscalar(use_rv) and not isinstance(use_rv, (bytes, np.ndarray)):
        requested = np.full(n, bool(use_rv))
    else:
        requested = np.asarray(use_rv, dtype=bool)
        if requested.shape != (n,):
            raise ValueError(
                "use_rv must be a scalar, a column name, or a boolean "
                "array of shape ({},); got {}".format(n, requested.shape)
            )
    if np.any(requested & ~usable):
        raise ValueError(
            "{} star(s) requested use_rv=True but have no usable "
            "radial_velocity / radial_velocity_error".format(
                int(np.sum(requested & ~usable))
            )
        )
    return requested


def _fill_corr(rho, data, a, b, i, j):
    col = "{}_{}_corr".format(a, b)
    if col not in data.columns:
        return
    r = np.clip(np.asarray(data[col], dtype=float), -0.999, 0.999)
    r = np.where(np.isfinite(r), r, 0.0)
    rho[:, i, j] = rho[:, j, i] = r


def _symmetrize_cov(cov, n):
    if cov.shape != (n, 6, 6):
        raise ValueError("cov must have shape (n_stars, 6, 6); got {}".format(cov.shape))
    return 0.5 * (cov + np.swapaxes(cov, -1, -2))


def pack_observables(data, cov=None, use_rv=None):
    """Pack (α, δ, π, μα*, μδ, vr) means and a 6×6 covariance per star.

    Parameters
    ----------
    data : DataFrame or ndarray
        DataFrame with Gaia column names, or an array of shape
        ``(n, 6)`` (means only; requires ``cov``) or ``(n, 12)``
        ``[mean, error]`` interleaved.
    cov : ndarray, optional
        ``(n, 6, 6)`` covariance of the observables. Overrides packed
        errors and correlations. For stars with ``use_rv=False`` the RV
        row/column is replaced by the uninformative placeholder.
    use_rv : None, bool, str, or array-like of bool
        Per-star RV inclusion; see ``resolve_use_rv``.

    Returns
    -------
    obs : ndarray, shape (n, 6)
        ``[ra, dec, parallax, pmra, pmdec, radial_velocity]``. Unused
        RVs are filled with :data:`MISSING_RV_VALUE`.
    obs_cov : ndarray, shape (n, 6, 6)
        Observable covariance in the same units. Unused RV rows/columns
        are replaced by an uninformative placeholder.
    use_rv : ndarray, shape (n,), dtype bool
    """
    if hasattr(data, "columns"):
        n = len(data)
        missing = [c for c in _ASTROMETRY_MEAN_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError("DataFrame is missing columns: {}".format(missing))
        use_rv_mask = resolve_use_rv(data, use_rv=use_rv, n=n, has_cov=cov is not None)
        obs = np.column_stack(
            [np.asarray(data[c], dtype=float) for c in _ASTROMETRY_MEAN_COLUMNS]
        )
        if "radial_velocity" in data.columns:
            vr = np.asarray(data["radial_velocity"], dtype=float)
        else:
            vr = np.full(n, np.nan)
        vr = np.where(use_rv_mask, vr, MISSING_RV_VALUE)
        obs = np.column_stack([obs, vr])
        if cov is not None:
            obs_cov = _symmetrize_cov(np.asarray(cov, dtype=float), n)
        else:
            sig = np.zeros((n, 6))
            if "ra_error" in data.columns:
                sig[:, 0] = np.asarray(data["ra_error"], dtype=float) * _MAS_TO_DEG
            else:
                sig[:, 0] = 0.05 * _MAS_TO_DEG
            if "dec_error" in data.columns:
                sig[:, 1] = np.asarray(data["dec_error"], dtype=float) * _MAS_TO_DEG
            else:
                sig[:, 1] = 0.05 * _MAS_TO_DEG
            for j, name in enumerate(_ASTROMETRY_ERROR_COLUMNS[2:], start=2):
                if name not in data.columns:
                    raise ValueError("DataFrame is missing column: {}".format(name))
                sig[:, j] = np.asarray(data[name], dtype=float)
            if "radial_velocity_error" in data.columns:
                sig[:, 5] = np.asarray(data["radial_velocity_error"], dtype=float)
            else:
                sig[:, 5] = np.nan
            sig[~use_rv_mask, 5] = MISSING_RV_SIGMA
            ast_ok = sig[:, :5]
            if np.any(ast_ok <= 0) or not np.isfinite(ast_ok).all():
                raise ValueError(
                    "All astrometric uncertainties must be positive and finite"
                )
            if np.any(use_rv_mask) and (
                np.any(sig[use_rv_mask, 5] <= 0)
                or not np.isfinite(sig[use_rv_mask, 5]).all()
            ):
                raise ValueError(
                    "Radial-velocity uncertainties must be positive and finite "
                    "for stars with use_rv=True"
                )
            rho = np.zeros((n, 6, 6))
            for i in range(6):
                rho[:, i, i] = 1.0
            _fill_corr(rho, data, "ra", "dec", 0, 1)
            _fill_corr(rho, data, "ra", "parallax", 0, 2)
            _fill_corr(rho, data, "ra", "pmra", 0, 3)
            _fill_corr(rho, data, "ra", "pmdec", 0, 4)
            _fill_corr(rho, data, "dec", "parallax", 1, 2)
            _fill_corr(rho, data, "dec", "pmra", 1, 3)
            _fill_corr(rho, data, "dec", "pmdec", 1, 4)
            _fill_corr(rho, data, "parallax", "pmra", 2, 3)
            _fill_corr(rho, data, "parallax", "pmdec", 2, 4)
            _fill_corr(rho, data, "pmra", "pmdec", 3, 4)
            obs_cov = sig[:, :, None] * rho * sig[:, None, :]
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (6, 12):
            raise ValueError(
                "array data must have shape (n, 6) or (n, 12); got {}".format(arr.shape)
            )
        n = arr.shape[0]
        use_rv_mask = resolve_use_rv(arr, use_rv=use_rv, n=n, has_cov=cov is not None)
        if arr.shape[1] == 6:
            if cov is None:
                raise ValueError("shape (n, 6) supplies only means and requires cov")
            obs = arr.copy()
            obs_cov = _symmetrize_cov(np.asarray(cov, dtype=float), n)
        else:
            obs = arr[:, 0::2].copy()
            sig = arr[:, 1::2].copy()
            sig[~use_rv_mask, 5] = MISSING_RV_SIGMA
            ast_ok = sig[:, :5]
            if np.any(ast_ok <= 0) or not np.isfinite(ast_ok).all():
                raise ValueError(
                    "All astrometric uncertainties must be positive and finite"
                )
            if np.any(use_rv_mask) and (
                np.any(sig[use_rv_mask, 5] <= 0)
                or not np.isfinite(sig[use_rv_mask, 5]).all()
            ):
                raise ValueError(
                    "Radial-velocity uncertainties must be positive and finite "
                    "for stars with use_rv=True"
                )
            if cov is not None:
                obs_cov = _symmetrize_cov(np.asarray(cov, dtype=float), n)
            else:
                obs_cov = np.zeros((n, 6, 6))
                for i in range(6):
                    obs_cov[:, i, i] = sig[:, i] ** 2
        obs[~use_rv_mask, 5] = MISSING_RV_VALUE

    if np.any(use_rv_mask):
        unused = ~use_rv_mask
        obs_cov[unused, 5, :] = 0.0
        obs_cov[unused, :, 5] = 0.0
        obs_cov[unused, 5, 5] = MISSING_RV_SIGMA ** 2
    else:
        obs_cov[:, 5, :] = 0.0
        obs_cov[:, :, 5] = 0.0
        obs_cov[:, 5, 5] = MISSING_RV_SIGMA ** 2

    if not np.isfinite(obs[:, :5]).all():
        raise ValueError(
            "astrometric observables contain NaN or inf; drop stars "
            "without measured astrometry first"
        )
    if np.any(use_rv_mask) and not np.isfinite(obs[use_rv_mask, 5]).all():
        raise ValueError(
            "radial_velocity contains NaN or inf for stars with use_rv=True"
        )
    if np.any(obs[:, 2] <= 0):
        raise ValueError("Parallax must be positive (mas)")
    ast_cov = obs_cov[:, :5, :5]
    try:
        np.linalg.cholesky(ast_cov)
        if np.any(use_rv_mask):
            np.linalg.cholesky(obs_cov[use_rv_mask])
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Each star's observation covariance must be positive definite"
        ) from exc
    return obs, obs_cov, use_rv_mask


def _observables_to_phase(obs):
    ra, dec, plx, pmra, pmdec, vr = np.asarray(obs, dtype=float).T
    coord = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=(1.0 / plx) * u.kpc,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        radial_velocity=vr * u.km / u.s,
        frame="icrs",
    )
    xyz = np.vstack(coord.cartesian.xyz.to_value(u.pc)).T
    vxyz = np.vstack(coord.velocity.d_xyz.to_value(u.km / u.s)).T
    return np.hstack([xyz, vxyz])


def _phase_jacobian(obs, rel=1e-6, abs_step=None):
    """Central-difference Jacobian, shape (n, 6_phase, 6_obs)."""
    obs = np.asarray(obs, dtype=float)
    n = obs.shape[0]
    if abs_step is None:
        abs_step = np.array([1e-8, 1e-8, 1e-5, 1e-5, 1e-5, 1e-4])
    jac = np.zeros((n, 6, 6))
    for j in range(6):
        h = rel * np.maximum(np.abs(obs[:, j]), 1.0) + abs_step[j]
        obs_hi = obs.copy()
        obs_lo = obs.copy()
        obs_hi[:, j] += h
        obs_lo[:, j] -= h
        if j == 2:
            obs_lo[:, 2] = np.maximum(obs_lo[:, 2], 0.5 * obs[:, 2])
        jac[:, :, j] = (_observables_to_phase(obs_hi) - _observables_to_phase(obs_lo)) / (
            obs_hi[:, [j]] - obs_lo[:, [j]]
        )
    return jac


def observables_to_phase_space(obs, obs_cov=None):
    """Convert Gaia observables to ICRS ``(X, Y, Z, V_X, V_Y, V_Z)``.

    Parameters
    ----------
    obs : ndarray, shape (n, 6)
        ``[ra, dec, parallax, pmra, pmdec, radial_velocity]`` in deg,
        deg, mas, mas/yr, mas/yr, km/s.
    obs_cov : ndarray, shape (n, 6, 6), optional
        Covariance of ``obs`` in those units. The returned
        ``phase_cov`` is the linearized :math:`J C J^\\top`.

    Returns
    -------
    phase : ndarray, shape (n, 6)
        Positions in pc and space velocities in km/s.
    phase_cov : ndarray, shape (n, 6, 6), or None
    """
    obs = np.asarray(obs, dtype=float)
    phase = _observables_to_phase(obs)
    if obs_cov is None:
        return phase, None
    jac = _phase_jacobian(obs)
    phase_cov = np.einsum("nij,njk,nlk->nil", jac, obs_cov, jac)
    phase_cov = 0.5 * (phase_cov + np.swapaxes(phase_cov, -1, -2))
    ridge = np.diag(np.array([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6]))
    phase_cov = phase_cov + ridge
    return phase, phase_cov


def phase_space_to_observables(phase):
    """Inverse of :func:`observables_to_phase_space` (means only).

    Parameters
    ----------
    phase : ndarray, shape (n, 6)
        ICRS ``(X, Y, Z)`` in pc and ``(V_X, V_Y, V_Z)`` in km/s.

    Returns
    -------
    ndarray, shape (n, 6)
        ``[ra, dec, parallax, pmra, pmdec, radial_velocity]``.
    """
    phase = np.asarray(phase, dtype=float)
    n = phase.shape[0]
    if n == 0:
        return np.zeros((0, 6))
    rep = CartesianRepresentation(
        phase[:, 0] * u.pc,
        phase[:, 1] * u.pc,
        phase[:, 2] * u.pc,
    )
    dif = CartesianDifferential(
        phase[:, 3] * u.km / u.s,
        phase[:, 4] * u.km / u.s,
        phase[:, 5] * u.km / u.s,
    )
    coord = SkyCoord(rep.with_differentials(dif), frame="icrs")
    dist_kpc = coord.distance.to_value(u.kpc)
    return np.column_stack([
        coord.ra.deg,
        coord.dec.deg,
        1.0 / dist_kpc,
        coord.pm_ra_cosdec.to_value(u.mas / u.yr),
        coord.pm_dec.to_value(u.mas / u.yr),
        coord.radial_velocity.to_value(u.km / u.s),
    ])


def coast_dataframe(df, traceback_myr):
    """Return a copy of ``df`` coasted ``traceback_myr`` into the past.

    ICRS space velocities are held fixed: :math:`r_\\mathrm{past} = r - v t`
    with ``t`` in Myr. Sky coordinates and proper motions are recomputed
    at the new position. Stars without RV are coasted with a placeholder
    :math:`v_r = 0` (tangential motion only).

    Parameters
    ----------
    df : DataFrame
        Must contain ``ra``, ``dec``, ``parallax``, ``pmra``, ``pmdec``.
    traceback_myr : float
        Lookback time in Myr. Zero returns ``df`` unchanged; negative
        values forecast forward.

    Returns
    -------
    pandas.DataFrame
    """
    t = float(traceback_myr)
    if t == 0.0:
        return df
    cols = []
    for col in OBS_MEAN_COLUMNS:
        if col == "radial_velocity":
            if col in df.columns:
                vr = np.asarray(df[col], dtype=float)
            else:
                vr = np.full(len(df), np.nan)
            vr = np.where(np.isfinite(vr), vr, MISSING_RV_VALUE)
            cols.append(vr)
        else:
            cols.append(np.asarray(df[col], dtype=float))
    obs = np.column_stack(cols)
    phase = _observables_to_phase(obs)
    phase_past = phase.copy()
    phase_past[:, :3] = phase[:, :3] - phase[:, 3:] * (t * PC_PER_KMS_MYR)
    dist = np.linalg.norm(phase_past[:, :3], axis=1)
    if np.any(~np.isfinite(dist) | (dist < 1e-3)):
        raise ValueError(
            "traceback of {:g} Myr placed a star too close to the origin "
            "to convert to sky coordinates".format(t)
        )
    obs_past = phase_space_to_observables(phase_past)
    out = df.copy()
    for i, col in enumerate(OBS_MEAN_COLUMNS):
        out[col] = obs_past[:, i]
    return out


def astrometric_covariance(df):
    """Per-star :math:`3\\times 3` covariance of ``(parallax, pmra, pmdec)``.

    Units are mas and mas/yr. Off-diagonal terms use Gaia ``*_corr``
    columns when present.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    ndarray, shape (n, 3, 3)
    """
    n = len(df)
    sig = np.column_stack([
        np.asarray(df["parallax_error"], dtype=float),
        np.asarray(df["pmra_error"], dtype=float),
        np.asarray(df["pmdec_error"], dtype=float),
    ])
    if np.any(sig <= 0) or not np.isfinite(sig).all():
        raise ValueError("parallax and proper-motion uncertainties must be positive")
    rho = np.zeros((n, 3, 3))
    rho[:, 0, 0] = rho[:, 1, 1] = rho[:, 2, 2] = 1.0
    _fill_corr(rho, df, "parallax", "pmra", 0, 1)
    _fill_corr(rho, df, "parallax", "pmdec", 0, 2)
    _fill_corr(rho, df, "pmra", "pmdec", 1, 2)
    cov = sig[:, :, None] * rho * sig[:, None, :]
    return 0.5 * (cov + np.swapaxes(cov, -1, -2))


def star_uses_rv(df, result=None):
    """Per-star mask of stars that contribute an independent RV.

    Parameters
    ----------
    df : DataFrame
    result : dict, optional
        Clustering result with a ``use_rv`` array; preferred over columns.

    Returns
    -------
    ndarray of bool, shape (n,)
    """
    n = len(df)
    if result is not None and result.get("use_rv") is not None:
        mask = np.asarray(result["use_rv"], dtype=bool)
        if mask.shape == (n,):
            return mask
    if "use_rv" in df.columns:
        return np.asarray(df["use_rv"], dtype=bool)
    if "radial_velocity" not in df.columns:
        return np.zeros(n, dtype=bool)
    vr = np.asarray(df["radial_velocity"], dtype=float)
    if "radial_velocity_error" in df.columns:
        err = np.asarray(df["radial_velocity_error"], dtype=float)
        return np.isfinite(vr) & np.isfinite(err) & (err > 0)
    return np.isfinite(vr)
