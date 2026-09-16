"""Two-stage association clustering in 6D phase space.

Stage 1: association vs field in ICRS (X, Y, Z, Vx, Vy, Vz) with
extreme deconvolution (one compact component + a wide field). Radial
velocity may be missing or excluded per star; those stars still
constrain position and tangential velocity.

Stage 2: number of spatial clusters among association stars, compared
by BIC for K = 1, 2, 3.

The hard assignment is ``cluster_id`` (-1 = field); ``member_prob`` is
P(association).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

from gaia_clustering.coords import (
    PC_PER_KMS_MYR,
    _MAS_TO_DEG,
    pack_observables,
    observables_to_phase_space,
    phase_space_to_observables,
)
from gaia_clustering.xd import compare_cluster_counts, fit_extreme_deconvolution


def cluster_association(
    data,
    cov=None,
    k_grid=(1, 2, 3),
    membership_probability_threshold=0.5,
    use_rv=None,
    **fit_kwargs,
):
    """Two-stage Gaussian-mixture clustering of an association.

    Stage 1 clusters in 6D ICRS phase space (one association component
    plus a wide field). Stage 2 clusters member positions and selects
    :math:`K` by BIC.

    ``data`` is a Gaia-like DataFrame or a packed array; see
    :func:`~gaia_clustering.coords.pack_observables`. Radial velocities
    are used only for stars selected by ``use_rv``. Stars without RV
    still constrain position and tangential velocity.

    Parameters
    ----------
    data : DataFrame or ndarray
        Gaia astrometry, optionally with independent RVs.
    cov : ndarray, shape (n, 6, 6), optional
        Observable covariance; overrides packed Gaia errors.
    k_grid : sequence of int
        Spatial cluster counts compared among members.
    membership_probability_threshold : float
        ``P(member)`` cut for hard membership and stage 2.
    use_rv : None, bool, str, or array-like of bool
        Per-star RV inclusion; see
        :func:`~gaia_clustering.coords.resolve_use_rv`.
    **fit_kwargs
        Forwarded to :func:`~gaia_clustering.xd.fit_extreme_deconvolution`
        (e.g. ``n_init``, ``max_iter``, ``random_state``).

    Returns
    -------
    dict
        ``obs``, ``obs_cov``, ``phase``, ``phase_cov``, ``use_rv``,
        ``comparison`` (spatial BIC table), ``best`` (membership fields),
        ``fits``, and ``table`` (input catalogue plus membership columns).
    """
    fit_kwargs = dict(fit_kwargs)
    fit_kwargs.pop("include_field", None)
    obs, obs_cov, use_rv_mask = pack_observables(data, cov=cov, use_rv=use_rv)
    phase, phase_cov = observables_to_phase_space(obs, obs_cov)
    membership = fit_extreme_deconvolution(
        phase, phase_cov, n_clusters=1, include_field=True, **fit_kwargs
    )
    member = membership["member_prob"] >= membership_probability_threshold
    cluster_id = np.full(len(phase), -1, dtype=int)
    spatial_comparison = spatial_best = spatial_fits = None
    if member.sum() >= max(6, min(k_grid) + 2):
        spatial_comparison, spatial_best, spatial_fits = compare_cluster_counts(
            phase[member, :3],
            phase_cov[member][:, :3, :3],
            k_grid=k_grid,
            include_field=False,
            **fit_kwargs,
        )
        cluster_id[member] = spatial_best["cluster_id"]
        structure = spatial_best["structure"]
        preferred_K = spatial_best["preferred_K"]
    else:
        structure = "too few members to test spatial structure"
        preferred_K = 1
        if member.any():
            cluster_id[member] = 0

    result = {
        "n_clusters": preferred_K,
        "preferred_K": preferred_K,
        "structure": structure,
        "member_prob": membership["member_prob"],
        "cluster_id": cluster_id,
        "membership": membership,
        "spatial_best": spatial_best,
        "use_rv": use_rv_mask,
    }
    table = None
    if hasattr(data, "copy"):
        table = data.copy()
        table["member_prob"] = membership["member_prob"]
        table["cluster_id"] = cluster_id
        table["is_member"] = member
        table["use_rv"] = use_rv_mask
    return {
        "obs": obs,
        "obs_cov": obs_cov,
        "phase": phase,
        "phase_cov": phase_cov,
        "use_rv": use_rv_mask,
        "comparison": spatial_comparison,
        "best": result,
        "fits": spatial_fits,
        "table": table,
    }


def cluster_recovery(true_labels, member_prob, threshold=0.5):
    """Confusion matrix of association vs field against synthetic labels.

    Parameters
    ----------
    true_labels : array-like
        ``>= 0`` for true members, ``-1`` for field.
    member_prob : array-like
        Predicted ``P(member)``.
    threshold : float
        Hard-assignment cut.

    Returns
    -------
    pandas.DataFrame
        2×2 counts with rows "clustered as association/field".
    """
    y_true = np.asarray(true_labels) >= 0
    y_pred = np.asarray(member_prob) >= threshold
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    return pd.DataFrame(
        {
            "true association": [tp, fn],
            "true field": [fp, tn],
        },
        index=["clustered as association", "clustered as field"],
    )


def synthesize_two_lobe_association(
    n_per_lobe=60,
    n_field=40,
    lobe_offset_pc=22.0,
    lobe_size_pc=7.0,
    v_disp_kms=1.2,
    v_outward_kms=2.0,
    expansion_age_myr=8.0,
    random_state=0,
    rv_fraction=0.0,
):
    """Draw a two-lobe expanding association plus field stars.

    Members coast at constant velocity. Birth positions are compact;
    present positions are ``r_birth + v_pec * t``.

    Parameters
    ----------
    n_per_lobe, n_field : int
        Number of stars in each lobe and in the field.
    lobe_offset_pc, lobe_size_pc : float
        Lobe centre offset and birth-position scatter.
    v_disp_kms : float
        Isotropic peculiar-velocity jitter of members (km/s).
    v_outward_kms : float
        Mean recession speed from the association centre (km/s).
    expansion_age_myr : float
        Time since birth.
    random_state : int
        RNG seed.
    rv_fraction : float
        Fraction of stars that keep a usable RV (the rest have NaN RV,
        matching the Gaia-RV-ignored default).

    Returns
    -------
    df : pandas.DataFrame
        Gaia-like catalogue with ``true_label`` (``0, 1`` = lobes, ``-1``
        = field).
    meta : dict
        True phase space, centre, and expansion parameters.
    """
    rng = np.random.default_rng(random_state)
    center = SkyCoord(
        ra=302.5 * u.deg,
        dec=36.6 * u.deg,
        distance=2100 * u.pc,
        pm_ra_cosdec=-2.80 * u.mas / u.yr,
        pm_dec=-5.90 * u.mas / u.yr,
        radial_velocity=-28.0 * u.km / u.s,
        frame="icrs",
    )
    r0 = np.array(center.cartesian.xyz.to_value(u.pc))
    v0 = np.array(center.velocity.d_xyz.to_value(u.km / u.s))
    los = r0 / np.linalg.norm(r0)
    e1 = np.cross(los, np.array([0.0, 0.0, 1.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(los, e1)

    def draw_lobe(sign, n):
        offset = sign * lobe_offset_pc * e1
        r_birth = r0 + offset + rng.normal(0.0, lobe_size_pc, size=(n, 3))
        dr_birth = r_birth - r0
        rhat = dr_birth / np.clip(
            np.linalg.norm(dr_birth, axis=1, keepdims=True), 1e-8, None
        )
        v_pec = v_outward_kms * rhat + rng.normal(0.0, v_disp_kms, size=(n, 3))
        r = r_birth + v_pec * (expansion_age_myr * PC_PER_KMS_MYR)
        v = v0 + v_pec
        return np.hstack([r, v])

    lobe_a = draw_lobe(+1.0, n_per_lobe)
    lobe_b = draw_lobe(-1.0, n_per_lobe)
    field_r = r0 + rng.uniform(-90.0, 90.0, size=(n_field, 3))
    field_v = v0 + rng.normal(0.0, 18.0, size=(n_field, 3))
    field = np.hstack([field_r, field_v])

    phase_true = np.vstack([lobe_a, lobe_b, field])
    labels = np.concatenate([
        np.zeros(n_per_lobe, dtype=int),
        np.ones(n_per_lobe, dtype=int),
        np.full(n_field, -1, dtype=int),
    ])
    obs_true = phase_space_to_observables(phase_true)

    n = obs_true.shape[0]
    sig = np.column_stack([
        np.full(n, 0.04 * _MAS_TO_DEG),
        np.full(n, 0.04 * _MAS_TO_DEG),
        rng.uniform(0.010, 0.018, n),
        rng.uniform(0.010, 0.018, n),
        rng.uniform(0.010, 0.018, n),
        rng.uniform(1.5, 4.0, n),
    ])
    rho_pm = 0.15
    obs_cov = np.zeros((n, 6, 6))
    for i in range(6):
        obs_cov[:, i, i] = sig[:, i] ** 2
    obs_cov[:, 2, 3] = obs_cov[:, 3, 2] = rho_pm * sig[:, 2] * sig[:, 3]
    obs_cov[:, 2, 4] = obs_cov[:, 4, 2] = 0.08 * sig[:, 2] * sig[:, 4]
    obs_cov[:, 3, 4] = obs_cov[:, 4, 3] = -0.12 * sig[:, 3] * sig[:, 4]

    noise = np.array([
        rng.multivariate_normal(np.zeros(6), obs_cov[i]) for i in range(n)
    ])
    obs = obs_true + noise
    obs[:, 2] = np.clip(obs[:, 2], 0.05, None)

    keep_rv = rng.random(n) < float(rv_fraction)
    vr = obs[:, 5].copy()
    vr_err = sig[:, 5].copy()
    vr[~keep_rv] = np.nan
    vr_err[~keep_rv] = np.nan

    df = pd.DataFrame({
        "name": [f"star {i}" for i in range(n)],
        "ra": obs[:, 0],
        "dec": obs[:, 1],
        "parallax": obs[:, 2],
        "pmra": obs[:, 3],
        "pmdec": obs[:, 4],
        "radial_velocity": vr,
        "ra_error": sig[:, 0] / _MAS_TO_DEG,
        "dec_error": sig[:, 1] / _MAS_TO_DEG,
        "parallax_error": sig[:, 2],
        "pmra_error": sig[:, 3],
        "pmdec_error": sig[:, 4],
        "radial_velocity_error": vr_err,
        "parallax_pmra_corr": np.full(n, rho_pm),
        "parallax_pmdec_corr": np.full(n, 0.08),
        "pmra_pmdec_corr": np.full(n, -0.12),
        "true_label": labels,
        "phot_g_mean_mag": rng.uniform(6.0, 12.0, n),
    })
    meta = {
        "r0": r0,
        "v0": v0,
        "v_outward_kms": v_outward_kms,
        "expansion_age_myr": expansion_age_myr,
        "e1": e1,
        "e2": e2,
        "los": los,
        "phase_true": phase_true,
        "obs_true": obs_true,
        "obs_cov": obs_cov,
        "v_disp_kms": v_disp_kms,
    }
    return df, meta
