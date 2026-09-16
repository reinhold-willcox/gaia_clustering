"""Hierarchical Bayesian inference of the association velocity Gaussian.

Generative model per star
-------------------------
Population velocities ``v = (vα, vδ[, vr]) ~ N(μ, Σ)``. Each star has a
latent velocity drawn from that Gaussian (non-centered) and an independent
latent distance with prior ``p(d) ∝ d²``; parallax is ``π = 1/d``.
Proper motions are ``μα,δ = vα,δ π / 4.74047``, then compared to the
Gaia observables ``(π, μα, μδ)`` with their 3×3 covariance.

Radial velocity is optional per star. Stars with an independent RV
measurement contribute a 1D Gaussian likelihood on ``vr``. Gaia RVs
are never used. If no star has RV, the population is 2D ``(vα, vδ)``.
"""

from __future__ import annotations

import inspect
import os
import warnings
from pathlib import Path

# Keep PyTensor's compile cache in a writable project directory when
# $HOME/.pytensor is locked or not writable (notebooks, sandboxes).
_compiledir = Path.cwd() / "data" / "cache" / "pytensor"
try:
    _compiledir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTENSOR_FLAGS", "compiledir={}".format(_compiledir))
except OSError:
    pass

import numpy as np
import pytensor
import pytensor.tensor as pt
import pymc as pm
import arviz as az

try:
    pytensor.config.compiledir = str(_compiledir)
except Exception:
    pass

from gaia_clustering.coords import AU_KMS, astrometric_covariance, star_uses_rv


def _prepare_velocity_inputs(df, use_rv=None):
    """Pack astrometry and optional RVs for the hierarchical model."""
    n = len(df)
    if n < 2:
        raise ValueError("Need at least 2 stars to constrain a velocity covariance")
    pi = np.asarray(df["parallax"], dtype=float)
    pma = np.asarray(df["pmra"], dtype=float)
    pmd = np.asarray(df["pmdec"], dtype=float)
    if np.any(pi <= 0) or not np.isfinite(pi).all():
        raise ValueError("Parallax must be positive and finite")
    ast_means = np.column_stack([pi, pma, pmd])
    ast_cov = astrometric_covariance(df)
    try:
        ast_chol = np.linalg.cholesky(ast_cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Each star's (π, μα, μδ) covariance must be positive definite"
        ) from exc

    rv_mask = star_uses_rv(df) if use_rv is None else np.asarray(use_rv, dtype=bool)
    if rv_mask.shape != (n,):
        raise ValueError("use_rv must have shape ({},)".format(n))
    n_rv = int(rv_mask.sum())
    n_vel = 3 if n_rv > 0 else 2

    v_alpha = AU_KMS * pma / pi
    v_delta = AU_KMS * pmd / pi
    if n_vel == 3:
        vr = np.asarray(df["radial_velocity"], dtype=float)
        vr_err = np.asarray(df["radial_velocity_error"], dtype=float)
        if np.any(vr_err[rv_mask] <= 0) or not np.isfinite(vr[rv_mask]).all():
            raise ValueError("RV values/errors must be finite and positive for use_rv stars")
        v_obs = np.column_stack([v_alpha, v_delta, np.where(rv_mask, vr, np.nan)])
        v_obs_mean = np.array([
            np.nanmean(v_alpha),
            np.nanmean(v_delta),
            np.nanmean(vr[rv_mask]),
        ])
        vr_obs = vr[rv_mask]
        vr_sigma = vr_err[rv_mask]
        rv_index = np.flatnonzero(rv_mask)
    else:
        v_obs = np.column_stack([v_alpha, v_delta])
        v_obs_mean = np.mean(v_obs, axis=0)
        vr_obs = vr_sigma = rv_index = None

    return {
        "n_stars": n,
        "n_vel": n_vel,
        "n_rv": n_rv,
        "ast_means": ast_means,
        "ast_chol": ast_chol,
        "v_obs": v_obs,
        "v_obs_mean": v_obs_mean,
        "rv_mask": rv_mask,
        "rv_index": rv_index,
        "vr_obs": vr_obs,
        "vr_sigma": vr_sigma,
        "pi": pi,
    }


def build_velocity_model(df, use_rv=None, parallax_bounds=None):
    """Build the PyMC model (no sampling).

    Parameters
    ----------
    df : DataFrame
        Member stars with Gaia astrometry. Optional
        ``radial_velocity`` / ``radial_velocity_error`` for a subset.
    use_rv : array-like of bool, optional
        Per-star RV inclusion. Default: stars with finite independent RV.
    parallax_bounds : (float, float), optional
        ``(π_min, π_max)`` in mas for the volume-uniform distance prior.
        Defaults to a factor of four around the observed parallaxes.

    Returns
    -------
    model : pymc.Model
    v_obs : ndarray, shape (n_stars, n_vel)
        Naive inverted velocities for plotting and initialisation.
    prep : dict
        Packed astrometry, RV mask, and dimensions.
    """
    prep = _prepare_velocity_inputs(df, use_rv=use_rv)
    n_stars = prep["n_stars"]
    n_vel = prep["n_vel"]
    ast_means = prep["ast_means"]
    ast_chol = prep["ast_chol"]
    v_obs = prep["v_obs"]
    v_obs_mean = prep["v_obs_mean"]
    pi_obs = prep["pi"]

    if parallax_bounds is None:
        pi_min = 0.25 * float(np.min(pi_obs))
        pi_max = 4.0 * float(np.max(pi_obs))
    else:
        pi_min, pi_max = map(float, parallax_bounds)
    if not (0.0 < pi_min < pi_max):
        raise ValueError("parallax_bounds must satisfy 0 < π_min < π_max")
    d_min = 1.0 / pi_max
    d_max = 1.0 / pi_min
    d_min3 = d_min ** 3
    d_max3 = d_max ** 3
    d_obs = 1.0 / pi_obs
    d3_init = np.clip(d_obs ** 3, d_min3 * (1.0 + 1e-6), d_max3 * (1.0 - 1e-6))

    v_obs_std = np.nanstd(v_obs, axis=0, ddof=1)
    v_obs_std = np.where(np.isfinite(v_obs_std) & (v_obs_std > 0.0), v_obs_std, 1.0)
    z_init = np.clip((np.nan_to_num(v_obs, nan=0.0) - v_obs_mean) / v_obs_std, -2.0, 2.0)

    with pm.Model() as model:
        Venv = pm.Normal("Venv", mu=v_obs_mean, sigma=50.0, shape=n_vel)
        chol, corr, stds = pm.LKJCholeskyCov(
            "Senv",
            n=n_vel,
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(sigma=30.0),
            compute_corr=True,
        )
        cov = pm.Deterministic("cov", chol.dot(chol.T))
        pm.Deterministic("corr", corr)
        pm.Deterministic("stds", stds)
        pm.Deterministic("sigma_1d", pt.sqrt(pt.mean(stds ** 2)))

        z = pm.Normal(
            "v_true_z",
            mu=0.0,
            sigma=1.0,
            shape=(n_stars, n_vel),
            initval=z_init,
        )
        v_true = pm.Deterministic("v_true", Venv + pt.dot(z, chol.T))

        d3 = pm.Uniform(
            "d^3",
            lower=d_min3,
            upper=d_max3,
            shape=n_stars,
            initval=d3_init,
        )
        pi_true = pm.Deterministic("π_true", 1.0 / d3 ** (1.0 / 3.0))

        pmad_true = v_true[:, :2] * pi_true[:, None] / AU_KMS
        predicted_ast = pt.concatenate([pi_true[:, None], pmad_true], axis=1)
        pm.MvNormal(
            "Likelihood(π,μα,μδ)",
            mu=predicted_ast,
            chol=ast_chol,
            observed=ast_means,
        )

        if n_vel == 3 and prep["n_rv"] > 0:
            rv_index = np.asarray(prep["rv_index"], dtype="int64")
            pm.Normal(
                "Likelihood(vr)",
                mu=v_true[rv_index, 2],
                sigma=prep["vr_sigma"],
                observed=prep["vr_obs"],
            )

    return model, v_obs, prep


def _summarize_posterior(idata):
    summary_kwargs = dict(
        var_names=["Venv", "cov", "corr", "stds", "sigma_1d"],
        round_to=5,
    )
    summary_params = inspect.signature(az.summary).parameters
    if "ci_prob" in summary_params:
        summary_kwargs["ci_prob"] = 0.95
    elif "hdi_prob" in summary_params:
        summary_kwargs["hdi_prob"] = 0.95
    if "skipna" in summary_params:
        summary_kwargs["skipna"] = True
    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return az.summary(idata, **summary_kwargs)


def sample_velocity_model(
    model,
    v_obs,
    prep,
    draws=1000,
    tune=1000,
    chains=4,
    cores=4,
    target_accept=0.95,
    random_seed=0,
    return_inferencedata=True,
):
    """Sample :func:`build_velocity_model` with NUTS.

    Parameters
    ----------
    model : pymc.Model
    v_obs : ndarray
        Naive inverted velocities from the builder.
    prep : dict
        Bookkeeping from :func:`build_velocity_model`.
    draws, tune, chains, cores : int
        Forwarded to :func:`pymc.sample`.
    target_accept : float
        NUTS target acceptance probability.
    random_seed : int
    return_inferencedata : bool
        If True (default), include ArviZ ``InferenceData`` as ``trace``.

    Returns
    -------
    dict
        Posterior-mean ``mean``, ``cov``, ``corr``, ``std``, scalar
        ``sigma_1d`` with 16/50/84% ``sigma_1d_quantiles``, plus
        ``summary`` and optionally ``trace``.
    """
    with model:
        idata = pm.sample(
            draws,
            tune=tune,
            chains=chains,
            cores=cores,
            nuts={"target_accept": target_accept},
            random_seed=random_seed,
            return_inferencedata=True,
        )

    post = idata.posterior
    n_vel = prep["n_vel"]
    components = ("alpha", "delta", "r")[:n_vel]
    std = post["stds"].mean(dim=("chain", "draw")).values
    cov = post["cov"].mean(dim=("chain", "draw")).values
    sigma_1d = float(post["sigma_1d"].mean(dim=("chain", "draw")).values)
    sigma_1d_samples = np.asarray(post["sigma_1d"].values).ravel()
    lo, med, hi = np.quantile(sigma_1d_samples, [0.16, 0.50, 0.84])

    result = {
        "mean": post["Venv"].mean(dim=("chain", "draw")).values,
        "cov": cov,
        "corr": post["corr"].mean(dim=("chain", "draw")).values,
        "std": std,
        "sigma_1d": sigma_1d,
        "sigma_1d_quantiles": (float(lo), float(med), float(hi)),
        "v_obs": np.asarray(v_obs, dtype=float),
        "n_vel": n_vel,
        "n_rv": prep["n_rv"],
        "n_stars": prep["n_stars"],
        "components": components,
        "summary": _summarize_posterior(idata),
        "model": model,
    }
    if return_inferencedata:
        result["trace"] = idata
    return result


def fit_velocity_dispersion(
    df,
    use_rv=None,
    parallax_bounds=None,
    draws=1000,
    tune=1000,
    chains=4,
    cores=4,
    target_accept=0.95,
    random_seed=0,
    return_inferencedata=True,
):
    """Build and sample the association velocity Gaussian.

    ``df`` should already be restricted to likely members. See
    :func:`build_velocity_model` for the generative model.

    Parameters
    ----------
    df : DataFrame
        Member stars.
    use_rv : array-like of bool, optional
    parallax_bounds : tuple of float, optional
        ``(π_min, π_max)`` in mas.
    draws, tune, chains, cores, target_accept, random_seed, return_inferencedata
        Forwarded to :func:`sample_velocity_model`.

    Returns
    -------
    dict
        Same keys as :func:`sample_velocity_model`.
    """
    model, v_obs, prep = build_velocity_model(
        df, use_rv=use_rv, parallax_bounds=parallax_bounds
    )
    return sample_velocity_model(
        model,
        v_obs,
        prep,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        random_seed=random_seed,
        return_inferencedata=return_inferencedata,
    )
