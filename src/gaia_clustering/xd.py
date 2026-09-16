"""Extreme deconvolution (Bovy, Hogg & Roweis 2011) for noisy Gaia stars.

A Gaussian mixture whose component covariances are the *intrinsic*
scatter; each star is compared to ``T_k + S_i`` where ``S_i`` is that
star's measurement covariance. Optional wide field component with
fixed covariance for association vs field separation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _log_mvn(x, mu, cov):
    """Log density of N(mu, cov) at each row of x. cov is (n, d, d)."""
    d = x.shape[1]
    delta = x - mu
    sign, logdet = np.linalg.slogdet(cov)
    if np.any(sign <= 0):
        return np.full(x.shape[0], -np.inf)
    sol = np.linalg.solve(cov, delta[:, :, None])[..., 0]
    mahal = np.einsum("ni,ni->n", delta, sol)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + mahal)


def _kmeans(x, k, rng, n_init=8, max_iter=80):
    """k-means++. ``x`` should already be scaled."""
    n, d = x.shape
    best_inertia = np.inf
    best_centers = best_labels = None
    for _ in range(n_init):
        centers = np.empty((k, d))
        centers[0] = x[rng.integers(n)]
        closest = np.full(n, np.inf)
        for j in range(1, k):
            dist = np.sum((x - centers[j - 1]) ** 2, axis=1)
            closest = np.minimum(closest, dist)
            p = closest / np.maximum(closest.sum(), 1e-30)
            centers[j] = x[rng.choice(n, p=p)]
        labels = np.zeros(n, dtype=int)
        for _it in range(max_iter):
            dist = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = dist.argmin(axis=1)
            new_centers = centers.copy()
            for j in range(k):
                mask = new_labels == j
                if np.any(mask):
                    new_centers[j] = x[mask].mean(axis=0)
            if np.array_equal(new_labels, labels):
                centers = new_centers
                labels = new_labels
                break
            centers = new_centers
            labels = new_labels
        inertia = np.sum((x - centers[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()
    return best_centers, best_labels


def _scale_vector(d):
    scale6 = np.array([15.0, 15.0, 15.0, 3.0, 3.0, 3.0])
    if d <= 6:
        return scale6[:d]
    return np.resize(scale6, d)


def _default_ridge(d):
    ridge6 = np.array([0.25, 0.25, 0.25, 0.04, 0.04, 0.04])
    return np.diag(ridge6[:d] if d <= 6 else np.resize(ridge6, d))


def _default_field_cov(d):
    std6 = np.array([100.0, 100.0, 100.0, 25.0, 25.0, 25.0])
    std = std6[:d] if d <= 6 else np.resize(std6, d)
    return np.diag(std ** 2)


def _precision_weighted_mean(x, S):
    """Mean of ``x`` with per-dimension weights ``1 / S_ii``.

    Stars whose measurement variance is huge in a coordinate (missing
    RV, after the Jacobian) do not pull that coordinate of the field
    centre toward the placeholder value.
    """
    var = np.clip(np.diagonal(S, axis1=-2, axis2=-1), 1e-30, None)
    w = 1.0 / var
    return (w * x).sum(axis=0) / w.sum(axis=0)


def _e_step(x, S, weights, means, covs):
    n, n_comp = x.shape[0], len(weights)
    log_comp = np.empty((n, n_comp))
    for j in range(n_comp):
        log_comp[:, j] = np.log(weights[j] + 1e-300) + _log_mvn(x, means[j], covs[j] + S)
    m = np.max(log_comp, axis=1, keepdims=True)
    resp = np.exp(log_comp - m)
    resp_sum = resp.sum(axis=1, keepdims=True)
    resp = resp / np.maximum(resp_sum, 1e-300)
    log_like = float(np.sum(m + np.log(np.maximum(resp_sum, 1e-300))))
    return resp, log_like


def _m_step_component(x, S, T, mu, w, ridge):
    """XD M-step for one Gaussian. ``w`` is the responsibility vector."""
    n_eff = float(w.sum())
    Ttot = T + S
    Rt = np.linalg.solve(Ttot, np.repeat(T.T[None, :, :], x.shape[0], axis=0))
    R = np.swapaxes(Rt, -1, -2)
    delta = x - mu
    b = mu + np.einsum("nij,nj->ni", R, delta)
    B = T - np.einsum("nij,jk->nik", R, T)
    mu_new = (w[:, None] * b).sum(axis=0) / n_eff
    centered = b - mu_new
    Tnew = np.einsum("n,ni,nj->ij", w, centered, centered)
    Tnew += np.einsum("n,nij->ij", w, B)
    Tnew /= n_eff
    return mu_new, 0.5 * (Tnew + Tnew.T) + ridge


def fit_extreme_deconvolution(
    x,
    S,
    n_clusters,
    include_field=True,
    field_cov=None,
    n_init=6,
    max_iter=120,
    tol=1e-5,
    ridge=None,
    random_state=0,
):
    """Fit a Gaussian mixture, optionally plus a fixed-covariance field.

    Parameters
    ----------
    x : ndarray, shape (n, d)
        Observed points (6D phase space or 3D position).
    S : ndarray, shape (n, d, d)
        Measurement covariance of each point.
    n_clusters : int
        Number of free Gaussian components K (not counting the field).
    include_field : bool
        If True, add a wide field component with fixed covariance
        (default 100 pc, 25 km/s in 6D).
    field_cov : ndarray, shape (d, d), optional
        Field covariance when ``include_field`` is True.
    n_init : int
        Random initialisations; the highest-likelihood run is kept.
    max_iter : int
        EM iterations per initialisation.
    tol : float
        Relative log-likelihood convergence tolerance.
    ridge : ndarray, shape (d, d), optional
        Added to every free covariance after each M-step.
    random_state : int
        RNG seed.

    Returns
    -------
    result : dict
        ``weights``, ``means``, ``covs`` (field is index 0 if present),
        ``responsibility``, ``member_prob``, ``cluster_id``,
        ``log_likelihood``, ``bic``, ``n_params``.
        ``cluster_id`` is -1 for field stars (member probability < 0.5).
    """
    x = np.asarray(x, dtype=float)
    S = np.asarray(S, dtype=float)
    n, d = x.shape
    k = int(n_clusters)
    if k < 1:
        raise ValueError("n_clusters must be >= 1")
    if n < k + 2:
        raise ValueError("Need more stars than clusters to fit the mixture")

    if ridge is None:
        ridge = _default_ridge(d)
    if include_field:
        if field_cov is None:
            field_cov = _default_field_cov(d)
        field_cov = 0.5 * (field_cov + field_cov.T)
        n_comp = k + 1
        field_index = 0
        cluster_slice = slice(1, n_comp)
        n_params = k + k * d + k * d * (d + 1) // 2
    else:
        field_cov = None
        n_comp = k
        field_index = None
        cluster_slice = slice(0, n_comp)
        n_params = (k - 1) + k * d + k * d * (d + 1) // 2

    rng = np.random.default_rng(random_state)
    scale = _scale_vector(d)
    init_dim = min(3, d)
    xw = x[:, :init_dim] / scale[:init_dim]

    best = None
    for _start in range(n_init):
        means = np.zeros((n_comp, d))
        covs = np.zeros((n_comp, d, d))
        weights = np.full(n_comp, 1.0 / n_comp)
        if include_field:
            weights[0] = 0.2
            weights[1:] = 0.8 / k
            means[0] = _precision_weighted_mean(x, S)
            covs[0] = field_cov

        _, labels = _kmeans(xw, k, rng)
        for j, lab in enumerate(range(k)):
            idx = cluster_slice.start + j
            mask = labels == lab
            if mask.sum() < 3:
                means[idx] = x.mean(axis=0) + rng.normal(0.0, scale, size=d)
                covs[idx] = np.diag(scale ** 2)
            else:
                means[idx] = x[mask].mean(axis=0)
                c = np.cov(x[mask], rowvar=False, ddof=1)
                if np.ndim(c) < 2:
                    c = np.diag(scale ** 2)
                covs[idx] = 0.5 * (c + c.T) + ridge

        log_like = -np.inf
        resp = None
        for _it in range(max_iter):
            resp, new_ll = _e_step(x, S, weights, means, covs)
            if np.isfinite(log_like) and abs(new_ll - log_like) < tol * (1.0 + abs(new_ll)):
                log_like = new_ll
                break
            log_like = new_ll

            n_eff = resp.sum(axis=0)
            weights = np.maximum(n_eff / n, 1e-3 if include_field else 1e-4)
            weights /= weights.sum()
            if include_field:
                means[0] = _precision_weighted_mean(x, S)
                covs[0] = field_cov
            for j in range(cluster_slice.start, cluster_slice.stop):
                if n_eff[j] < 1e-3:
                    means[j] = x[rng.integers(n)]
                    covs[j] = np.diag(scale ** 2)
                    continue
                means[j], covs[j] = _m_step_component(
                    x, S, covs[j], means[j], resp[:, j], ridge
                )

        resp, log_like = _e_step(x, S, weights, means, covs)
        if best is None or log_like > best["log_likelihood"]:
            best = {
                "weights": weights.copy(),
                "means": means.copy(),
                "covs": covs.copy(),
                "responsibility": resp.copy(),
                "log_likelihood": log_like,
            }

    resp = best["responsibility"]
    if include_field:
        member_prob = 1.0 - resp[:, field_index]
        cluster_id = resp[:, cluster_slice].argmax(axis=1)
        cluster_id = np.where(member_prob >= 0.5, cluster_id, -1)
    else:
        member_prob = np.ones(n)
        cluster_id = resp.argmax(axis=1)

    bic = n_params * np.log(n) - 2.0 * best["log_likelihood"]
    best.update({
        "n_clusters": k,
        "include_field": include_field,
        "member_prob": member_prob,
        "cluster_id": cluster_id,
        "n_params": n_params,
        "bic": bic,
        "aic": 2.0 * n_params - 2.0 * best["log_likelihood"],
    })
    return best


def compare_cluster_counts(x, S, k_grid=(1, 2, 3), include_field=False, **fit_kwargs):
    """Fit each ``K`` in ``k_grid`` and pick the minimum-BIC model.

    Parameters
    ----------
    x : ndarray, shape (n, d)
    S : ndarray, shape (n, d, d)
    k_grid : sequence of int
    include_field : bool
    **fit_kwargs
        Forwarded to :func:`fit_extreme_deconvolution`.

    Returns
    -------
    comparison : pandas.DataFrame
        One row per ``K``, sorted by BIC.
    best : dict
        Winning fit, with ``preferred_K`` and ``structure``.
    fits : dict
        All fits keyed by ``K``.
    """
    fits = {}
    rows = []
    for k in k_grid:
        result = fit_extreme_deconvolution(
            x, S, n_clusters=k, include_field=include_field, **fit_kwargs
        )
        fits[k] = result
        n_mem = int((result["member_prob"] >= 0.5).sum())
        rows.append({
            "K": k,
            "log_likelihood": result["log_likelihood"],
            "n_params": result["n_params"],
            "BIC": result["bic"],
            "AIC": result["aic"],
            "n_members": n_mem,
            "n_field": int(len(result["member_prob"]) - n_mem),
            "field_weight": result["weights"][0] if include_field else 0.0,
        })
    comparison = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    best_k = int(comparison.loc[0, "K"])
    best = fits[best_k]
    best["preferred_K"] = best_k
    best["structure"] = (
        "single cluster" if best_k == 1
        else "{} clusters (multi-lobe)".format(best_k)
    )
    return comparison, best, fits
