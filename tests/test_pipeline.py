"""Tests that do not need Gaia TAP or NUTS sampling."""

import numpy as np
import pandas as pd

from gaia_clustering.clustering import cluster_association, cluster_recovery, synthesize_two_lobe_association
from gaia_clustering.coords import pack_observables, observables_to_phase_space, phase_space_to_observables
from gaia_clustering.query import attach_radial_velocities
from gaia_clustering.velocity import build_velocity_model


def test_phase_space_roundtrip():
    df, _ = synthesize_two_lobe_association(n_per_lobe=8, n_field=4, rv_fraction=1.0)
    obs, _, _ = pack_observables(df, use_rv=True)
    phase, _ = observables_to_phase_space(obs)
    obs2 = phase_space_to_observables(phase)
    phase2, _ = observables_to_phase_space(obs2)
    assert np.allclose(phase, phase2, rtol=1e-5, atol=1e-3)


def test_missing_rv_is_uninformative():
    df, _ = synthesize_two_lobe_association(n_per_lobe=10, n_field=5, rv_fraction=0.0)
    obs, cov, use_rv = pack_observables(df, use_rv=False)
    assert not use_rv.any()
    assert np.allclose(cov[:, 5, 5], 1.0e6)


def test_cluster_recovers_association():
    df, _ = synthesize_two_lobe_association(
        n_per_lobe=40, n_field=25, rv_fraction=0.0, random_state=1,
    )
    fit = cluster_association(
        df, use_rv=False, k_grid=(1, 2), n_init=3, max_iter=60, random_state=1,
    )
    rec = cluster_recovery(df["true_label"], fit["best"]["member_prob"], threshold=0.5)
    tp = rec.loc["clustered as association", "true association"]
    fn = rec.loc["clustered as field", "true association"]
    recall = tp / (tp + fn)
    assert recall > 0.7
    assert fit["best"]["preferred_K"] in (1, 2)


def test_attach_radial_velocities():
    catalog = pd.DataFrame({
        "source_id": [1, 2, 3],
        "ra": [1.0, 2.0, 3.0],
    })
    rv = pd.DataFrame({
        "source_id": [2, 3],
        "radial_velocity": [-10.0, -12.0],
        "radial_velocity_error": [1.0, 1.5],
    })
    merged, n = attach_radial_velocities(catalog, rv)
    assert n == 2
    assert np.isnan(merged.loc[merged["source_id"] == 1, "radial_velocity"].iloc[0])
    assert merged.loc[merged["source_id"] == 2, "radial_velocity"].iloc[0] == -10.0


def test_velocity_model_builds_without_rv():
    df, _ = synthesize_two_lobe_association(n_per_lobe=12, n_field=0, rv_fraction=0.0)
    model, v_obs, prep = build_velocity_model(df)
    assert prep["n_vel"] == 2
    assert v_obs.shape == (len(df), 2)
    assert "Likelihood(π,μα,μδ)" in model.named_vars


def test_velocity_model_builds_with_partial_rv():
    df, _ = synthesize_two_lobe_association(n_per_lobe=12, n_field=0, rv_fraction=0.4, random_state=2)
    members = df.dropna(subset=["ra"]).copy()
    model, v_obs, prep = build_velocity_model(members)
    assert prep["n_vel"] == 3
    assert prep["n_rv"] >= 1
    assert v_obs.shape == (len(members), 3)
    assert "Likelihood(vr)" in model.named_vars
