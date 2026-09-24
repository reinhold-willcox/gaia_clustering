"""Tests that do not need Gaia TAP or NUTS sampling."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
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
    assert "Likelihood(ϖ,μα,μδ)" in model.named_vars


def test_velocity_model_builds_with_partial_rv():
    df, _ = synthesize_two_lobe_association(n_per_lobe=12, n_field=0, rv_fraction=0.4, random_state=2)
    members = df.dropna(subset=["ra"]).copy()
    model, v_obs, prep = build_velocity_model(members)
    assert prep["n_vel"] == 3
    assert prep["n_rv"] >= 1
    assert v_obs.shape == (len(members), 3)
    assert "Likelihood(vr)" in model.named_vars


def test_default_cache_dir(monkeypatch, tmp_path):
    from gaia_clustering.query import _resolve_cache_dir, default_cache_dir

    monkeypatch.delenv("GAIA_CLUSTERING_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert default_cache_dir() == Path.home() / ".cache" / "gaia_clustering"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_cache_dir() == tmp_path / "xdg" / "gaia_clustering"
    monkeypatch.setenv("GAIA_CLUSTERING_CACHE", str(tmp_path / "custom"))
    assert default_cache_dir() == tmp_path / "custom"
    assert _resolve_cache_dir(None) == tmp_path / "custom"
    assert _resolve_cache_dir(False) is None
    assert _resolve_cache_dir(tmp_path / "explicit") == tmp_path / "explicit"


def _fake_gaia_query(n=180, radius_deg=1.0, search_factor=2.0, pi0=5.0, plx_range=(2.0, 12.5)):
    from gaia_clustering.pipeline import GaiaQuery

    rng = np.random.default_rng(0)
    ra0, dec0 = 100.0, 25.0
    search = radius_deg * search_factor
    # Mix stars inside the selected cone, in the search annulus, and a few far away.
    n_in = n // 2
    n_mid = n // 3
    n_out = n - n_in - n_mid
    rho = np.concatenate([
        rng.uniform(0.0, radius_deg * 0.95, n_in),
        rng.uniform(radius_deg * 1.05, search * 0.95, n_mid),
        rng.uniform(search * 1.1, search * 1.4, n_out),
    ])
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    ra = ra0 + rho * np.cos(theta) / np.cos(np.radians(dec0))
    dec = dec0 + rho * np.sin(theta)
    plx = rng.normal(pi0, 1.2, n)
    plx[: n_in // 3] = rng.uniform(0.05, 0.3 * pi0, n_in // 3)
    eplx = rng.uniform(0.04, 0.25, n)
    extended = pd.DataFrame({
        "source_id": np.arange(n),
        "ra": ra,
        "dec": dec,
        "parallax": plx,
        "parallax_error": eplx,
        "pmra": rng.normal(0.0, 2.0, n),
        "pmdec": rng.normal(0.0, 2.0, n),
        "pmra_error": 0.1,
        "pmdec_error": 0.1,
        "ra_error": 0.1,
        "dec_error": 0.1,
        "phot_g_mean_mag": 10.0,
        "ruwe": 1.0,
    })
    info = {
        "input_name": "fake",
        "simbad_main_id": "fake",
        "simbad_ra": ra0,
        "simbad_dec": dec0,
        "center_ra": ra0,
        "center_dec": dec0,
        "kind": "star",
        "status": "ok",
        "radius_deg": float(radius_deg),
        "radius_arcmin": float(radius_deg) * 60.0,
        "position_radius_arcmin": float(radius_deg) * 60.0,
        "search_radius_deg": float(search),
        "search_radius_arcmin": float(search) * 60.0,
        "preview_factor": float(search_factor),
        "preview_radius_deg": float(search),
        "preview_radius_arcmin": float(search) * 60.0,
        "parallax_range": None if plx_range is None else tuple(plx_range),
        "parallax_window": None,
        "parallax_cuts": None if plx_range is None else tuple(plx_range),
        "reference_parallax": float(pi0),
        "reference_parallax_source": "star",
        "n_extended": int(n),
    }
    query = GaiaQuery(name="fake", query=info, catalog=extended.copy(), extended=extended)
    return query.refine_search()


def test_on_sky_separation_wraps_ra():
    from gaia_clustering.query import on_sky_separation_deg, subset_cone

    ra0, dec0 = 0.2, 10.0
    df = pd.DataFrame({
        "ra": [359.9, 0.3, 40.0],
        "dec": [10.0, 10.0, 10.0],
    })
    sep = on_sky_separation_deg(df["ra"], df["dec"], ra0, dec0)
    assert sep[0] < 0.5
    assert sep[1] < 0.5
    assert sep[2] > 30.0
    inner = subset_cone(df, ra0, dec0, 1.0)
    assert set(inner["ra"]) == {359.9, 0.3}


def test_parse_position_center_formats():
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from gaia_clustering.query import parse_position_center

    ra, dec = parse_position_center("308.11 +41.23")
    assert np.isclose(ra, 308.11) and np.isclose(dec, 41.23)
    ra2, dec2 = parse_position_center((308.11, 41.23))
    assert np.allclose([ra, dec], [ra2, dec2])
    ra3, dec3 = parse_position_center("20h32m25.8s +41d18m31s")
    expected = SkyCoord("20h32m25.8s +41d18m31s")
    assert np.isclose(ra3, expected.ra.deg) and np.isclose(dec3, expected.dec.deg)
    ra4, dec4 = parse_position_center(SkyCoord(10 * u.deg, -5 * u.deg))
    assert np.isclose(ra4, 10.0) and np.isclose(dec4, -5.0)


def test_plot_query_side_by_side():
    matplotlib.use("Agg")
    query = _fake_gaia_query()
    fig, axes = query.plot_query()
    try:
        assert len(axes) == 3
        assert fig._suptitle.get_text().startswith("fake")
        assert "selected" in fig._suptitle.get_text()
        assert query.query["n_catalog"] < query.query["n_extended"]
        assert query.query["n_catalog"] == len(query.catalog)
        from gaia_clustering.query import cone_mask, parallax_bounds_mask, resolved_parallax_bounds

        assert cone_mask(
            query.catalog, query.query["simbad_ra"], query.query["simbad_dec"],
            query.query["radius_deg"],
        ).all()
        in_cone = cone_mask(
            query.extended, query.query["simbad_ra"], query.query["simbad_dec"],
            query.query["radius_deg"],
        )
        in_plx = parallax_bounds_mask(query.extended, resolved_parallax_bounds(query.query))
        assert len(query.catalog) == int((in_cone & in_plx).sum())
        assert query.query["n_catalog"] < query.query["n_cone"]
        cuts = query.query["parallax_range"]
        constants = []
        for line in axes[2].lines:
            xd = np.asarray(line.get_xdata(), dtype=float)
            if xd.size and np.allclose(xd, xd[0]):
                constants.append(float(xd[0]))
        assert any(np.isclose(x, cuts[0]) for x in constants)
        assert any(np.isclose(x, cuts[1]) for x in constants)
        assert not any(ln.get_linestyle() in (":", "dotted") for ln in axes[2].lines)
        sky_circle = next(
            ln for ln in axes[0].lines
            if ln.get_color() == "red" and np.asarray(ln.get_xdata()).size > 2
            and ln.get_marker() in (None, "None", "")
        )
        cut_lws = [
            line.get_linewidth()
            for line in axes[2].lines
            if np.asarray(line.get_xdata()).size
            and np.allclose(np.asarray(line.get_xdata(), dtype=float), cuts[0])
        ]
        assert cut_lws
        assert np.allclose(cut_lws[0], sky_circle.get_linewidth())
        assert axes[0].get_legend() is None
        assert axes[1].get_legend() is None
        assert axes[2].get_legend() is None
        assert fig.legends
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        assert labels == [
            "consistent position",
            "consistent proper motion",
            "consistent parallax",
            "accepted stars",
            "2-cone KDE",
            "boundary values",
        ]
        assert len(axes[2].patches) == 0
        kde_lines = [
            ln for ln in axes[2].lines
            if np.asarray(ln.get_xdata()).size > 2 and ln.get_color() == "black"
        ]
        assert len(kde_lines) == 1
        y0, y1 = axes[2].get_ylim()
        kde_peak = float(np.nanmax(np.asarray(kde_lines[0].get_ydata(), dtype=float)))
        assert np.isclose(kde_peak, 0.9 * (y1 - y0), rtol=1e-5)
        sky_box = "".join(t.get_text() for t in axes[0].texts)
        assert "Center: (100.00, +25.00), radius: 60'" in sky_box
        assert "\n" not in sky_box
        assert "reduction factor" not in sky_box
        assert "selected" not in axes[0].get_title()
        pm_texts = " ".join(t.get_text() for t in axes[1].texts)
        assert "Center:" in pm_texts
        assert "(" in pm_texts and ")" in pm_texts
        assert any(c.get_offsets().shape[0] > 0 for c in axes[1].collections)
        x_marks = [
            c for c in axes[0].collections
            if c.get_zorder() >= 7 and c.get_offsets().shape[0] == 1
        ]
        assert x_marks
    finally:
        plt.close(fig)


def test_plot_query_alpha():
    matplotlib.use("Agg")
    query = _fake_gaia_query(n=40)
    fig, axes = query.plot_query(parallax_alpha=0.2)
    try:
        assert len(axes[2].collections) == 0
        gauss = [
            ln.get_alpha() for ln in axes[2].lines
            if ln.get_alpha() is not None and np.asarray(ln.get_xdata()).size > 2
        ]
        assert gauss and all(np.isclose(a, 0.2) for a in gauss)
    finally:
        plt.close(fig)


def test_gaussian_kde_fixed_uses_constant_bandwidth():
    from gaia_clustering.plots import _gaussian_kde_fixed

    xs = np.linspace(-6.0, 6.0, 1201)
    samples = np.array([0.0, 0.0, 0.0])
    dens = _gaussian_kde_fixed(xs, samples, 1.0)
    assert dens is not None
    assert np.isclose(dens[np.argmin(np.abs(xs))], 1.0 / np.sqrt(2.0 * np.pi), rtol=1e-3)
    assert np.isclose(np.trapezoid(dens, xs), 1.0, atol=0.02)


def test_plot_query_kde_bandwidth():
    matplotlib.use("Agg")
    query = _fake_gaia_query(n=40)

    def _kde_y(bandwidth):
        fig, axes = query.plot_query(kde_bandwidth=bandwidth)
        try:
            kde_lines = [
                ln for ln in axes[2].lines
                if np.asarray(ln.get_xdata()).size > 2 and ln.get_color() == "black"
            ]
            assert len(kde_lines) == 1
            return np.asarray(kde_lines[0].get_ydata(), dtype=float)
        finally:
            plt.close(fig)

    y_narrow = _kde_y(0.05)
    y_wide = _kde_y(2.0)
    assert not np.allclose(y_narrow, y_wide)
    try:
        query.plot_query(kde_bandwidth=0.0)
        raise AssertionError("expected kde_bandwidth error")
    except ValueError as exc:
        assert "kde_bandwidth" in str(exc)


def test_plot_query_kde_uses_sky_and_pm_stars():
    matplotlib.use("Agg")
    from gaia_clustering.pipeline import GaiaQuery
    from gaia_clustering.plots import _gaussian_kde_fixed

    ra0, dec0 = 100.0, 25.0
    n_sky_pm, n_sky_only = 20, 20
    extended = pd.DataFrame({
        "source_id": np.arange(n_sky_pm + n_sky_only),
        "ra": ra0,
        "dec": dec0,
        "parallax": np.concatenate([
            np.full(n_sky_pm, 8.0),
            np.full(n_sky_only, 1.0),
        ]),
        "parallax_error": 0.05,
        "pmra": np.concatenate([
            np.zeros(n_sky_pm),
            np.full(n_sky_only, 10.0),
        ]),
        "pmdec": 0.0,
        "pmra_error": 0.1,
        "pmdec_error": 0.1,
        "ra_error": 0.1,
        "dec_error": 0.1,
        "phot_g_mean_mag": 10.0,
        "ruwe": 1.0,
    })
    info = {
        "input_name": "fake",
        "simbad_main_id": "fake",
        "simbad_ra": ra0,
        "simbad_dec": dec0,
        "center_ra": ra0,
        "center_dec": dec0,
        "kind": "star",
        "status": "ok",
        "radius_deg": 1.0,
        "position_radius_arcmin": 60.0,
        "search_radius_arcmin": 60.0,
        "pm_center": (0.0, 0.0),
        "pm_radius": 2.0,
        "parallax_range": None,
        "n_extended": len(extended),
    }
    query = GaiaQuery(name="fake", query=info, catalog=extended.copy(), extended=extended)
    query.refine_search()
    fig, axes = query.plot_query(parallax_range=(0.0, 12.0), kde_bandwidth=0.2)
    try:
        kde_lines = [
            ln for ln in axes[2].lines
            if np.asarray(ln.get_xdata()).size > 2 and ln.get_color() == "black"
        ]
        assert len(kde_lines) == 1
        xs = np.asarray(kde_lines[0].get_xdata(), dtype=float)
        ys = np.asarray(kde_lines[0].get_ydata(), dtype=float)
        peak_x = float(xs[np.nanargmax(ys)])
        assert peak_x > 6.0
        expected = _gaussian_kde_fixed(xs, np.full(n_sky_pm, 8.0), 0.2)
        expected = expected * (0.9 * (axes[2].get_ylim()[1] - axes[2].get_ylim()[0])) / np.nanmax(expected)
        assert np.allclose(ys, expected, rtol=1e-3, atol=1e-3)
    finally:
        plt.close(fig)


def test_gaia_query_refine_search_changes_catalog():
    query = _fake_gaia_query()
    n0 = len(query.catalog)
    query.refine_search(position_radius_arcmin=36.0)
    n_small = len(query.catalog)
    assert n_small < n0
    query.refine_search(parallax_range=None)
    assert len(query.catalog) >= n_small
    try:
        query.refine_search(position_radius_arcmin=540.0)
        raise AssertionError("expected downloaded-cone error")
    except ValueError as exc:
        assert "downloaded cone" in str(exc)


def test_reset_search_undoes_refine():
    query = _fake_gaia_query()
    n0 = len(query.catalog)
    ra0, dec0 = query.query["center_ra"], query.query["center_dec"]
    radius0 = query.query["position_radius_arcmin"]
    plx0 = query.query.get("parallax_range")
    query.refine_search(
        position_center="101.5 +26.0",
        position_radius_arcmin=36.0,
        pm_center=(0.0, 0.0),
        pm_radius=1.5,
        parallax_range=(4.0, 6.0),
    )
    assert len(query.catalog) < n0
    assert query.query["pm_radius"] == 1.5
    query.reset_search()
    assert len(query.catalog) == n0
    assert np.isclose(query.query["center_ra"], ra0)
    assert np.isclose(query.query["center_dec"], dec0)
    assert np.isclose(query.query["position_radius_arcmin"], radius0)
    assert query.query.get("parallax_range") == plx0
    assert query.query.get("pm_radius") in (None, 0)
    query.reset_search()
    assert len(query.catalog) == n0


def test_refine_search_then_plot_query_draws_cuts():
    matplotlib.use("Agg")
    query = _fake_gaia_query()
    n0 = len(query.catalog)
    query.refine_search(
        pm_center=(0.0, 0.0),
        pm_radius=1.5,
        parallax_range=(4.0, 6.0),
    )
    fig, axes = query.plot_query()
    try:
        assert len(query.catalog) <= n0
        from gaia_clustering.query import proper_motion_mask

        assert proper_motion_mask(query.catalog, query.query).all()
        assert np.all(query.catalog["parallax"] >= 4.0)
        assert np.all(query.catalog["parallax"] <= 6.0)
        assert query.query["pm_center"] == (0.0, 0.0)
        assert query.query["pm_radius"] == 1.5
        assert any(np.asarray(ln.get_xdata()).size > 2 for ln in axes[1].lines)
        pm_box = "".join(t.get_text() for t in axes[1].texts)
        assert "Center: (0.00, +0.00), radius: 1.5" in pm_box
        assert "\n" not in pm_box
        constants = []
        for line in axes[2].lines:
            xd = np.asarray(line.get_xdata(), dtype=float)
            if xd.size and np.allclose(xd, xd[0]):
                constants.append(float(xd[0]))
        assert any(np.isclose(x, 4.0) for x in constants)
        assert any(np.isclose(x, 6.0) for x in constants)
    finally:
        plt.close(fig)


def test_plot_query_criterion_colors():
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba
    from gaia_clustering.pipeline import GaiaQuery
    from gaia_clustering.plots import _QUERY_CUT_COLORS

    ra0, dec0 = 100.0, 25.0
    stars = [
        # sky, pm, plx
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ]
    ra, dec, pmra, pmdec, plx = [], [], [], [], []
    for in_sky, in_pm, in_plx in stars:
        ra.append(ra0 if in_sky else ra0 + 3.0)
        dec.append(dec0)
        pmra.append(0.0 if in_pm else 10.0)
        pmdec.append(0.0)
        plx.append(5.0 if in_plx else 0.2)
    n = len(stars)
    extended = pd.DataFrame({
        "source_id": np.arange(n),
        "ra": ra,
        "dec": dec,
        "parallax": plx,
        "parallax_error": 0.1,
        "pmra": pmra,
        "pmdec": pmdec,
        "pmra_error": 0.1,
        "pmdec_error": 0.1,
        "ra_error": 0.1,
        "dec_error": 0.1,
        "phot_g_mean_mag": 10.0,
        "ruwe": 1.0,
    })
    info = {
        "input_name": "fake",
        "simbad_main_id": "fake",
        "simbad_ra": ra0,
        "simbad_dec": dec0,
        "center_ra": ra0,
        "center_dec": dec0,
        "kind": "star",
        "status": "ok",
        "radius_deg": 1.0,
        "radius_arcmin": 60.0,
        "position_radius_arcmin": 60.0,
        "search_radius_deg": 4.0,
        "search_radius_arcmin": 240.0,
        "preview_factor": 2.0,
        "preview_radius_deg": 4.0,
        "preview_radius_arcmin": 240.0,
        "parallax_range": (4.0, 6.0),
        "parallax_window": None,
        "parallax_cuts": (4.0, 6.0),
        "reference_parallax": 5.0,
        "reference_parallax_source": "star",
        "pm_center": (0.0, 0.0),
        "pm_radius": 2.0,
        "n_extended": n,
    }
    query = GaiaQuery(name="fake", query=info, catalog=extended.copy(), extended=extended)
    query.refine_search()
    fig, axes = query.plot_query()
    try:
        keys = ("all", "sky_pm", "sky_plx", "pm_plx", "sky", "pm", "plx", "none")
        expected = {to_rgba(_QUERY_CUT_COLORS[k])[:3] for k in keys}
        found = set()
        for ax in (axes[0], axes[1]):
            for c in ax.collections:
                if c.get_zorder() >= 7:
                    continue
                fc = np.asarray(c.get_facecolors())
                if fc.size == 0:
                    continue
                for rgba in np.atleast_2d(fc):
                    found.add(tuple(np.round(to_rgba(rgba)[:3], 5)))
        expected_r = {tuple(np.round(c, 5)) for c in expected}
        assert expected_r <= found
        gauss = set()
        for ln in axes[2].lines:
            if ln.get_alpha() is None or np.asarray(ln.get_xdata()).size <= 2:
                continue
            gauss.add(tuple(np.round(to_rgba(ln.get_color())[:3], 5)))
        assert gauss <= expected_r
        assert len(query.catalog) == 1
    finally:
        plt.close(fig)


def test_association_uses_cone_mean_parallax():
    from gaia_clustering.query import apply_query_cuts, cone_mask

    query = _fake_gaia_query()
    info = dict(query.query)
    info["kind"] = "group"
    info["reference_parallax"] = None
    info["reference_parallax_source"] = None
    info["parallax_range"] = None
    info["parallax_cuts"] = None
    info["parallax_window"] = (0.5, 2.0)
    catalog = apply_query_cuts(query.extended, info)
    in_cone = query.extended.loc[cone_mask(
        query.extended, info["simbad_ra"], info["simbad_dec"], info["radius_deg"],
    )]
    expected = float(np.mean(in_cone.loc[in_cone["parallax"] > 0, "parallax"]))
    assert info["reference_parallax_source"] == "cone_mean"
    assert np.isclose(info["reference_parallax"], expected)
    lo, hi = 0.5 * expected, 2.0 * expected
    assert np.all((catalog["parallax"] >= lo) & (catalog["parallax"] <= hi))


def test_select_parallax_range_and_plot_axis_ranges():
    matplotlib.use("Agg")
    query = _fake_gaia_query()
    query.refine_search(parallax_range=(2.0, 4.0))
    assert query.query["parallax_lo_mas"] == 2.0
    assert query.query["parallax_hi_mas"] == 4.0
    assert np.all(query.catalog["parallax"] >= 2.0)
    assert np.all(query.catalog["parallax"] <= 4.0)
    fig, axes = query.plot_query(
        ra_range=(99.8, 100.2),
        dec_range=(24.9, 25.1),
        pmra_range=(-1.0, 1.0),
        pmdec_range=(-1.5, 0.5),
        parallax_range=(0.5, 6.0),
        density_range=(0.0, 3.0),
    )
    try:
        assert np.allclose(axes[0].get_xlim(), (100.2, 99.8))
        assert np.allclose(axes[0].get_ylim(), (24.9, 25.1))
        assert np.allclose(axes[1].get_xlim(), (-1.0, 1.0))
        assert np.allclose(axes[1].get_ylim(), (-1.5, 0.5))
        assert np.allclose(axes[2].get_xlim(), (0.5, 6.0))
        assert np.allclose(axes[2].get_ylim(), (0.0, 3.0))
        fig.canvas.draw()
        widths = [ax.get_position().width for ax in axes]
        heights = [ax.get_position().height for ax in axes]
        assert np.allclose(widths, widths[0], rtol=0.05)
        assert np.allclose(heights, heights[0], rtol=0.05)
        constants = []
        for line in axes[2].lines:
            xd = np.asarray(line.get_xdata(), dtype=float)
            if xd.size and np.allclose(xd, xd[0]):
                constants.append(float(xd[0]))
        assert any(np.isclose(x, 2.0) for x in constants)
        assert any(np.isclose(x, 4.0) for x in constants)
    finally:
        plt.close(fig)


def test_plot_bic_comparison_is_relative_to_minimum():
    matplotlib.use("Agg")
    from gaia_clustering.plots import plot_bic_comparison

    comparison = pd.DataFrame({
        "K": [2, 1, 3],
        "BIC": [10.0, 15.0, 22.0],
    })
    ax = plot_bic_comparison(comparison)
    try:
        heights = [float(p.get_height()) for p in ax.patches]
        assert np.allclose(sorted(heights), [0.0, 5.0, 12.0])
        labels = [t.get_text() for t in ax.texts]
        assert "0.00" in labels
        assert "5.00" in labels
        assert "12.00" in labels
        assert ax.get_ylabel() == r"BIC $-$ BIC$_{\rm min}$"
    finally:
        plt.close(ax.figure)


def test_lobe_hsv_cmaps():
    from matplotlib.colors import rgb_to_hsv
    from gaia_clustering.plots import _lobe_display_cmap, _lobe_hues

    hues3 = _lobe_hues(3)
    assert np.allclose(hues3, (2.0 / 3.0, 0.0, 1.0 / 6.0))
    hues6 = _lobe_hues(6)
    assert len(hues6) == 6
    assert np.allclose(np.diff(hues6), hues6[1] - hues6[0])
    cm = _lobe_display_cmap(0, 3)
    hsv1 = rgb_to_hsv(np.asarray(cm(1.0)[:3], dtype=float))
    hsv0 = rgb_to_hsv(np.asarray(cm(0.0)[:3], dtype=float))
    assert np.isclose(hsv1[0], hues3[0], atol=0.03)
    assert np.isclose(hsv1[1], 1.0, atol=0.05)
    assert np.isclose(hsv1[2], 0.5, atol=0.05)
    assert np.isclose(hsv0[1], 0.25, atol=0.05)
    assert np.isclose(hsv0[2], 1.0, atol=0.05)
    assert np.isclose(hsv0[0], hsv1[0], atol=0.05)


def _fake_membership_fit(n_lobes=2, n_per=8, n_field=6):
    rng = np.random.default_rng(4)
    n_mem = n_lobes * n_per
    n = n_mem + n_field
    cluster_id = np.concatenate([
        np.repeat(np.arange(n_lobes), n_per),
        np.full(n_field, -1),
    ])
    member_prob = np.concatenate([
        rng.uniform(0.7, 1.0, n_mem),
        rng.uniform(0.0, 0.4, n_field),
    ])
    df = pd.DataFrame({
        "ra": rng.normal(100.0, 0.2, n),
        "dec": rng.normal(25.0, 0.2, n),
        "parallax": rng.uniform(2.0, 4.0, n),
        "parallax_error": 0.05,
        "pmra": rng.normal(0.0, 1.0, n),
        "pmdec": rng.normal(-3.0, 1.0, n),
        "pmra_error": 0.1,
        "pmdec_error": 0.1,
        "ra_error": 0.1,
        "dec_error": 0.1,
        "phot_g_mean_mag": 12.0,
    })
    best = {
        "member_prob": member_prob,
        "cluster_id": cluster_id,
        "n_clusters": n_lobes,
        "preferred_K": n_lobes,
    }
    comparison = pd.DataFrame({
        "K": [1, 2, 3][: max(n_lobes, 1)],
        "BIC": np.arange(max(n_lobes, 1), dtype=float) + 10.0,
    })
    return df, {"best": best, "comparison": comparison}


def test_plot_summary_multi_lobe_colorbars():
    matplotlib.use("Agg")
    from gaia_clustering.plots import plot_membership_summary, _membership_colors, _lobe_display_cmap

    df, fit = _fake_membership_fit(n_lobes=2)
    fig, axes = plot_membership_summary(df, fit)
    try:
        assert len(axes) == 4
        assert r"$\varpi$ [mas]" in axes[2].get_xlabel()
        fig.canvas.draw()
        cb_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
        assert len(cb_axes) == 2
        cb_axes = sorted(cb_axes, key=lambda a: a.get_position().x0)
        assert [ax.get_title() for ax in cb_axes] == ["1", "2"]
        assert cb_axes[0].get_ylabel() == ""
        assert r"P(\mathrm{member})" in cb_axes[-1].get_ylabel()
        assert all(t.get_text() == "" for t in cb_axes[0].get_yticklabels())
        assert any(t.get_text() for t in cb_axes[-1].get_yticklabels())
        assert axes[3].yaxis.get_label_position() == "right"
        colors = _membership_colors(fit["best"])
        ids = fit["best"]["cluster_id"]
        p = fit["best"]["member_prob"]
        for k in (0, 1):
            mask = ids == k
            expected = _lobe_display_cmap(k, 2)(p[mask])
            assert np.allclose(colors[mask], expected, atol=1e-6)
    finally:
        plt.close(fig)


def test_corner_and_3d_multi_lobe_colorbars():
    matplotlib.use("Agg")
    from gaia_clustering.plots import plot_association_corner, plot_association_3d

    df, fit = _fake_membership_fit(n_lobes=3)
    fig = plot_association_corner(df, fit["best"])
    try:
        fig.canvas.draw()
        cb_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
        assert len(cb_axes) == 3
        cb_axes = sorted(cb_axes, key=lambda a: a.get_position().x0)
        assert [ax.get_title() for ax in cb_axes] == ["1", "2", "3"]
        assert cb_axes[0].get_ylabel() == ""
        assert r"P(\mathrm{member})" in cb_axes[-1].get_ylabel()
    finally:
        plt.close(fig)
    fig3 = plot_association_3d(df, fit["best"], traceback=None, show_vectors=False)
    scaled = [
        t for t in fig3.data
        if t.name.startswith("lobe") and t.marker.showscale
    ]
    assert len(scaled) == 3
    titles = [t.marker.colorbar.title.text for t in scaled]
    assert titles == ["1", "2", "3"]
    assert [t.marker.colorbar.showticklabels for t in scaled] == [False, False, True]


def test_plot_summary_single_lobe_one_colorbar():
    matplotlib.use("Agg")
    from gaia_clustering.plots import plot_membership_summary

    df, fit = _fake_membership_fit(n_lobes=1)
    fig, _ = plot_membership_summary(df, fit)
    try:
        cb_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
        assert len(cb_axes) == 1
        fig.canvas.draw()
        assert cb_axes[0].get_title() == "1"
        assert r"P(\mathrm{member})" in cb_axes[0].get_ylabel()
    finally:
        plt.close(fig)


def test_analyze_association_accepts_gaia_query():
    df, _ = synthesize_two_lobe_association(
        n_per_lobe=16, n_field=8, rv_fraction=0.0, random_state=3,
    )
    from gaia_clustering.pipeline import GaiaQuery, analyze_association

    query = GaiaQuery(
        name="synthetic",
        query={
            "kind": "group",
            "status": "catalog",
            "radius_deg": 1.0,
            "radius_arcmin": 60.0,
            "position_radius_arcmin": 60.0,
            "preview_radius_deg": 2.0,
            "preview_radius_arcmin": 120.0,
            "n_extended": len(df),
        },
        catalog=df,
        extended=df,
    )
    result = analyze_association(
        query, infer_velocity=False, n_init=2, max_iter=40, random_state=3,
    )
    assert result.name == "synthetic"
    assert len(result.catalog) == len(df)
    assert result.n_members + result.n_field == len(df)
    assert result.velocity_cluster_id is None


def test_velocity_cluster_id_required_for_multi_lobe():
    from gaia_clustering.pipeline import _select_velocity_members

    df, _ = synthesize_two_lobe_association(
        n_per_lobe=20, n_field=10, rv_fraction=0.0, random_state=4,
    )
    fit = cluster_association(
        df, use_rv=False, k_grid=(1, 2), n_init=3, max_iter=60, random_state=4,
    )
    table = fit["table"]
    preferred_K = int(fit["best"]["preferred_K"])
    if preferred_K <= 1:
        members, cid = _select_velocity_members(table, preferred_K, None)
        assert len(members) >= 3
        assert cid in (0, None)
        return
    try:
        _select_velocity_members(table, preferred_K, None)
        raise AssertionError("expected ValueError without velocity_cluster_id")
    except ValueError as exc:
        assert "velocity_cluster_id" in str(exc)
    members, cid = _select_velocity_members(table, preferred_K, 0)
    assert cid == 0
    assert np.all(np.asarray(members["cluster_id"]) == 0)
    assert np.all(members["is_member"])
    try:
        _select_velocity_members(table, preferred_K, 99)
        raise AssertionError("expected ValueError for unknown lobe")
    except ValueError as exc:
        assert "99" in str(exc)


def test_analyze_association_multi_lobe_requires_velocity_cluster_id():
    df, _ = synthesize_two_lobe_association(
        n_per_lobe=40, n_field=20, rv_fraction=0.0, random_state=5,
    )
    from gaia_clustering.pipeline import analyze_association

    result = analyze_association(
        name="synthetic",
        catalog=df,
        infer_velocity=False,
        n_init=3,
        max_iter=60,
        random_state=5,
    )
    preferred_K = int(result.membership["best"]["preferred_K"])
    if preferred_K <= 1:
        return
    try:
        analyze_association(
            name="synthetic",
            catalog=df,
            infer_velocity=True,
            velocity_cluster_id=None,
            n_init=3,
            max_iter=60,
            random_state=5,
        )
        raise AssertionError("expected ValueError when preferred_K > 1")
    except ValueError as exc:
        assert "velocity_cluster_id" in str(exc)

    # Selection path without NUTS: resolved lobe matches the request.
    from gaia_clustering.pipeline import _select_velocity_members

    members, cid = _select_velocity_members(result.table, preferred_K, 0)
    assert cid == 0
    assert len(members) >= 3
    assert np.all(members["cluster_id"] == 0)


def test_query_gaia_cone_skips_optional_cuts(monkeypatch):
    from gaia_clustering.query import query_gaia_cone

    captured = {}

    def fake_tap(adql, cache_dir=None):
        captured["adql"] = adql
        return pd.DataFrame(columns=["source_id"])

    monkeypatch.setattr("gaia_clustering.query.gaia_tap_query", fake_tap)
    query_gaia_cone(10.0, 20.0)
    adql = captured["adql"]
    assert "phot_g_mean_mag < 20" in adql
    assert "ruwe <" not in adql
    assert "parallax_over_error >" not in adql
    assert "parallax >=" not in adql
    assert "parallax <=" not in adql
    assert "CIRCLE('ICRS', 10.0, 20.0, {}".format(10.0 / 60.0) in adql

    query_gaia_cone(10.0, 20.0, ruwe_max=1.4, plx_snr_min=5.0, g_max=12)
    adql = captured["adql"]
    assert "ruwe < 1.4" in adql
    assert "parallax_over_error > 5.0" in adql
    assert "phot_g_mean_mag < 12" in adql
    query_gaia_cone(10.0, 20.0, parallax_lo=1.5, parallax_hi=12.0)
    adql = captured["adql"]
    assert "parallax >= 1.5" in adql
    assert "parallax <= 12.0" in adql


def test_unrecognized_name_raises_name_resolution_error(monkeypatch):
    from gaia_clustering.query import NameResolutionError, fetch_neighbourhood

    monkeypatch.setattr(
        "gaia_clustering.query.resolve_name",
        lambda name: {
            "input_name": name,
            "simbad_main_id": None,
            "simbad_ra": float("nan"),
            "simbad_dec": float("nan"),
            "otype": None,
            "kind": "unknown",
            "gaia_source_id": None,
            "status": "unrecognized",
        },
    )
    try:
        fetch_neighbourhood("not-a-real-star", cache_dir=False)
        raise AssertionError("expected NameResolutionError")
    except NameResolutionError as exc:
        assert "not-a-real-star" in str(exc)
        assert "SIMBAD" in str(exc)


def test_gaia_tap_timeout_is_wrapped(monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError

    from gaia_clustering.query import GaiaQueryError, gaia_tap_query

    def fake_urlopen(request, timeout=None):
        raise HTTPError(
            "https://gea.esac.esa.int/tap-server/tap/sync",
            408,
            "408",
            hdrs={},
            fp=BytesIO(),
        )

    monkeypatch.setattr("gaia_clustering.query.urlopen", fake_urlopen)
    try:
        gaia_tap_query("SELECT TOP 1 source_id FROM gaiadr3.gaia_source", cache_dir=None)
        raise AssertionError("expected GaiaQueryError")
    except GaiaQueryError as exc:
        assert "timed out" in str(exc)
        assert "HTTP 408" in str(exc)


def _minimal_cone_table(n, radius_arcmin):
    n = int(n)
    return pd.DataFrame({
        "source_id": np.arange(n) + int(round(radius_arcmin * 1000)),
        "ra": 100.0 + np.linspace(-0.01, 0.01, n),
        "dec": np.full(n, 25.0),
        "parallax": np.full(n, 5.0),
        "parallax_error": np.full(n, 0.1),
    })


def _patch_name_resolve(monkeypatch):
    monkeypatch.setattr(
        "gaia_clustering.query.resolve_name",
        lambda name: {
            "input_name": name,
            "simbad_main_id": name,
            "simbad_ra": 100.0,
            "simbad_dec": 25.0,
            "otype": "*",
            "kind": "star",
            "gaia_source_id": None,
            "status": "no_gaia_id",
        },
    )


def test_query_search_radius_matches_request(monkeypatch):
    from gaia_clustering.query import fetch_neighbourhood

    _patch_name_resolve(monkeypatch)
    radii = []

    def fake_cone(ra, dec, radius_arcmin=10, **kwargs):
        radii.append(float(radius_arcmin))
        return _minimal_cone_table(40, radius_arcmin)

    monkeypatch.setattr("gaia_clustering.query.query_gaia_cone", fake_cone)
    _, info, _ = fetch_neighbourhood(
        "src", position_radius_arcmin=10, cache_dir=False,
        return_extended=True,
    )
    assert radii == [10.0]
    assert np.isclose(info["search_radius_arcmin"], 10.0)
    assert np.isclose(info["position_radius_arcmin"], 10.0)


def test_refine_search_custom_center():
    query = _fake_gaia_query()
    query.refine_search(position_center="101.5 +26.0")
    assert np.isclose(query.query["center_ra"], 101.5)
    assert np.isclose(query.query["center_dec"], 26.0)
    assert np.isclose(query.query["simbad_ra"], 100.0)
    assert np.isclose(query.query["simbad_dec"], 25.0)


def test_plot_query_red_x_is_name_position():
    matplotlib.use("Agg")
    query = _fake_gaia_query()
    query.query["center_ra"] = 100.3
    query.query["center_dec"] = 25.1
    query.query["simbad_ra"] = 100.0
    query.query["simbad_dec"] = 25.0
    fig, axes = query.plot_query()
    try:
        x_marks = [
            c.get_offsets() for c in axes[0].collections
            if c.get_zorder() >= 7 and c.get_offsets().shape[0] == 1
        ]
        assert len(x_marks) == 1
        xy = np.asarray(x_marks[0][0], dtype=float)
        assert np.isclose(xy[0], 100.0, atol=0.05)
        assert np.isclose(xy[1], 25.0, atol=0.05)
        texts = " ".join(t.get_text() for t in axes[0].texts)
        assert "Center: (100.30, +25.10), radius:" in texts
        assert "\n" not in "".join(t.get_text() for t in axes[0].texts)
    finally:
        plt.close(fig)


def test_plot_query_draws_all_stars():
    matplotlib.use("Agg")
    query = _fake_gaia_query(n=400)
    fig, axes = query.plot_query(random_state=1)
    try:
        n_pts = sum(c.get_offsets().shape[0] for c in axes[0].collections)
        assert n_pts == 401  # all stars plus the default-center X
        assert "reduction factor" not in " ".join(t.get_text() for t in axes[0].texts)
    finally:
        plt.close(fig)


def test_magnitude_marker_sizes_linear_in_g():
    from gaia_clustering.plots import _magnitude_marker_sizes

    mag = np.array([8.0, 12.0, 16.0])
    s = _magnitude_marker_sizes(mag, ms=10.0, n=3, ms_min_frac=0.3, ms_floor=2.0)
    d = np.sqrt(s)
    assert np.isclose(d[0], 10.0)
    assert np.isclose(d[2], 3.0)
    assert np.isclose(d[1], 6.5)
    same = _magnitude_marker_sizes(np.array([11.0, 11.0]), ms=8.0, n=2)
    assert np.allclose(same, 8.0 ** 2)
    missing = _magnitude_marker_sizes(np.array([np.nan, 10.0, 20.0]), ms=10.0, n=3)
    assert np.isclose(np.sqrt(missing[0]), 3.0)
    assert np.isclose(np.sqrt(missing[1]), 10.0)
    assert np.isclose(np.sqrt(missing[2]), 3.0)


def test_plot_query_brighter_stars_are_larger():
    matplotlib.use("Agg")
    from gaia_clustering.pipeline import GaiaQuery

    ra0, dec0 = 100.0, 25.0
    mag = np.array([8.0, 12.0, 16.0])
    extended = pd.DataFrame({
        "source_id": np.arange(3),
        "ra": ra0,
        "dec": [dec0, dec0 + 0.1, dec0 + 0.2],
        "parallax": 5.0,
        "parallax_error": 0.1,
        "pmra": 0.0,
        "pmdec": 0.0,
        "pmra_error": 0.1,
        "pmdec_error": 0.1,
        "ra_error": 0.1,
        "dec_error": 0.1,
        "phot_g_mean_mag": mag,
        "ruwe": 1.0,
    })
    info = {
        "input_name": "fake",
        "simbad_main_id": "fake",
        "simbad_ra": ra0,
        "simbad_dec": dec0,
        "center_ra": ra0,
        "center_dec": dec0,
        "kind": "star",
        "status": "ok",
        "radius_deg": 1.0,
        "position_radius_arcmin": 60.0,
        "search_radius_arcmin": 60.0,
        "n_extended": 3,
    }
    query = GaiaQuery(name="fake", query=info, catalog=extended.copy(), extended=extended)
    fig, axes = query.plot_query(ms=10)
    try:
        stars = [
            c for c in axes[0].collections
            if c.get_zorder() < 7 and c.get_offsets().shape[0] > 0
        ]
        assert stars
        decs = np.concatenate([c.get_offsets()[:, 1] for c in stars])
        sizes = np.concatenate([c.get_sizes() for c in stars])
        order = np.argsort(decs)
        d = np.sqrt(sizes[order])
        assert np.isclose(d[0], 10.0)
        assert d[0] > d[1] > d[2]
        assert d[2] >= 2.0
    finally:
        plt.close(fig)


def test_preview_parallax_bounds_widen_accepted_cuts():
    from gaia_clustering.query import preview_parallax_bounds

    assert preview_parallax_bounds((2.0, 8.0)) == (1.5, 12.0)
    assert preview_parallax_bounds((5.0, 20.0)) == (3.75, 30.0)
    assert preview_parallax_bounds(None) is None


def test_tap_uses_parallax_range_as_requested(monkeypatch):
    from gaia_clustering.query import fetch_neighbourhood

    _patch_name_resolve(monkeypatch)
    captured = {}

    def fake_cone(ra, dec, radius_arcmin=10, parallax_lo=None, parallax_hi=None, **kwargs):
        captured["lo"] = parallax_lo
        captured["hi"] = parallax_hi
        return _minimal_cone_table(20, radius_arcmin)

    monkeypatch.setattr("gaia_clustering.query.query_gaia_cone", fake_cone)
    _, info, _ = fetch_neighbourhood(
        "src", position_radius_arcmin=10, cache_dir=False,
        return_extended=True, parallax_range=(2.0, 8.0),
    )
    assert captured["lo"] == 2.0
    assert captured["hi"] == 8.0
    assert info["parallax_lo_mas"] == 2.0
    assert info["parallax_hi_mas"] == 8.0


def test_tap_without_parallax_range_has_no_plx_filter(monkeypatch):
    from gaia_clustering.query import fetch_neighbourhood

    monkeypatch.setattr(
        "gaia_clustering.query.resolve_name",
        lambda name: {
            "input_name": name,
            "simbad_main_id": name,
            "simbad_ra": 100.0,
            "simbad_dec": 25.0,
            "otype": "*",
            "kind": "star",
            "gaia_source_id": 99,
            "status": "ok",
        },
    )
    seed = _minimal_cone_table(1, 10)
    seed["source_id"] = [99]
    seed["parallax"] = [10.0]
    monkeypatch.setattr(
        "gaia_clustering.query.query_gaia_by_source_ids",
        lambda *a, **k: seed,
    )
    captured = {}

    def fake_cone(ra, dec, radius_arcmin=10, parallax_lo=None, parallax_hi=None, **kwargs):
        captured["lo"] = parallax_lo
        captured["hi"] = parallax_hi
        return _minimal_cone_table(20, radius_arcmin)

    monkeypatch.setattr("gaia_clustering.query.query_gaia_cone", fake_cone)
    _, info, _ = fetch_neighbourhood(
        "star", position_radius_arcmin=10, cache_dir=False,
        return_extended=True,
    )
    assert captured["lo"] is None
    assert captured["hi"] is None
    assert info["parallax_range"] is None


def test_group_download_keeps_full_cone(monkeypatch):
    from gaia_clustering.query import fetch_neighbourhood

    monkeypatch.setattr(
        "gaia_clustering.query.resolve_name",
        lambda name: {
            "input_name": name,
            "simbad_main_id": name,
            "simbad_ra": 100.0,
            "simbad_dec": 25.0,
            "otype": "As*",
            "kind": "group",
            "gaia_source_id": None,
            "status": "no_gaia_id",
        },
    )
    table = _minimal_cone_table(5, 10)
    table["parallax"] = [1.0, 10.0, 10.0, 10.0, 100.0]

    def fake_cone(ra, dec, radius_arcmin=10, parallax_lo=None, parallax_hi=None, **kwargs):
        assert parallax_lo is None and parallax_hi is None
        return table.copy()

    monkeypatch.setattr("gaia_clustering.query.query_gaia_cone", fake_cone)
    catalog, info, extended = fetch_neighbourhood(
        "assoc", position_radius_arcmin=10, cache_dir=False,
        return_extended=True,
    )
    plx = np.sort(extended["parallax"].to_numpy())
    assert np.allclose(plx, [1.0, 10.0, 10.0, 10.0, 100.0])
    assert len(catalog) == len(extended)
    assert info["parallax_range"] is None
