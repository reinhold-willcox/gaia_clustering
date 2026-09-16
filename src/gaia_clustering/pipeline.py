"""End-to-end association analysis: query → membership → velocity dispersion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gaia_clustering.clustering import cluster_association
from gaia_clustering.plots import (
    plot_association_3d,
    plot_association_corner,
    plot_membership_summary,
    plot_velocity_corner,
    plot_velocity_gaussian,
)
from gaia_clustering.query import (
    DEFAULT_CACHE_DIR,
    attach_radial_velocities,
    fetch_neighbourhood,
    load_rv_table,
)
from gaia_clustering.velocity import fit_velocity_dispersion


@dataclass
class AssociationResult:
    """Container for one pipeline run.

    Attributes
    ----------
    name : str
        Target name supplied by the caller.
    query : dict
        Sesame/Gaia query metadata (object type, cone radius, row counts).
        When a catalogue was passed in, ``status`` is ``"catalog"``.
    catalog : pandas.DataFrame
        Gaia astrometry used for clustering (no Gaia radial velocities).
    membership : dict
        Output of :func:`~gaia_clustering.clustering.cluster_association`.
    velocity : dict or None
        Output of :func:`~gaia_clustering.velocity.fit_velocity_dispersion`,
        or ``None`` if inference was skipped.
    membership_threshold : float
        ``P(member)`` cut used for hard membership and the velocity fit.
    extras : dict
        Free-form slot for caller metadata.
    """

    name: str
    query: dict
    catalog: pd.DataFrame
    membership: dict
    velocity: dict | None = None
    membership_threshold: float = 0.5
    extras: dict = field(default_factory=dict)

    @property
    def table(self) -> pd.DataFrame:
        """Catalogue plus ``member_prob``, ``cluster_id``, ``is_member``, ``use_rv``."""
        tab = self.membership.get("table")
        return tab if tab is not None else self.catalog

    @property
    def members(self) -> pd.DataFrame:
        """Rows with ``is_member`` True (the velocity-fit sample)."""
        tab = self.table
        return tab.loc[tab["is_member"]].copy()

    @property
    def n_members(self) -> int:
        """Number of hard association members."""
        return int(self.table["is_member"].sum())

    @property
    def n_field(self) -> int:
        """Number of stars classified as field."""
        return int((~self.table["is_member"]).sum())

    def summary(self) -> dict:
        """JSON-serializable counts, preferred spatial ``K``, and velocity moments."""
        best = self.membership["best"]
        out = {
            "name": self.name,
            "kind": self.query.get("kind"),
            "n_catalog": int(len(self.catalog)),
            "n_members": self.n_members,
            "n_field": self.n_field,
            "n_with_rv": int(self.membership["use_rv"].sum()),
            "membership_threshold": self.membership_threshold,
            "structure": best.get("structure"),
            "preferred_K": best.get("preferred_K"),
        }
        if self.velocity is not None:
            out["n_vel"] = self.velocity["n_vel"]
            out["velocity_mean_kms"] = self.velocity["mean"].tolist()
            out["velocity_std_kms"] = self.velocity["std"].tolist()
            out["sigma_1d_kms"] = float(self.velocity["sigma_1d"])
            out["sigma_1d_quantiles_kms"] = list(self.velocity["sigma_1d_quantiles"])
            out["velocity_corr"] = self.velocity["corr"].tolist()
        return out

    def plot_summary(self, **kwargs):
        """Sky, tangential-velocity, and spatial-K BIC panels."""
        return plot_membership_summary(self.table, self.membership, **kwargs)

    def plot_3d(self, traceback=(0, 10), n_frames=9, **kwargs):
        """Interactive sky positions with residual velocity arrows and traceback.

        Parameters
        ----------
        traceback : tuple of float
            Lookback-time range in Myr for the slider. A scalar is also
            accepted by the underlying plotter.
        n_frames : int
            Number of traceback frames between the range endpoints.
        **kwargs
            Forwarded to :func:`~gaia_clustering.plots.plot_association_3d`.
        """
        return plot_association_3d(
            self.table, self.membership["best"],
            traceback=traceback, n_frames=n_frames, **kwargs,
        )

    def plot_corner(self, **kwargs):
        """5D (or 6D with RV) observable corner, coloured by membership."""
        return plot_association_corner(self.table, self.membership["best"], **kwargs)

    def plot_velocity(self, **kwargs):
        """Posterior velocity ellipsoid of the member Gaussian."""
        if self.velocity is None:
            raise RuntimeError("No velocity inference was run")
        return plot_velocity_gaussian(self.velocity, **kwargs)

    def plot_velocity_corner(self, **kwargs):
        """Corner plot of the hierarchical velocity posterior."""
        if self.velocity is None:
            raise RuntimeError("No velocity inference was run")
        return plot_velocity_corner(self.velocity, **kwargs)

    def save(self, output_dir, traceback=(0, 10)):
        """Write the membership table, JSON summary, and standard plots.

        Parameters
        ----------
        output_dir : path-like
            Destination directory (created if needed).
        traceback : tuple of float
            Lookback-time range (Myr) for ``association_3d.html``.

        Returns
        -------
        pathlib.Path
            The output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(output_dir / "membership.csv", index=False)
        Path(output_dir / "summary.json").write_text(
            json.dumps(self.summary(), indent=2, default=_json_default),
            encoding="utf-8",
        )
        fig, _ = self.plot_summary()
        fig.savefig(output_dir / "membership_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        fig = self.plot_corner()
        fig.savefig(output_dir / "membership_corner.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        fig3d = self.plot_3d(traceback=traceback)
        fig3d.write_html(str(output_dir / "association_3d.html"), include_plotlyjs="cdn")
        if self.velocity is not None:
            vfig = self.plot_velocity()
            vfig.write_html(str(output_dir / "velocity_3d.html"), include_plotlyjs="cdn")
            if "trace" in self.velocity:
                cfig = self.plot_velocity_corner()
                cfig.savefig(output_dir / "velocity_corner.png", dpi=150, bbox_inches="tight")
                plt.close(cfig)
        return output_dir


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def analyze_association(
    name=None,
    catalog=None,
    rv_table=None,
    rv_path=None,
    use_rv=None,
    radius_deg=1.5,
    g_max=13.0,
    ruwe_max=1.4,
    plx_snr_min=5.0,
    parallax_window=(0.4, 2.5),
    membership_probability_threshold=0.5,
    k_grid=(1, 2, 3),
    infer_velocity=True,
    cache_dir=DEFAULT_CACHE_DIR,
    n_init=5,
    max_iter=100,
    random_state=0,
    velocity_draws=1000,
    velocity_tune=1000,
    velocity_chains=4,
    velocity_cores=4,
):
    """Query Gaia, infer membership, and fit the member velocity Gaussian.

    Parameters
    ----------
    name : str, optional
        Cluster/association name, or a star in the association. Used to
        query Gaia unless ``catalog`` is supplied.
    catalog : DataFrame or path, optional
        Pre-built Gaia-like table. Skips the TAP query.
    rv_table : DataFrame, optional
        Independently measured radial velocities. Gaia RVs are ignored.
    rv_path : path-like, optional
        CSV/parquet of independent RVs; loaded if ``rv_table`` is omitted.
    use_rv : bool, str, or array-like of bool, optional
        Per-star RV inclusion. Default: use rows that have a usable
        independent RV (see :func:`~gaia_clustering.coords.resolve_use_rv`).
    radius_deg : float
        Gaia cone radius in degrees.
    g_max : float
        Faint Gaia *G* magnitude limit.
    ruwe_max : float
        Maximum renormalised unit weight error.
    plx_snr_min : float
        Minimum ``parallax_over_error``.
    parallax_window : tuple of float or None
        Multiplicative parallax window around a resolved star's parallax.
        Ignored for groups with no Gaia source. ``None`` disables the cut.
    membership_probability_threshold : float
        ``P(member)`` cut for hard membership and the velocity fit.
    k_grid : sequence of int
        Spatial cluster counts compared by BIC among members.
    infer_velocity : bool
        If True, run hierarchical Bayesian velocity inference on members.
    cache_dir : path-like or None
        Directory for pickled Gaia TAP results.
    n_init, max_iter, random_state
        Extreme-deconvolution fitting controls.
    velocity_draws, velocity_tune, velocity_chains, velocity_cores : int
        NUTS sampling controls for the velocity model.

    Returns
    -------
    AssociationResult
    """
    query_info = {"input_name": name, "kind": None, "status": "catalog"}
    if catalog is None:
        if not name:
            raise ValueError("Provide a target name or a catalog")
        print("Resolving {!r} and querying Gaia DR3...".format(name))
        catalog, query_info = fetch_neighbourhood(
            name,
            radius_deg=radius_deg,
            g_max=g_max,
            ruwe_max=ruwe_max,
            plx_snr_min=plx_snr_min,
            parallax_window=parallax_window,
            cache_dir=cache_dir,
        )
        print(
            "  {} ({}) → {} Gaia stars in a {:.2f} deg cone".format(
                query_info.get("simbad_main_id") or name,
                query_info.get("kind"),
                len(catalog),
                radius_deg,
            )
        )
    else:
        if isinstance(catalog, (str, Path)):
            catalog = pd.read_csv(catalog)
        else:
            catalog = catalog.copy()
        if name is None:
            name = str(query_info["input_name"] or "catalog")

    if rv_path is not None:
        rv_table = load_rv_table(rv_path)
    if rv_table is not None:
        catalog, n_rv = attach_radial_velocities(catalog, rv_table)
        print("  attached independent RVs for {} stars".format(n_rv))

    needed = ["ra", "dec", "parallax", "pmra", "pmdec",
              "parallax_error", "pmra_error", "pmdec_error"]
    missing = [c for c in needed if c not in catalog.columns]
    if missing:
        raise ValueError("Catalog is missing columns: {}".format(missing))
    catalog = catalog.dropna(subset=needed).copy()
    catalog = catalog.loc[np.asarray(catalog["parallax"], dtype=float) > 0].copy()
    if len(catalog) < 8:
        raise ValueError(
            "Only {} stars left after quality cuts; need more for clustering".format(
                len(catalog)
            )
        )

    print("Clustering {} stars with extreme deconvolution...".format(len(catalog)))
    membership = cluster_association(
        catalog,
        k_grid=k_grid,
        membership_probability_threshold=membership_probability_threshold,
        use_rv=use_rv,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    n_mem = int(membership["table"]["is_member"].sum())
    print(
        "  structure: {}; members / field = {} / {}".format(
            membership["best"]["structure"], n_mem, len(catalog) - n_mem,
        )
    )
    print("  stars contributing RV: {}".format(int(membership["use_rv"].sum())))

    velocity = None
    if infer_velocity:
        members = membership["table"].loc[membership["table"]["is_member"]]
        if len(members) < 3:
            print("  too few members for velocity inference; skipping")
        else:
            dim = "3D" if membership["use_rv"][membership["table"]["is_member"].values].any() else "2D tangential"
            print("Inferring {} velocity dispersion ({} members)...".format(dim, len(members)))
            velocity = fit_velocity_dispersion(
                members,
                use_rv=members["use_rv"].values if "use_rv" in members.columns else None,
                draws=velocity_draws,
                tune=velocity_tune,
                chains=velocity_chains,
                cores=velocity_cores,
                random_seed=random_state,
            )
            sig = velocity["sigma_1d"]
            lo, med, hi = velocity["sigma_1d_quantiles"]
            print(
                "  σ_1D = {:.2f} km/s  (16/50/84% = {:.2f} / {:.2f} / {:.2f})".format(
                    sig, lo, med, hi
                )
            )
            print("  component σ [km/s] = {}".format(
                np.array2string(velocity["std"], precision=2)
            ))

    return AssociationResult(
        name=name,
        query=query_info,
        catalog=catalog,
        membership=membership,
        velocity=velocity,
        membership_threshold=membership_probability_threshold,
    )
