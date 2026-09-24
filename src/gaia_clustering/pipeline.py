"""End-to-end association analysis: query → inspect cuts → membership → velocity."""

from __future__ import annotations

import json
from copy import deepcopy
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
    apply_query_cuts,
    arcmin_to_deg,
    attach_radial_velocities,
    fetch_neighbourhood,
    load_rv_table,
    parse_pm_center,
    parse_position_center,
)
from gaia_clustering.velocity import fit_velocity_dispersion

_UNSET = object()


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
    velocity_cluster_id : int or None
        Spatial lobe (``cluster_id``) used for the velocity fit when more
        than one lobe is preferred. ``None`` if velocity was skipped or
        there was only one lobe.
    extras : dict
        Free-form slot for caller metadata.
    """

    name: str
    query: dict
    catalog: pd.DataFrame
    membership: dict
    velocity: dict | None = None
    membership_threshold: float = 0.5
    velocity_cluster_id: int | None = None
    extras: dict = field(default_factory=dict)

    @property
    def table(self) -> pd.DataFrame:
        """Catalogue plus ``member_prob``, ``cluster_id``, ``is_member``, ``use_rv``."""
        tab = self.membership.get("table")
        return tab if tab is not None else self.catalog

    @property
    def members(self) -> pd.DataFrame:
        """All hard association members (``is_member`` True), any lobe."""
        tab = self.table
        return tab.loc[tab["is_member"]].copy()

    @property
    def velocity_members(self) -> pd.DataFrame:
        """Stars used for the velocity fit (one lobe when multi-lobe)."""
        tab = self.table
        mask = tab["is_member"]
        if self.velocity_cluster_id is not None:
            mask = mask & (np.asarray(tab["cluster_id"]) == int(self.velocity_cluster_id))
        return tab.loc[mask].copy()

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
            "velocity_cluster_id": self.velocity_cluster_id,
        }
        if self.velocity is not None:
            out["n_velocity_members"] = int(len(self.velocity_members))
            out["n_vel"] = self.velocity["n_vel"]
            out["velocity_mean_kms"] = self.velocity["mean"].tolist()
            out["velocity_std_kms"] = self.velocity["std"].tolist()
            out["sigma_1d_kms"] = float(self.velocity["sigma_1d"])
            out["sigma_1d_quantiles_kms"] = list(self.velocity["sigma_1d_quantiles"])
            out["velocity_corr"] = self.velocity["corr"].tolist()
        return out

    def plot_summary(self, **kwargs):
        """Sky, tangential-velocity, parallax, and spatial-K ΔBIC panels."""
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


@dataclass
class GaiaQuery:
    """Gaia neighbourhood prepared for inspection, then clustering.

    :func:`~gaia_clustering.pipeline.query_association` downloads the
    broad TAP cone. :meth:`refine_search` tightens centres, radii, and
    the parallax range in memory (no new TAP query). :meth:`reset_search`
    restores the TAP cuts. :attr:`catalog` is the subset that will be
    clustered. :attr:`extended` is the full downloaded sample shown by
    :meth:`plot_query`.
    """

    name: str
    query: dict
    catalog: pd.DataFrame
    extended: pd.DataFrame
    _base_query: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._base_query is None:
            self._base_query = deepcopy(dict(self.query))

    def refine_search(
        self,
        position_center=_UNSET,
        position_radius_arcmin=None,
        pm_center=_UNSET,
        pm_radius=_UNSET,
        parallax_range=_UNSET,
    ):
        """Tighten sky, proper-motion, and parallax cuts on the downloaded sample.

        No new TAP query is made. ``position_radius_arcmin`` must not
        exceed the cone that was already downloaded. Accepted stars
        must pass the on-sky cone, the proper-motion cone (if set),
        and the parallax range (if set).

        Parameters
        ----------
        position_center : str, pair, or SkyCoord, optional
            New on-sky cone centre. Omitted leaves the current centre.
        position_radius_arcmin : float, optional
            New on-sky cone radius in arcminutes.
        pm_center : pair of float, optional
            Proper-motion cone centre ``(pmra, pmdec)`` in mas/yr.
        pm_radius : float or None, optional
            Proper-motion cone radius in mas/yr. ``None`` or ``0``
            disables the PM cut.
        parallax_range : tuple of float or None, optional
            Hard parallax limits in mas, ``(lo, hi)``. ``None`` clears
            the parallax cut.

        Returns
        -------
        GaiaQuery
            ``self``, with :attr:`catalog` and :attr:`query` updated.
        """
        info = dict(self.query)
        if position_radius_arcmin is not None:
            search = float(info.get(
                "search_radius_arcmin",
                info.get(
                    "preview_radius_arcmin",
                    info.get(
                        "position_radius_arcmin",
                        info.get(
                            "radius_arcmin",
                            float(info.get("preview_radius_deg", info.get("radius_deg", 0.0))) * 60.0,
                        ),
                    ),
                ),
            ))
            position_radius_arcmin = float(position_radius_arcmin)
            if position_radius_arcmin > search + 1e-9:
                raise ValueError(
                    "position_radius_arcmin={:.3f} exceeds the downloaded cone "
                    "({:.3f} arcmin); call query_association again with a "
                    "larger position_radius_arcmin".format(
                        position_radius_arcmin, search,
                    )
                )
            info["position_radius_arcmin"] = position_radius_arcmin
            info["radius_arcmin"] = position_radius_arcmin
            info["radius_deg"] = arcmin_to_deg(position_radius_arcmin)
        if position_center is not _UNSET and position_center is not None:
            cra, cdec = parse_position_center(position_center)
            info["center_ra"] = cra
            info["center_dec"] = cdec
            info["position_center"] = (
                position_center if isinstance(position_center, str)
                else (float(cra), float(cdec))
            )
            info["center_position"] = info["position_center"]
        if parallax_range is not _UNSET:
            if parallax_range is None:
                info["parallax_range"] = None
                info["parallax_cuts"] = None
                info["parallax_window"] = None
            else:
                cuts = tuple(parallax_range)
                if len(cuts) != 2:
                    raise ValueError("parallax_range must be a (lo, hi) pair")
                pair = (float(cuts[0]), float(cuts[1]))
                info["parallax_range"] = pair
                info["parallax_cuts"] = pair
                info["parallax_window"] = None
        if pm_center is not _UNSET:
            info["pm_center"] = None if pm_center is None else parse_pm_center(pm_center)
        if pm_radius is not _UNSET:
            if pm_radius is None:
                info["pm_radius"] = None
            else:
                info["pm_radius"] = float(pm_radius)
                if not np.isfinite(info["pm_radius"]) or info["pm_radius"] < 0:
                    raise ValueError("pm_radius must be a non-negative finite value in mas/yr")
                if info["pm_radius"] == 0:
                    info["pm_radius"] = None
        catalog = apply_query_cuts(self.extended, info)
        self.query = info
        self.catalog = catalog
        return self

    def reset_search(self):
        """Undo :meth:`refine_search` and restore the original TAP search.

        Centres, radii, and the parallax range go back to the values
        from :func:`query_association`. No new TAP query is made.

        Returns
        -------
        GaiaQuery
            ``self``, with :attr:`catalog` and :attr:`query` restored.
        """
        info = deepcopy(self._base_query)
        catalog = apply_query_cuts(self.extended, info)
        self.query = info
        self.catalog = catalog
        return self

    def plot_query(self, **kwargs):
        """Sky, proper-motion, and parallax diagnostics.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`~gaia_clustering.plots.plot_query`.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of Axes
        """
        from gaia_clustering.plots import plot_query as plot_fn
        return plot_fn(self, **kwargs)


def query_association(
    name,
    position_radius_arcmin,
    g_max,
    parallax_range=None,
    cache_dir=None,
    top=10000,
):
    """Resolve ``name`` and download a broad Gaia neighbourhood.

    This is the first search: TAP uses the name-resolved centre,
    ``position_radius_arcmin``, ``g_max``, and an optional parallax
    range. Tighten centres, radii, and the parallax range afterwards
    with :meth:`GaiaQuery.refine_search` (no new TAP query), then
    inspect with :meth:`GaiaQuery.plot_query`. Clustering is *not* run;
    pass the result to :func:`analyze_association` after the cuts look
    right.

    Parameters
    ----------
    name : str
        Cluster/association name, or a star in the association.
    position_radius_arcmin : float
        TAP cone radius in arcminutes.
    g_max : float or None
        Faint Gaia *G* magnitude limit. ``None`` skips the cut.
    parallax_range : tuple of float or None
        Hard TAP parallax limits in mas, ``(lo, hi)``. ``None``
        (default) does not filter on parallax.
    cache_dir : path-like, False, or None
        Directory for pickled Gaia TAP results. ``None`` (default) uses
        ``~/.cache/gaia_clustering`` (or ``$XDG_CACHE_HOME/gaia_clustering``).
        ``False`` disables caching.
    top : int
        Maximum cone-search rows.

    Returns
    -------
    GaiaQuery
    """
    if not name:
        raise ValueError("Provide a target name")
    print("Resolving {!r} and querying Gaia DR3...".format(name))
    catalog, query_info, extended = fetch_neighbourhood(
        name,
        position_radius_arcmin=position_radius_arcmin,
        g_max=g_max,
        parallax_range=parallax_range,
        cache_dir=cache_dir,
        top=top,
        return_extended=True,
    )
    print(
        "  {} ({}) → {} Gaia stars in a {:.1f} arcmin cone".format(
            query_info.get("simbad_main_id") or name,
            query_info.get("kind"),
            len(extended),
            position_radius_arcmin,
        )
    )
    return GaiaQuery(
        name=name,
        query=query_info,
        catalog=catalog,
        extended=extended,
    )


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def _lobe_member_counts(table):
    """Return sorted ``(cluster_id, n)`` pairs for hard members."""
    members = table.loc[table["is_member"]]
    if len(members) == 0 or "cluster_id" not in members.columns:
        return []
    ids = np.asarray(members["cluster_id"], dtype=int)
    ids = ids[ids >= 0]
    if ids.size == 0:
        return []
    out = []
    for k in np.unique(ids):
        out.append((int(k), int(np.sum(ids == k))))
    return out


def _select_velocity_members(table, preferred_K, velocity_cluster_id):
    """Restrict the velocity sample to one spatial lobe when needed.

    Parameters
    ----------
    table : DataFrame
        Membership table with ``is_member`` and ``cluster_id``.
    preferred_K : int
        Preferred number of spatial lobes.
    velocity_cluster_id : int or None
        ``cluster_id`` of the lobe to fit. Required when ``preferred_K > 1``.

    Returns
    -------
    members : DataFrame
        Stars used for the velocity fit.
    cluster_id : int or None
        Resolved lobe id (``0`` when there is a single lobe).
    """
    preferred_K = int(preferred_K) if preferred_K is not None else 1
    members = table.loc[table["is_member"]].copy()
    counts = dict(_lobe_member_counts(table))
    valid = sorted(counts)

    if preferred_K <= 1:
        if velocity_cluster_id is None:
            return members, (0 if valid else None)
        cid = int(velocity_cluster_id)
        if valid and cid not in counts:
            raise ValueError(
                "velocity_cluster_id={} is not among member lobes {}; "
                "single-lobe fits usually omit this argument".format(cid, valid)
            )
        if valid:
            members = members.loc[np.asarray(members["cluster_id"]) == cid].copy()
        return members, cid

    if velocity_cluster_id is None:
        detail = ", ".join(
            "id {} ({} stars)".format(k, counts[k]) for k in valid
        ) or "none"
        raise ValueError(
            "preferred_K={} spatial lobes; pass velocity_cluster_id to choose "
            "which lobe to fit (matching table cluster_id / colorbar label − 1). "
            "Available: {}".format(preferred_K, detail)
        )
    cid = int(velocity_cluster_id)
    if cid not in counts:
        detail = ", ".join(
            "id {} ({} stars)".format(k, counts[k]) for k in valid
        ) or "none"
        raise ValueError(
            "velocity_cluster_id={} is not among member lobes. Available: {}".format(
                cid, detail,
            )
        )
    members = members.loc[np.asarray(members["cluster_id"]) == cid].copy()
    return members, cid


def analyze_association(
    name=None,
    catalog=None,
    rv_table=None,
    rv_path=None,
    use_rv=None,
    position_radius_arcmin=10,
    g_max=20,
    parallax_range=None,
    membership_probability_threshold=0.5,
    k_grid=(1, 2, 3),
    infer_velocity=True,
    velocity_cluster_id=None,
    cache_dir=None,
    n_init=5,
    max_iter=100,
    random_state=0,
    velocity_draws=1000,
    velocity_tune=1000,
    velocity_chains=4,
    velocity_cores=4,
):
    """Infer membership and fit the member velocity Gaussian.

    The usual interactive path is :func:`query_association`, then
    :meth:`GaiaQuery.refine_search` and :meth:`GaiaQuery.plot_query`
    to confirm the cuts, then this function on the :class:`GaiaQuery`.
    Passing a target name still queries Gaia (used by the CLI).

    When spatial clustering prefers more than one lobe
    (``preferred_K > 1``) and ``infer_velocity`` is True,
    ``velocity_cluster_id`` is **required**: the hierarchical velocity
    model is fit to that lobe only (matching the ``cluster_id`` column;
    summary colorbars are labelled ``cluster_id + 1``).

    Parameters
    ----------
    name : str or GaiaQuery, optional
        A :class:`GaiaQuery` from :func:`query_association`, or a
        cluster/association/star name used to query Gaia unless
        ``catalog`` is supplied.
    catalog : DataFrame or path, optional
        Pre-built Gaia-like table. Skips the TAP query.
    rv_table : DataFrame, optional
        Independently measured radial velocities. Gaia RVs are ignored.
    rv_path : path-like, optional
        CSV/parquet of independent RVs; loaded if ``rv_table`` is omitted.
    use_rv : bool, str, or array-like of bool, optional
        Per-star RV inclusion. Default: use rows that have a usable
        independent RV (see :func:`~gaia_clustering.coords.resolve_use_rv`).
    position_radius_arcmin : float
        Gaia cone radius in arcminutes when querying by ``name``.
    g_max : float or None
        Faint Gaia *G* magnitude limit. ``None`` skips the cut.
    parallax_range : tuple of float or None
        Hard TAP parallax limits in mas, ``(lo, hi)``, when querying
        by ``name``. ``None`` (default) does not filter on parallax.
    membership_probability_threshold : float
        ``P(member)`` cut for hard membership and the velocity fit.
    k_grid : sequence of int
        Spatial cluster counts compared by BIC among members.
    infer_velocity : bool
        If True, run hierarchical Bayesian velocity inference on members.
    velocity_cluster_id : int or None
        Spatial lobe to fit when ``preferred_K > 1``. Same integer as
        ``cluster_id`` in the membership table (``0, 1, …``). Ignored
        when ``infer_velocity`` is False; optional when there is only
        one preferred lobe.
    cache_dir : path-like, False, or None
        Directory for pickled Gaia TAP results. ``None`` (default) uses
        ``~/.cache/gaia_clustering`` (or ``$XDG_CACHE_HOME/gaia_clustering``).
        ``False`` disables caching.
    n_init, max_iter, random_state
        Extreme-deconvolution fitting controls.
    velocity_draws, velocity_tune, velocity_chains, velocity_cores : int
        NUTS sampling controls for the velocity model.

    Returns
    -------
    AssociationResult
    """
    query_info = {"input_name": name, "kind": None, "status": "catalog"}
    if isinstance(name, GaiaQuery):
        query_obj = name
        name = query_obj.name
        catalog = query_obj.catalog.copy()
        query_info = dict(query_obj.query)
        print(
            "Using GaiaQuery {!r}: {} selected stars "
            "({} downloaded)".format(
                name,
                len(catalog),
                query_info.get("n_extended", len(query_obj.extended)),
            )
        )
    elif catalog is None:
        if not name:
            raise ValueError("Provide a GaiaQuery, a target name, or a catalog")
        print("Resolving {!r} and querying Gaia DR3...".format(name))
        catalog, query_info = fetch_neighbourhood(
            name,
            position_radius_arcmin=position_radius_arcmin,
            g_max=g_max,
            parallax_range=parallax_range,
            cache_dir=cache_dir,
        )
        print(
            "  {} ({}) → {} Gaia stars in a {:.1f} arcmin cone".format(
                query_info.get("simbad_main_id") or name,
                query_info.get("kind"),
                len(catalog),
                position_radius_arcmin,
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
    preferred_K = int(membership["best"].get("preferred_K") or 1)
    print(
        "  structure: {}; members / field = {} / {}".format(
            membership["best"]["structure"], n_mem, len(catalog) - n_mem,
        )
    )
    print("  stars contributing RV: {}".format(int(membership["use_rv"].sum())))

    velocity = None
    resolved_velocity_cluster_id = None
    if infer_velocity:
        members, resolved_velocity_cluster_id = _select_velocity_members(
            membership["table"], preferred_K, velocity_cluster_id,
        )
        if len(members) < 3:
            print("  too few members for velocity inference; skipping")
            resolved_velocity_cluster_id = None
        else:
            use_rv_mem = (
                members["use_rv"].values if "use_rv" in members.columns else None
            )
            dim = "3D" if (
                use_rv_mem is not None and np.asarray(use_rv_mem).any()
            ) else "2D tangential"
            lobe_txt = (
                " (cluster_id={})".format(resolved_velocity_cluster_id)
                if preferred_K > 1 else ""
            )
            print(
                "Inferring {} velocity dispersion ({} members{})...".format(
                    dim, len(members), lobe_txt,
                )
            )
            velocity = fit_velocity_dispersion(
                members,
                use_rv=use_rv_mem,
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
        velocity_cluster_id=resolved_velocity_cluster_id,
    )
