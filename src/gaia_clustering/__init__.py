"""Membership and hierarchical velocity-dispersion pipeline for Gaia.

Query a stellar association (or a star inside one), inspect the Gaia
cone and parallax cuts, separate members from nearby field stars with
extreme deconvolution, and infer the association velocity Gaussian.
Gaia radial velocities are never used; independently measured RVs may
be attached per star. When more than one spatial lobe is preferred,
pass ``velocity_cluster_id`` to fit velocities for a single lobe.

The usual entry point is :func:`query_association` followed by
:meth:`GaiaQuery.refine_search`, :meth:`GaiaQuery.plot_query`, and
:func:`analyze_association`.
"""

from gaia_clustering.clustering import cluster_association, synthesize_two_lobe_association
from gaia_clustering.pipeline import AssociationResult, GaiaQuery, analyze_association, query_association
from gaia_clustering.plots import (
    plot_association_3d,
    plot_association_corner,
    plot_bic_comparison,
    plot_parallax_membership,
    plot_proper_motion_membership,
    plot_query,
    plot_sky_membership,
    plot_velocity_corner,
    plot_velocity_gaussian,
)

__all__ = [
    "AssociationResult",
    "GaiaQuery",
    "analyze_association",
    "cluster_association",
    "plot_association_3d",
    "plot_association_corner",
    "plot_bic_comparison",
    "plot_parallax_membership",
    "plot_proper_motion_membership",
    "plot_query",
    "plot_sky_membership",
    "plot_velocity_corner",
    "plot_velocity_gaussian",
    "query_association",
    "synthesize_two_lobe_association",
]

__version__ = "0.1.0"
