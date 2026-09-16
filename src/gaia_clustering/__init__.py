"""Membership and hierarchical velocity-dispersion pipeline for Gaia.

Query a stellar association (or a star inside one), separate members from
nearby field stars with extreme deconvolution, and infer the association
velocity Gaussian. Gaia radial velocities are never used; independently
measured RVs may be attached per star.

The usual entry point is :func:`analyze_association`.
"""

from gaia_clustering.clustering import cluster_association, synthesize_two_lobe_association
from gaia_clustering.pipeline import AssociationResult, analyze_association
from gaia_clustering.plots import (
    plot_association_3d,
    plot_association_corner,
    plot_bic_comparison,
    plot_proper_motion_membership,
    plot_sky_membership,
    plot_velocity_corner,
    plot_velocity_gaussian,
)

__all__ = [
    "AssociationResult",
    "analyze_association",
    "cluster_association",
    "plot_association_3d",
    "plot_association_corner",
    "plot_bic_comparison",
    "plot_proper_motion_membership",
    "plot_sky_membership",
    "plot_velocity_corner",
    "plot_velocity_gaussian",
    "synthesize_two_lobe_association",
]

__version__ = "0.1.0"
