"""Command-line entry point: ``gaia-clustering "Cyg OB3"``."""

from __future__ import annotations

import argparse
from pathlib import Path

from gaia_clustering.pipeline import analyze_association


def build_parser():
    """Return the ``gaia-clustering`` argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Query Gaia around a cluster, association, or star; "
            "infer membership by extreme deconvolution; and fit the "
            "association velocity dispersion hierarchically."
        )
    )
    p.add_argument(
        "name",
        nargs="?",
        help="Cluster/association name, or a star in the association",
    )
    p.add_argument(
        "--catalog",
        help="Pre-built CSV of Gaia astrometry (skips the TAP query)",
    )
    p.add_argument(
        "--rv-file",
        help="CSV of independent radial velocities (Gaia RVs are ignored)",
    )
    p.add_argument("--radius", type=float, default=1.5, help="Cone radius in degrees")
    p.add_argument("--g-max", type=float, default=13.0, help="Gaia G magnitude limit")
    p.add_argument("--ruwe-max", type=float, default=1.4)
    p.add_argument("--plx-snr-min", type=float, default=5.0)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="P(member) cut for hard membership and the velocity fit",
    )
    p.add_argument(
        "--no-velocity",
        action="store_true",
        help="Skip hierarchical velocity inference",
    )
    p.add_argument(
        "--draws", type=int, default=1000, help="NUTS draws per chain",
    )
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument(
        "--traceback",
        type=float,
        nargs=2,
        metavar=("T0", "T1"),
        default=(0.0, 10.0),
        help="Lookback-time range (Myr) for the interactive 3D slider",
    )
    p.add_argument(
        "-o", "--output",
        default="results",
        help="Directory for CSV, JSON, PNG, and HTML products",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Gaia TAP cache directory (default: data/cache)",
    )
    return p


def main(argv=None):
    """Run the pipeline from the command line.

    Parameters
    ----------
    argv : list of str, optional
        If omitted, ``sys.argv[1:]`` is used.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    if not args.name and not args.catalog:
        raise SystemExit("Provide a target name or --catalog")
    kwargs = {}
    if args.cache_dir is not None:
        kwargs["cache_dir"] = args.cache_dir
    result = analyze_association(
        name=args.name,
        catalog=args.catalog,
        rv_path=args.rv_file,
        radius_deg=args.radius,
        g_max=args.g_max,
        ruwe_max=args.ruwe_max,
        plx_snr_min=args.plx_snr_min,
        membership_probability_threshold=args.threshold,
        infer_velocity=not args.no_velocity,
        velocity_draws=args.draws,
        velocity_chains=args.chains,
        velocity_cores=args.cores,
        **kwargs,
    )
    out = Path(args.output)
    result.save(out, traceback=tuple(args.traceback))
    print("Wrote products to {}".format(out.resolve()))
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
