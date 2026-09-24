"""Synthetic two-lobe association, no Gaia query required.

Run from the repo root after ``pip install -e .``::

    python examples/run_synthetic.py
"""

from pathlib import Path

from gaia_clustering import analyze_association, synthesize_two_lobe_association


def main():
    df, meta = synthesize_two_lobe_association(
        n_per_lobe=50, n_field=30, rv_fraction=0.25, random_state=0,
    )
    print("Synthetic catalogue: {} stars, true σ = {:.1f} km/s, "
          "{} with independent RV".format(
              len(df), meta["v_disp_kms"], int(df["radial_velocity"].notna().sum())
          ))
    # Membership first without velocity so we can choose a lobe if BIC
    # prefers K > 1.
    preview = analyze_association(
        name="synthetic two-lobe association",
        catalog=df,
        use_rv=None,
        infer_velocity=False,
        membership_probability_threshold=0.5,
        random_state=0,
        n_init=5,
        max_iter=100,
    )
    preferred_K = int(preview.membership["best"]["preferred_K"])
    velocity_cluster_id = 0 if preferred_K > 1 else None
    if preferred_K > 1:
        print(
            "preferred_K = {}; fitting velocity for cluster_id = {}".format(
                preferred_K, velocity_cluster_id,
            )
        )
    result = analyze_association(
        name="synthetic two-lobe association",
        catalog=df,
        use_rv=None,
        infer_velocity=True,
        velocity_cluster_id=velocity_cluster_id,
        membership_probability_threshold=0.5,
        velocity_draws=400,
        velocity_tune=400,
        velocity_chains=2,
        velocity_cores=2,
        random_state=0,
    )
    out = Path("results/synthetic")
    result.save(out, traceback=(0, 8))
    print("Wrote", out.resolve())
    print(result.summary())


if __name__ == "__main__":
    main()
