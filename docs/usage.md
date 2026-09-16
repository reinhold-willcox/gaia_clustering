# Usage

## Python API

The usual entry point is {func}`gaia_clustering.analyze_association`.

```python
from gaia_clustering import analyze_association

# Cluster / association name
result = analyze_association("Cyg OB3", radius_deg=1.5, g_max=12)

# Or a star inside the association
result = analyze_association("Cyg X-1", radius_deg=2.0, g_max=12)

# Independent RVs (CSV with source_id or name + radial_velocity + error)
result = analyze_association(
    "Cyg OB3",
    rv_path="my_rvs.csv",
    membership_probability_threshold=0.9,
)

print(result.summary())
result.plot_summary()
result.plot_corner()
result.plot_3d(traceback=(0, 10))   # slider, Myr lookback
result.plot_velocity()
result.save("results/cyg_ob3")
```

Skip the Gaia query by passing a catalogue:

```python
result = analyze_association(
    name="my field",
    catalog="stars.csv",
    infer_velocity=True,
)
```

## Command line

```bash
gaia-clustering "Cyg OB3" --radius 1.5 --g-max 12 -o results/cyg_ob3
gaia-clustering "Cyg X-1" --rv-file my_rvs.csv --threshold 0.9 -o results/cygx1
gaia-clustering --catalog stars.csv --no-velocity -o results/local
```

## Independent radial velocities

Gaia RVs are never used. Attach ground-based (or other) RVs with a table:

| column | required | notes |
| --- | --- | --- |
| `source_id` or `name` / `designation` | yes | match key onto the Gaia catalogue |
| `radial_velocity` | yes | km/s |
| `radial_velocity_error` | yes | km/s |

A template is in `examples/rvs_template.csv`. Stars without a match stay
RV-free: they still inform membership and the tangential velocity
dispersion.

## Synthetic demo (no Gaia)

```bash
python examples/run_synthetic.py
```

## Outputs

{meth}`~gaia_clustering.pipeline.AssociationResult.save` writes:

- `membership.csv` — catalogue plus `member_prob`, `cluster_id`, `is_member`, `use_rv`
- `summary.json` — counts, preferred spatial $K$, $\mu$ and $\sigma$ of the velocity Gaussian
- `membership_summary.png` — sky, $v_{\alpha*}$–$v_{\delta}$, spatial BIC
- `membership_corner.png` — 5D astrometry (6D if RVs are present)
- `association_3d.html` — interactive positions, residual velocity arrows, traceback slider
- `velocity_3d.html`, `velocity_corner.png` — hierarchical velocity posterior (if sampled)
