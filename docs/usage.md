# Usage

## Python API

Query Gaia first, inspect the cone and parallax window, then cluster:

```python
from gaia_clustering import query_association, analyze_association

query = query_association("Cyg OB3", radius_deg=1.5, g_max=12)
query.plot_query()   # preview neighbourhood vs selected cone and ϖ cuts

# Optional: tighten or widen cuts without a new TAP query
# (radius must stay inside the preview cone, default 2× radius_deg)
query.select(radius_deg=1.2, parallax_window=(0.5, 2.0))
query.plot_query()

result = analyze_association(query)
print(result.summary())
result.plot_summary()
result.plot_corner()
result.plot_3d(traceback=(0, 10))   # slider, Myr lookback
result.plot_velocity()
result.save("results/cyg_ob3")
```

A star inside the association works the same way; SIMBAD coordinates
centre the cone, and a Gaia DR3 source id (when Sesame publishes one)
sets the reference parallax for the multiplicative window:

```python
query = query_association("Cyg X-1", radius_deg=2.0, g_max=12)
query.plot_query()
result = analyze_association(query, rv_path="my_rvs.csv", membership_probability_threshold=0.9)
```

Skip the Gaia query by passing a catalogue:

```python
result = analyze_association(
    name="my field",
    catalog="stars.csv",
    infer_velocity=True,
)
```

The CLI still runs query and clustering in one shot (no interactive
preview). `analyze_association("Cyg OB3", ...)` remains valid for
scripts that already know the cuts.

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
