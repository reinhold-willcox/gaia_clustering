# Usage

## Python API

Query Gaia first, tighten the cuts in memory, inspect the three panels,
then cluster and fit velocities:

```python
from gaia_clustering import query_association, analyze_association

query = query_association("Cyg OB3", position_radius_arcmin=10, g_max=20)
query.refine_search(
    position_radius_arcmin=8,
    pm_center=(-3.2, -4.1),
    pm_radius=2.0,
    parallax_range=(2.0, 8.0),
)
query.plot_query(parallax_range=(0.0, 12.0))
# query.reset_search()   # undo refine_search; back to the TAP download

result = analyze_association(query)
# If plot_summary / preferred_K shows more than one lobe, choose one:
# result = analyze_association(query, velocity_cluster_id=0)

print(result.summary())
result.plot_summary()
result.plot_corner()
result.plot_3d(traceback=(0, 10))   # slider, Myr lookback
result.plot_velocity()
result.save("results/cyg_ob3")
```

`query_association` is a broad TAP search: name-resolved centre,
on-sky radius, *G* limit, and an optional parallax range. Accepted
stars after `refine_search` must pass the on-sky cone, the
proper-motion cone (when `pm_radius` is set), and the parallax range
(when set). `reset_search` restores the original TAP cuts.
`plot_query` only draws; it does not change the cuts.

### Multi-lobe velocity fits

Spatial clustering may prefer $K>1$ lobes among members
(`preferred_K` in `result.summary()`). The membership plots colour each
lobe separately (colorbar titles `1…K` correspond to
`cluster_id = 0…K−1`).

When `infer_velocity=True` and `preferred_K > 1`, you **must** pass
`velocity_cluster_id` — the hierarchical velocity model is fit to that
lobe only:

```python
result = analyze_association(query, velocity_cluster_id=0)
print(result.velocity_cluster_id, len(result.velocity_members))
```

Omit `velocity_cluster_id` when there is a single preferred lobe, or when
`infer_velocity=False`.

A star inside the association works the same way; SIMBAD coordinates
centre the TAP cone, and `refine_search(position_center=...)` can
move the selected cone without a new download (the red X still marks
the name-resolved default):

```python
query = query_association("Cyg X-1", position_radius_arcmin=120, g_max=20)
query.refine_search(position_center="20h32m25.8s +41d18m31s")
query.plot_query()
result = analyze_association(
    query,
    rv_path="my_rvs.csv",
    membership_probability_threshold=0.9,
)
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
gaia-clustering "Cyg OB3" --radius 10 --g-max 20 -o results/cyg_ob3
gaia-clustering "Cyg X-1" --rv-file my_rvs.csv --threshold 0.9 -o results/cygx1
gaia-clustering --catalog stars.csv --no-velocity -o results/local
# Multi-lobe velocity (cluster_id from membership table / colorbar − 1):
gaia-clustering "Cyg OB3" --velocity-cluster-id 0 -o results/cyg_ob3
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

## Worked example

The notebook {doc}`example_systems` walks through a full association
analysis (query → refine → membership → velocity → plots). A no-network
synthetic demo is also available:

```bash
python examples/run_synthetic.py
```

## Outputs

{meth}`~gaia_clustering.pipeline.AssociationResult.save` writes:

- `membership.csv` — catalogue plus `member_prob`, `cluster_id`, `is_member`, `use_rv`
- `summary.json` — counts, preferred spatial $K$, chosen velocity lobe, $\mu$ and $\sigma$
- `membership_summary.png` — sky, $v_{\alpha*}$–$v_{\delta}$, parallax, spatial BIC
- `membership_corner.png` — 5D astrometry (6D if RVs are present)
- `association_3d.html` — interactive positions, residual velocity arrows, traceback slider
- `velocity_3d.html`, `velocity_corner.png` — hierarchical velocity posterior (if sampled)
