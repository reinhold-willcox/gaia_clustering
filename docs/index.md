# gaia_clustering

<div class="gc-hero" markdown="1">

<p class="gc-brand">gaia_clustering</p>

<p class="gc-lede">
Query Gaia around a stellar association, separate members from the field with
extreme deconvolution, and fit a hierarchical velocity Gaussian — with clear
inspection plots at every step.
</p>

<div class="gc-actions">
<a class="gc-btn gc-btn-primary" href="install.html">Install</a>
<a class="gc-btn gc-btn-secondary" href="usage.html">Usage</a>
<a class="gc-btn gc-btn-secondary" href="example_systems.html">Worked example</a>
</div>

</div>

```{toctree}
:maxdepth: 2
:caption: User guide

install
usage
method
example_systems
```

```{toctree}
:maxdepth: 2
:caption: API

api
```

## Pipeline at a glance

<div class="gc-steps" markdown="1">

<div class="gc-step" markdown="1">

**1. Query**

Resolve a name with Sesame / SIMBAD and download a Gaia DR3 neighbourhood.

</div>

<div class="gc-step" markdown="1">

**2. Refine**

Tighten sky, proper-motion, and parallax cuts in memory; inspect with `plot_query`.

</div>

<div class="gc-step" markdown="1">

**3. Members**

Extreme deconvolution in 6D phase space separates the association from the field.

</div>

<div class="gc-step" markdown="1">

**4. Lobes**

Compare spatial $K$ by BIC. When $K>1$, pick one lobe for the velocity fit.

</div>

<div class="gc-step" markdown="1">

**5. Velocity**

Hierarchical Bayesian sampling of the member velocity Gaussian (PyMC / NUTS).

</div>

</div>

**Gaia radial velocities are never used.** Independently measured RVs can be attached per star.

```python
from gaia_clustering import query_association, analyze_association

query = query_association("Cyg OB3", position_radius_arcmin=10, g_max=20)
query.refine_search(parallax_range=(2.0, 8.0))
query.plot_query()

result = analyze_association(query)          # single lobe: no extra arg
# result = analyze_association(query, velocity_cluster_id=0)  # required if K > 1

print(result.summary())
result.plot_summary()
result.plot_3d(traceback=(0, 10))
```

See the {doc}`worked example <example_systems>` notebook for a full end-to-end run,
and {doc}`method` for the membership and velocity models.
