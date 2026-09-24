# gaia_clustering

[![Tests](https://github.com/reinhold-willcox/gaia_clustering/actions/workflows/tests.yml/badge.svg)](https://github.com/reinhold-willcox/gaia_clustering/actions/workflows/tests.yml)
[![Documentation](https://github.com/reinhold-willcox/gaia_clustering/actions/workflows/docs.yml/badge.svg)](https://reinhold-willcox.github.io/gaia_clustering/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Query Gaia around a stellar association (or a star inside one), separate members from nearby field stars with extreme deconvolution, and infer the association velocity dispersion with hierarchical Bayesian sampling.

**Documentation:** [https://reinhold-willcox.github.io/gaia_clustering/](https://reinhold-willcox.github.io/gaia_clustering/)

Gaia radial velocities are never used. Independently measured RVs can be attached per star. When spatial clustering prefers more than one lobe, the velocity fit takes a required `velocity_cluster_id`.

## Install

```bash
git clone https://github.com/reinhold-willcox/gaia_clustering.git
cd gaia_clustering
conda env create -f environment.yml
conda activate gaia_clustering
```

Or, in an environment that already has NumPy, Astropy, PyMC, and Plotly:

```bash
pip install -e .
```

## Quick start

```python
from gaia_clustering import query_association, analyze_association

query = query_association("Cyg OB3", position_radius_arcmin=10, g_max=20)
query.plot_query()
result = analyze_association(query)
# If preferred_K > 1: analyze_association(query, velocity_cluster_id=0)
print(result.summary())
result.plot_3d(traceback=(0, 10))
result.save("results/cyg_ob3")
```

```bash
gaia-clustering "Cyg OB3" --radius 10 --g-max 20 -o results/cyg_ob3
python examples/run_synthetic.py
```

See the [usage guide](https://reinhold-willcox.github.io/gaia_clustering/usage.html), the [worked example notebook](https://reinhold-willcox.github.io/gaia_clustering/example_systems.html), and the [method page](https://reinhold-willcox.github.io/gaia_clustering/method.html).

## Tests

```bash
python -m pytest tests
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
