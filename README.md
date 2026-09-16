# gaia_clustering

[![Tests](https://github.com/rwillcox/gaia_clustering/actions/workflows/tests.yml/badge.svg)](https://github.com/rwillcox/gaia_clustering/actions/workflows/tests.yml)
[![Documentation](https://github.com/rwillcox/gaia_clustering/actions/workflows/docs.yml/badge.svg)](https://rwillcox.github.io/gaia_clustering/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Query Gaia around a stellar association (or a star inside one), separate members from nearby field stars with extreme deconvolution, and infer the association velocity dispersion with hierarchical Bayesian sampling.

**Documentation:** [https://rwillcox.github.io/gaia_clustering/](https://rwillcox.github.io/gaia_clustering/)

Gaia radial velocities are never used. Independently measured RVs can be attached per star.

## Install

```bash
git clone https://github.com/rwillcox/gaia_clustering.git
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
from gaia_clustering import analyze_association

result = analyze_association("Cyg OB3", radius_deg=1.5, g_max=12)
print(result.summary())
result.plot_3d(traceback=(0, 10))
result.save("results/cyg_ob3")
```

```bash
gaia-clustering "Cyg OB3" --radius 1.5 --g-max 12 -o results/cyg_ob3
python examples/run_synthetic.py
```

See the [usage guide](https://rwillcox.github.io/gaia_clustering/usage.html) for independent RVs, CLI flags, and plot products, and the [method page](https://rwillcox.github.io/gaia_clustering/method.html) for the membership and velocity models.

## Tests

```bash
python -m pytest tests
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
