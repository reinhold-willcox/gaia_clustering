# Installation

## Conda environment

The scientific stack (NumPy, Astropy, PyMC, Plotly) is easiest from conda-forge.
A ready-made spec lives in `environment.yml` at the repository root:

```bash
git clone https://github.com/reinhold-willcox/gaia_clustering.git
cd gaia_clustering
conda env create -f environment.yml
conda activate gaia_clustering
```

Or install into an existing environment that already has those packages
(for example the `cyg_ob3` env used during development):

```bash
conda activate cyg_ob3
pip install -e .
```

## From source (pip)

```bash
pip install -e ".[dev]"
```

Optional extras:

- `dev` — pytest
- `docs` — Sphinx, Furo, MyST

## Build the documentation locally

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`.
