# gaia_clustering documentation

Gaia membership and hierarchical velocity-dispersion pipeline for stellar
associations.

```{toctree}
:maxdepth: 2
:caption: User guide

install
usage
method
```

```{toctree}
:maxdepth: 2
:caption: API

api
```

## What it does

1. Resolve a cluster, association, or star with CDS Sesame / SIMBAD.
2. Query Gaia DR3 around that position. **Gaia radial velocities are ignored.**
3. Separate members from field stars with extreme deconvolution in 6D
   ICRS phase space ([Bovy, Hogg & Roweis 2011](https://ui.adsabs.harvard.edu/abs/2011AnA...543A.106B)).
4. Infer the member velocity Gaussian with hierarchical Bayesian sampling.
5. Plot sky membership, a 5D/6D corner, and an interactive 3D traceback.

```python
from gaia_clustering import analyze_association

result = analyze_association("Cyg OB3", radius_deg=1.5, g_max=12)
print(result.summary())
result.plot_3d(traceback=(0, 10))
```
