# Method

## Coordinates

Gaia observables are packed as
$(\alpha, \delta, \varpi, \mu_{\alpha*}, \mu_{\delta}[, v_r])$
in degrees, mas, mas/yr, and km/s. Gaia `ra_error` / `dec_error` are mas
and are converted to degrees. The pipeline converts to ICRS phase space
$(X, Y, Z)$ in pc and $(V_X, V_Y, V_Z)$ in km/s, propagating each star's
covariance with a numerical Jacobian $C_\mathrm{phase} = J C_\mathrm{obs} J^\top$.

Radial velocity is optional per star. Unused $v_r$ is given an
uninformative $\sigma \sim 10^3\,\mathrm{km\,s^{-1}}$ so extreme
deconvolution does not pull on the line of sight.

## Membership (extreme deconvolution)

[Bovy, Hogg & Roweis (2011)](https://ui.adsabs.harvard.edu/abs/2011AnA...543A.106B)
extreme deconvolution is a Gaussian mixture that deconvolves measurement
noise: star $i$ is compared to component $k$ with covariance $T_k + S_i$.

Two stages:

1. **Association vs field** in 6D phase space. One compact association
   Gaussian plus a wide field component (default $100\,\mathrm{pc}$,
   $25\,\mathrm{km\,s^{-1}}$). `member_prob` is $1$ minus the field
   responsibility.
2. **Spatial structure** among members only (3D position). $K = 1, 2, 3$
   are compared by BIC. `cluster_id` is $-1$ for field and $0, 1, \ldots$
   for spatial lobes.

## Velocity dispersion

After membership, stars in **one** spatial lobe are fit with a
hierarchical Bayesian velocity Gaussian (PyMC / NUTS). When BIC prefers
a single lobe, all hard members are used. When `preferred_K > 1`,
{func}`~gaia_clustering.pipeline.analyze_association` requires
`velocity_cluster_id` (the same integer as the `cluster_id` column;
summary colorbars are labelled `cluster_id + 1`) and fits only that lobe.

The generative model:

- Population velocities $(v_\alpha, v_\delta[, v_r]) \sim \mathcal{N}(\mu, \Sigma)$
- Latent distance with $p(d) \propto d^2$ (uniform in volume); $\varpi = 1/d$
- Predicted proper motions $\mu_{\alpha,\delta} = v_{\alpha,\delta}\, \varpi / 4.74047$
- Gaia likelihood on $(\varpi, \mu_\alpha, \mu_\delta)$
- Independent RVs, when present, add a 1D likelihood on $v_r$ for those stars only

If nobody has RV the fit is 2D (tangential). The scalar summary is
$\sigma_{1\mathrm{D}} = \sqrt{\mathrm{mean}(\sigma_i^2)}$.

This Gaussian describes **today's** member velocity scatter, including
expansion or streaming. It is not a birth-time isotropic jitter unless
the association is cold and non-expanding.

## Query inspection

{func}`~gaia_clustering.pipeline.query_association` downloads a Gaia
cone at the name-resolved centre, using the requested radius, *G*
limit, and optional parallax range. {meth}`~gaia_clustering.pipeline.GaiaQuery.refine_search`
then tightens the on-sky centre and radius, an optional proper-motion
cone, and the parallax range in memory (no new TAP query). The clustered
catalogue is the stars that pass the selected on-sky cone, proper-motion
cone (if set), and parallax range (if set). {func}`~gaia_clustering.plots.plot_query`
then shows on-sky positions with the selected cone overlaid, proper
motions, and individual parallaxes. The figure title is the queried
name plus the accepted fraction. Colour encodes which cuts a star
passes: pale red, blue, and yellow for sky, proper motion, and
parallax alone; light purple, orange, and green for the two-cut
mixes; dark brown for stars that pass all three; very light grey for
none. The
cone may be recentred with ``position_center`` on
{meth}`~gaia_clustering.pipeline.GaiaQuery.refine_search`; a red X marks the
name-resolved default position. A
black KDE of parallaxes for stars that pass both the on-sky and
proper-motion cuts (default bandwidth = mean $\sigma_\varpi$ of
those stars, peak scaled to 90% of the axis) is overplotted on the parallax panel. Red boundary marks
show the refined cone, proper-motion cone, and parallax range. Clustering
starts only when the {class}`~gaia_clustering.pipeline.GaiaQuery` is
passed to {func}`~gaia_clustering.pipeline.analyze_association`.

## Traceback

The 3D plotter coasts at constant ICRS velocity, $r \to r - v t$, with
$t$ in Myr (no Galactic potential). Stars without RV are coasted on
tangential motion only ($v_r = 0$).
