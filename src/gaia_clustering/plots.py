"""Membership and velocity plots for the association pipeline."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, to_hex
from matplotlib.cm import ScalarMappable
import plotly.graph_objects as go

from gaia_clustering.coords import AU_KMS, coast_dataframe, star_uses_rv

fs_title = 20
fs_label = 16
fs_text = 14

MEMBERSHIP_CMAP = "viridis"
OUTLIER_COLOR = "lightblue"


def set_membership_cmap(cmap):
    """Set the default sequential colormap for membership plots.

    High :math:`P(\\mathrm{member})` is drawn at the dark end
    (the colormap is reversed at plot time).
    """
    global MEMBERSHIP_CMAP
    MEMBERSHIP_CMAP = cmap
    return cmap


def set_outlier_color(color):
    """Set the default marker/hatch colour for hard field outliers."""
    global OUTLIER_COLOR
    OUTLIER_COLOR = color
    return color


def _outlier_color(color=None):
    if color is None:
        color = OUTLIER_COLOR
    return color


def _as_cmap(cmap=None):
    if cmap is None:
        cmap = MEMBERSHIP_CMAP
    if isinstance(cmap, str):
        return plt.get_cmap(cmap)
    return cmap


def _membership_display_cmap(cmap=None):
    """Sequential map with high P(member) at the dark end."""
    return _as_cmap(cmap).reversed()


def _outlier_mask(result):
    if "cluster_id" in result:
        return np.asarray(result["cluster_id"]) < 0
    return np.asarray(result["member_prob"]) < 0.5


def _member_prob(result):
    return np.clip(np.asarray(result["member_prob"], dtype=float), 0.0, 1.0)


def _membership_mappable(cmap=None):
    return ScalarMappable(norm=Normalize(0.0, 1.0), cmap=_membership_display_cmap(cmap))


def _scatter_membership(ax, x, y, result, s=50, cmap=None, zorder=3, outlier_color=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = _member_prob(result)
    mark = _outlier_color(outlier_color)
    sc = ax.scatter(
        x, y, c=p, cmap=_membership_display_cmap(cmap),
        vmin=0.0, vmax=1.0, s=s, linewidths=0, zorder=zorder,
    )
    out = _outlier_mask(result)
    if np.any(out):
        ax.scatter(
            x[out], y[out],
            s=max(0.25 * s, 5.0),
            facecolors="white",
            edgecolors=mark,
            linewidths=1, zorder=zorder + 1,
        )
    return sc


def _plotly_membership_colorscale(cmap=None, n=32):
    cm = _membership_display_cmap(cmap)
    xs = np.linspace(0.0, 1.0, n)
    scale = []
    for x, (r, g, b, _) in zip(xs, cm(xs)):
        scale.append([
            float(x),
            "rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255)),
        ])
    return scale


def _add_membership_colorbar(obj, cmap=None, ax=None, **kwargs):
    kwargs.setdefault("label", r"$P(\mathrm{member})$")
    mappable = _membership_mappable(cmap)
    mappable.set_array([])
    if ax is None:
        return obj.colorbar(mappable, **kwargs)
    return plt.colorbar(mappable, ax=ax, **kwargs)


def _membership_histogram(ax, x, result, edges, cmap=None, outlier_color=None):
    x = np.asarray(x, dtype=float)
    p = _member_prob(result)
    out = _outlier_mask(result)
    finite = np.isfinite(x)
    x, p, out = x[finite], p[finite], out[finite]
    if x.size == 0:
        return
    mark = _outlier_color(outlier_color)
    cm = _membership_display_cmap(cmap)
    colors = cm(p)
    n_bins = len(edges) - 1
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ymax = 0
    hatch_rc = {"hatch.linewidth": 0.7}
    try:
        hatch_rc["hatch.color"] = mark
    except Exception:
        pass
    with plt.rc_context(hatch_rc):
        for b in range(n_bins):
            in_bin = np.flatnonzero(idx == b)
            if in_bin.size == 0:
                continue
            in_bin = in_bin[np.argsort(p[in_bin])]
            left, width = edges[b], edges[b + 1] - edges[b]
            y = 0.0
            for k in in_bin:
                ax.add_patch(Rectangle(
                    (left, y), width, 1.0,
                    facecolor=colors[k],
                    edgecolor="none",
                    linewidth=0,
                    zorder=2,
                ))
                if out[k]:
                    ax.add_patch(Rectangle(
                        (left, y), width, 1.0,
                        facecolor="none",
                        edgecolor=mark,
                        hatch="///",
                        linewidth=0.35,
                        zorder=3,
                    ))
                y += 1.0
            ax.add_patch(Rectangle(
                (left, 0.0), width, y,
                fill=False, edgecolor="0.25", linewidth=0.4, zorder=4,
            ))
            ymax = max(ymax, y)
    ax.set_ylim(0.0, max(ymax, 1.0) * 1.06)


def _hover_names(df, n=None):
    if n is None:
        n = len(df)
    for col in ("name", "input_name", "target_id", "designation", "simbad_main_id"):
        if col in df.columns:
            return np.asarray(df[col], dtype=str)
    if "source_id" in df.columns:
        return np.array(["Gaia DR3 {}".format(i) for i in df["source_id"]])
    return np.array(["star {}".format(i) for i in range(n)])


def _sky_space_coordinates(df, use_rv=None):
    ra = np.asarray(df["ra"], dtype=float)
    dec = np.asarray(df["dec"], dtype=float)
    plx = np.asarray(df["parallax"], dtype=float)
    dist = 1.0 / plx
    va = AU_KMS * np.asarray(df["pmra"], dtype=float) / plx
    vd = AU_KMS * np.asarray(df["pmdec"], dtype=float) / plx
    if use_rv is None:
        use_rv = star_uses_rv(df)
    if "radial_velocity" in df.columns:
        vr = np.asarray(df["radial_velocity"], dtype=float)
    else:
        vr = np.full(len(df), np.nan)
    vr = np.where(use_rv, vr, np.nan)
    data = np.column_stack([ra, dec, dist, va, vd, vr])
    labels = [
        r"$\alpha$ [deg]",
        r"$\delta$ [deg]",
        r"$d=1/\pi$ [kpc]",
        r"$v_{\alpha*}$ [km/s]",
        r"$v_{\delta}$ [km/s]",
        r"$v_r$ [km/s]",
    ]
    return data, labels


def _observable_coordinates(df, use_rv=None):
    data = np.column_stack([
        np.asarray(df["ra"], dtype=float),
        np.asarray(df["dec"], dtype=float),
        np.asarray(df["parallax"], dtype=float),
        np.asarray(df["pmra"], dtype=float),
        np.asarray(df["pmdec"], dtype=float),
    ])
    labels = [
        r"$\alpha$ [deg]",
        r"$\delta$ [deg]",
        r"$\pi$ [mas]",
        r"$\mu_{\alpha*}$ [mas/yr]",
        r"$\mu_{\delta}$ [mas/yr]",
    ]
    if use_rv is None:
        use_rv = star_uses_rv(df)
    if np.any(use_rv) and "radial_velocity" in df.columns:
        vr = np.asarray(df["radial_velocity"], dtype=float)
        vr = np.where(use_rv, vr, np.nan)
        data = np.column_stack([data, vr])
        labels.append(r"$v_r$ [km/s]")
    return data, labels


def plot_sky_membership(df, result, true_labels=None, ax=None, cmap=None, colorbar=None, outlier_color=None):
    """Sky positions coloured by :math:`P(\\mathrm{member})`.

    Parameters
    ----------
    df : DataFrame
        Must contain ``ra`` and ``dec`` (degrees).
    result : dict
        Clustering ``best`` result with ``member_prob`` and ``cluster_id``.
    true_labels : array-like, optional
        If given, true field stars (label ``< 0``) are circled.
    ax : matplotlib.axes.Axes, optional
    cmap, outlier_color
        Override the module defaults.
    colorbar : bool, optional
        Default True when a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
        if colorbar is None:
            colorbar = True
    _scatter_membership(
        ax, df["ra"], df["dec"], result, cmap=cmap, outlier_color=outlier_color,
    )
    if true_labels is not None:
        field = np.asarray(true_labels) < 0
        ax.scatter(
            np.asarray(df["ra"])[field],
            np.asarray(df["dec"])[field],
            facecolors="none",
            edgecolors="k",
            s=90,
            linewidths=0.8,
            label="true field",
            zorder=2,
        )
    ax.set_xlabel(r"$\alpha$ [deg]", fontsize=fs_label)
    ax.set_ylabel(r"$\delta$ [deg]", fontsize=fs_label)
    ax.set_title("Sky clustering ($K={}$)".format(result["n_clusters"]), fontsize=fs_title)
    ax.tick_params(labelsize=fs_text)
    ax.invert_xaxis()
    ax.set_aspect("equal", adjustable="datalim")
    if colorbar:
        _add_membership_colorbar(plt, cmap=cmap, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_proper_motion_membership(df, result, ax=None, cmap=None, colorbar=None, outlier_color=None):
    """Tangential velocities :math:`(v_{\\alpha*}, v_{\\delta})` with Gaia error bars.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
        if colorbar is None:
            colorbar = True
    va = AU_KMS * np.asarray(df["pmra"], dtype=float) / np.asarray(df["parallax"], dtype=float)
    vd = AU_KMS * np.asarray(df["pmdec"], dtype=float) / np.asarray(df["parallax"], dtype=float)
    eva = AU_KMS * np.asarray(df["pmra_error"], dtype=float) / np.asarray(df["parallax"], dtype=float)
    evd = AU_KMS * np.asarray(df["pmdec_error"], dtype=float) / np.asarray(df["parallax"], dtype=float)
    ax.errorbar(
        va, vd, xerr=eva, yerr=evd,
        fmt="none", ecolor="0.75", elinewidth=0.8, zorder=1,
    )
    _scatter_membership(ax, va, vd, result, cmap=cmap, outlier_color=outlier_color)
    ax.set_xlabel(r"$v_{\alpha*}$ [km/s]", fontsize=fs_label)
    ax.set_ylabel(r"$v_{\delta}$ [km/s]", fontsize=fs_label)
    ax.set_title("Tangential-velocity clustering", fontsize=fs_title)
    ax.tick_params(labelsize=fs_text)
    ax.set_aspect("equal", adjustable="datalim")
    if colorbar:
        _add_membership_colorbar(plt, cmap=cmap, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_bic_comparison(comparison, ax=None):
    """Bar chart of spatial-:math:`K` BIC values; the winner is highlighted.

    Parameters
    ----------
    comparison : pandas.DataFrame
        Output of :func:`~gaia_clustering.xd.compare_cluster_counts`.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if comparison is None:
        raise ValueError("No spatial-K comparison to plot")
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.2))
    tab = comparison.sort_values("K")
    colors = ["C0" if k == comparison.loc[0, "K"] else "0.75" for k in tab["K"]]
    ax.bar(tab["K"].astype(str), tab["BIC"], color=colors)
    ax.set_xlabel(r"spatial clusters $K$ among members", fontsize=fs_label)
    ax.set_ylabel("BIC", fontsize=fs_label)
    ax.set_title("Lobe-count comparison", fontsize=fs_title)
    ax.tick_params(labelsize=fs_text)
    return ax


def plot_membership_summary(df, fit, figsize=(15, 4.8), cmap=None, outlier_color=None):
    """Sky, tangential-velocity, and spatial-:math:`K` BIC in one figure.

    Parameters
    ----------
    df : DataFrame
    fit : dict
        Full :func:`~gaia_clustering.clustering.cluster_association` output.
    figsize : tuple of float
    cmap, outlier_color
        Membership styling.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_sky_membership(
        df, fit["best"], ax=axes[0], cmap=cmap, outlier_color=outlier_color, colorbar=False,
    )
    plot_proper_motion_membership(
        df, fit["best"], ax=axes[1], cmap=cmap, outlier_color=outlier_color, colorbar=True,
    )
    if fit.get("comparison") is not None:
        plot_bic_comparison(fit["comparison"], ax=axes[2])
    else:
        axes[2].text(0.5, 0.5, "too few members\nfor spatial $K$", ha="center", va="center")
        axes[2].set_axis_off()
    fig.tight_layout()
    return fig, axes


def plot_association_corner(
    df, result, bins=12, title=None, cmap=None, outlier_color=None,
):
    """Corner of :math:`(\\alpha, \\delta, \\pi, \\mu_{\\alpha*}, \\mu_{\\delta}[, v_r])`.

    Points and histogram slices are coloured by :math:`P(\\mathrm{member})`.
    :math:`v_r` is included only for stars that contribute radial velocity;
    otherwise the figure is the 5D astrometric corner.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data, labels = _observable_coordinates(df, use_rv=star_uses_rv(df, result))
    n = data.shape[1]
    mark = _outlier_color(outlier_color)

    pad = 0.08
    ranges = []
    for k in range(n):
        x = data[:, k]
        x = x[np.isfinite(x)]
        lo, hi = np.min(x), np.max(x)
        extra = pad * (hi - lo) if hi > lo else 1.0
        ranges.append((lo - extra, hi + extra))

    fig, axes = plt.subplots(
        n, n, figsize=(2.15 * n, 2.15 * n),
        gridspec_kw={"hspace": 0.04, "wspace": 0.04},
    )
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=mark, markersize=5,
            label="outlier",
        ),
        Patch(
            facecolor="none", edgecolor=mark, hatch="///",
            label="outlier (histogram)",
        ),
    ]

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i < j:
                ax.set_visible(False)
                continue
            if i == j:
                edges = np.linspace(ranges[j][0], ranges[j][1], bins + 1)
                _membership_histogram(
                    ax, data[:, j], result, edges, cmap=cmap, outlier_color=mark,
                )
                ax.set_xlim(ranges[j])
            else:
                _scatter_membership(
                    ax, data[:, j], data[:, i], result, s=18, cmap=cmap,
                    outlier_color=mark,
                )
                ax.set_xlim(ranges[j])
                ax.set_ylim(ranges[i])

            if i < n - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(labels[j], fontsize=fs_label)
            if j > 0:
                ax.tick_params(labelleft=False)
            elif i > 0:
                ax.set_ylabel(labels[i], fontsize=fs_label)
            ax.tick_params(labelsize=fs_text, direction="in", top=True, right=True)

    if title is None:
        if data.shape[1] == 6:
            title = r"Association vs field in $(\alpha,\delta,\pi,\mu_{\alpha*},\mu_{\delta},v_r)$"
        else:
            title = r"Association vs field in $(\alpha,\delta,\pi,\mu_{\alpha*},\mu_{\delta})$"
    _add_membership_colorbar(
        fig, cmap=cmap, ax=axes, fraction=0.03, pad=0.02, shrink=0.55,
    )
    fig.legend(
        handles=legend_handles, loc="upper right", fontsize=fs_text,
        frameon=False, bbox_to_anchor=(0.98, 0.98),
    )
    fig.suptitle(title, fontsize=fs_title, y=0.995)
    return fig


def _elev_azim_camera(elev, azim, r=1.8):
    elev_r = np.radians(elev)
    azim_r = np.radians(azim)
    return dict(
        eye=dict(
            x=float(r * np.cos(elev_r) * np.cos(azim_r)),
            y=float(r * np.cos(elev_r) * np.sin(azim_r)),
            z=float(r * np.sin(elev_r)),
        )
    )


def _plotly_camera_from_view(view):
    if view is None:
        return None
    if isinstance(view, dict):
        if any(k in view for k in ("eye", "center", "up")):
            return view
        return _elev_azim_camera(
            view.get("elev", 30.0),
            view.get("azim", -60.0),
            view.get("distance", view.get("r", 1.8)),
        )
    seq = tuple(view)
    if len(seq) == 2:
        elev, azim = seq
        r = 1.8
    elif len(seq) == 3:
        elev, azim, r = seq
    else:
        raise ValueError("view must be (elev, azim), (elev, azim, distance), or a dict")
    return _elev_azim_camera(float(elev), float(azim), float(r))


def _arrowhead_mesh3d(
    origins, dpos, span, color, head_size=0.02, head_width=0.008, n_sides=8,
):
    origins = np.asarray(origins, dtype=float)
    dpos = np.asarray(dpos, dtype=float)
    span = np.asarray(span, dtype=float)
    d_vis = dpos / span
    length = np.linalg.norm(d_vis, axis=1)
    keep = np.isfinite(length) & (length > 1e-12)
    if not np.any(keep):
        return None
    origins = origins[keep]
    d_vis = d_vis[keep]
    length = length[keep]
    u = d_vis / length[:, None]
    helper = np.zeros_like(u)
    helper[:, 0] = 1.0
    helper[np.abs(u[:, 0]) > 0.9] = (0.0, 1.0, 0.0)
    n1 = np.cross(u, helper)
    n1 /= np.clip(np.linalg.norm(n1, axis=1, keepdims=True), 1e-15, None)
    n2 = np.cross(u, n1)
    n = origins.shape[0]
    tip = d_vis
    base = d_vis - u * head_size
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    ring = (
        base[:, None, :]
        + head_width * (
            np.cos(angles)[None, :, None] * n1[:, None, :]
            + np.sin(angles)[None, :, None] * n2[:, None, :]
        )
    )
    verts_vis = np.concatenate([tip[:, None, :], ring], axis=1)
    verts = (origins[:, None, :] + verts_vis * span).reshape(-1, 3)
    faces = []
    for i in range(n):
        t = i * (n_sides + 1)
        for k in range(n_sides):
            faces.append((t, t + 1 + k, t + 1 + (k + 1) % n_sides))
        for k in range(1, n_sides - 1):
            faces.append((t + 1, t + 1 + k, t + 2 + k))
    faces = np.asarray(faces, dtype=int)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color,
        opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=1, diffuse=0.15, specular=0, roughness=1, fresnel=0),
        lightposition=dict(x=0, y=0, z=1e5),
        hoverinfo="skip",
        showlegend=False,
        showscale=False,
    )


def _nice_scale_value(v):
    v = float(v)
    if not np.isfinite(v) or v <= 0:
        return 1.0
    exp = np.floor(np.log10(v))
    frac = v / 10.0 ** exp
    if frac < 1.5:
        nice = 1.0
    elif frac < 3.5:
        nice = 2.0
    elif frac < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * 10.0 ** exp


def _camera_distance(camera):
    if not camera:
        return 1.8
    eye = camera.get("eye") if isinstance(camera, dict) else None
    if not eye:
        return 1.8
    x, y, z = float(eye.get("x", 0.0)), float(eye.get("y", 0.0)), float(eye.get("z", 0.0))
    r = float(np.sqrt(x * x + y * y + z * z))
    return r if r > 1e-9 else 1.8


def _velocity_scale_zoom_js(vis_frac, r0, label):
    return """
(function() {{
  var gd = document.getElementById('{{plot_id}}');
  var visFrac = {vis_frac};
  var r0 = {r0};
  var label = {label};
  var minPx = 24;
  var maxPx = 320;
  var lastPx = null;
  var r0Live = null;
  var a0Live = 1;
  var overlay = null;
  var shaft = null;
  var head = null;

  function hypot3(x, y, z) {{
    var r = Math.sqrt((+x || 0) * (+x || 0) + (+y || 0) * (+y || 0) + (+z || 0) * (+z || 0));
    return r > 1e-9 ? r : null;
  }}
  function sceneObj(gd) {{
    return gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
  }}
  function liveZoom(gd) {{
    var r = r0;
    var a = 1;
    var scene = sceneObj(gd);
    if (scene) {{
      try {{
        if (typeof scene.getCamera === 'function') {{
          var cam = scene.getCamera();
          var er = cam && cam.eye && hypot3(cam.eye.x, cam.eye.y, cam.eye.z);
          if (er) r = er;
        }}
      }} catch (err) {{}}
    }}
    if (r0Live === null) {{
      r0Live = r;
      a0Live = a || 1;
    }}
    return (r0Live / r) * (a / a0Live);
  }}
  function sceneWH(gd) {{
    var sc = sceneObj(gd);
    if (sc && sc.container) {{
      return {{w: sc.container.clientWidth, h: sc.container.clientHeight}};
    }}
    return {{w: gd.clientWidth || 700, h: 0.87 * (gd.clientHeight || 650)}};
  }}
  function arrowPx(gd) {{
    var s = sceneWH(gd);
    var cube = 0.60 * Math.min(s.w, s.h) * liveZoom(gd);
    var px = visFrac * cube;
    if (px < minPx) px = minPx;
    if (px > maxPx) px = maxPx;
    if (px > 0.40 * s.w) px = 0.40 * s.w;
    return px;
  }}
  function ensureOverlay(gd) {{
    if (overlay && overlay.parentNode === gd) return overlay;
    var id = 'gaia-vel-scale-' + gd.id;
    overlay = document.getElementById(id);
    if (overlay) {{
      shaft = overlay.querySelector('line');
      head = overlay.querySelector('polygon');
      if (overlay.parentNode !== gd) gd.appendChild(overlay);
      return overlay;
    }}
    if (!gd.style.position || gd.style.position === 'static') {{
      gd.style.position = 'relative';
    }}
    overlay = document.createElement('div');
    overlay.id = id;
    overlay.style.cssText = [
      'position:absolute','left:0','right:0','bottom:0','height:11%',
      'display:flex','align-items:center','justify-content:flex-end',
      'padding-right:6%','gap:8px','pointer-events:none','z-index:30',
      'box-sizing:border-box'
    ].join(';');
    overlay.innerHTML = [
      '<svg width="80" height="24" style="overflow:visible;flex:0 0 auto">',
      '  <line x1="0" y1="12" x2="68" y2="12" stroke="#222222" stroke-width="2.5"',
      '        stroke-linecap="butt"></line>',
      '  <polygon points="68,6 80,12 68,18" fill="#222222"></polygon>',
      '</svg>',
      '<span style="font:13px sans-serif;color:#222;background:rgba(255,255,255,0.92);',
      'border:1px solid rgba(34,34,34,0.22);border-radius:3px;padding:3px 8px;">',
      label, '</span>'
    ].join('');
    shaft = overlay.querySelector('line');
    head = overlay.querySelector('polygon');
    gd.appendChild(overlay);
    return overlay;
  }}
  function draw(px) {{
    if (!overlay || !shaft || !head) return;
    var svg = overlay.querySelector('svg');
    var headW = 12;
    var y = 12;
    var x2 = Math.max(px - headW, 1);
    svg.setAttribute('width', String(px + 2));
    shaft.setAttribute('x2', String(x2));
    head.setAttribute('points',
      x2 + ',' + (y - 6) + ' ' + px + ',' + y + ' ' + x2 + ',' + (y + 6));
  }}
  function apply(gd) {{
    if (!gd) return;
    ensureOverlay(gd);
    var px = arrowPx(gd);
    if (lastPx !== null && Math.abs(px - lastPx) < 0.5) return;
    lastPx = px;
    draw(px);
  }}
  function hook(gd) {{
    if (!gd || gd._gaiaVelScaleHooked) return;
    gd._gaiaVelScaleHooked = true;
    apply(gd);
    function loop() {{
      if (!document.body.contains(gd)) return;
      apply(gd);
      window.requestAnimationFrame(loop);
    }}
    window.requestAnimationFrame(loop);
  }}
  if (gd && (gd.on || gd.layout)) hook(gd);
  else {{
    var n = 0;
    var t = setInterval(function() {{
      gd = document.getElementById('{{plot_id}}');
      n += 1;
      if (gd && (gd.on || gd.layout)) {{ clearInterval(t); hook(gd); }}
      if (n > 200) clearInterval(t);
    }}, 50);
  }}
}})();
""".format(
        vis_frac=float(vis_frac),
        r0=float(r0),
        label='"{}"'.format(label.replace('"', '\\"')),
    )


class _Association3dFigure(go.Figure):
    def __init__(self, *args, scale_js=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._scale_js = scale_js

    def to_html(self, *args, **kwargs):
        cfg = dict(kwargs.get("config") or {})
        cfg.setdefault("scrollZoom", True)
        kwargs["config"] = cfg
        if self._scale_js:
            extra = kwargs.get("post_script")
            if extra is None:
                kwargs["post_script"] = self._scale_js
            elif isinstance(extra, (list, tuple)):
                kwargs["post_script"] = list(extra) + [self._scale_js]
            else:
                kwargs["post_script"] = [extra, self._scale_js]
        return super().to_html(*args, **kwargs)

    def show(self, *args, **kwargs):
        if self._scale_js and kwargs.get("post_script") is None:
            kwargs["post_script"] = self._scale_js
        return super().show(*args, **kwargs)

    def _repr_html_(self):
        return self.to_html(full_html=False, include_plotlyjs="cdn")

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs):
        return {"text/html": self._repr_html_()}

    def _ipython_display_(self):
        from IPython.display import display, HTML
        display(HTML(self._repr_html_()))


def _association_3d_traces(
    df, result, residual=True, arrow_frac=0.15, show_vectors=True,
    head_size=0.02, cmap=None, outlier_color=None, global_span=None, v0=None,
):
    """Build Plotly traces and metadata for one traceback epoch."""
    use_rv = star_uses_rv(df, result)
    coords, _ = _sky_space_coordinates(df, use_rv=use_rv)
    ra, dec, dist, va, vd, vr = (coords[:, k] for k in range(6))
    cluster_id = np.asarray(result["cluster_id"])
    p = _member_prob(result)
    out = _outlier_mask(result)
    mark = to_hex(_outlier_color(outlier_color))
    vel = np.column_stack([va, vd, vr])

    vel_plot = vel.copy()
    if residual:
        member = cluster_id >= 0
        src = vel[member] if np.any(member) else vel
        if v0 is None:
            v0 = np.zeros(3)
            for k in range(3):
                col = src[:, k]
                finite = np.isfinite(col)
                if np.any(finite):
                    v0[k] = col[finite].mean()
        vel_plot = vel - v0
    vel_plot = np.where(np.isfinite(vel_plot), vel_plot, 0.0)

    span = np.array([np.ptp(ra), np.ptp(dec), np.ptp(dist)])
    span = np.maximum(span, 1e-9)
    if global_span is not None:
        span = np.maximum(np.asarray(global_span, dtype=float), 1e-9)
    rms = np.sqrt(np.mean(vel_plot ** 2, axis=0))
    rms = np.maximum(rms, 1e-9)
    dpos = vel_plot / rms * arrow_frac * span
    d_vis = dpos / span
    vis_len = np.linalg.norm(d_vis, axis=1)
    vis_len = np.maximum(vis_len, 1e-15)
    u_vis = d_vis / vis_len[:, None]
    shaft_len = np.maximum(vis_len - head_size, 0.0)
    shaft = (u_vis * shaft_len[:, None]) * span
    head_width = 0.4 * head_size

    hover_names = _hover_names(df, n=len(ra))
    vr_hover = np.where(use_rv, vr, np.nan)

    traces = [
        go.Scatter3d(
            x=ra, y=dec, z=dist,
            mode="markers",
            marker=dict(
                size=5,
                color=p,
                colorscale=_plotly_membership_colorscale(cmap),
                cmin=0.0, cmax=1.0,
                colorbar=dict(title="P(member)", thickness=14, len=0.55, y=0.62),
            ),
            name="stars",
            showlegend=False,
            text=hover_names,
            customdata=np.column_stack([va, vd, vr_hover, cluster_id, p]),
            hovertemplate=(
                "%{text}<br>"
                "α = %{x:.3f} deg<br>"
                "δ = %{y:.3f} deg<br>"
                "d = 1/π = %{z:.3f} kpc<br>"
                "v<sub>α*</sub> = %{customdata[0]:.2f} km/s<br>"
                "v<sub>δ</sub> = %{customdata[1]:.2f} km/s<br>"
                "v<sub>r</sub> = %{customdata[2]:.2f} km/s<br>"
                "cluster = %{customdata[3]}<br>"
                "P(member) = %{customdata[4]:.3f}<extra></extra>"
            ),
        )
    ]
    if np.any(out):
        traces.append(go.Scatter3d(
            x=ra[out], y=dec[out], z=dist[out],
            mode="markers",
            marker=dict(size=2.2, color=mark),
            name="outlier",
            hoverinfo="skip",
        ))
    else:
        traces.append(go.Scatter3d(
            x=[ra[0]], y=[dec[0]], z=[dist[0]],
            mode="markers",
            marker=dict(size=0.1, color=mark, opacity=0),
            name="outlier",
            hoverinfo="skip",
            showlegend=False,
        ))
    if show_vectors:
        xs, ys, zs = [], [], []
        for i in range(len(ra)):
            xs += [ra[i], ra[i] + shaft[i, 0], None]
            ys += [dec[i], dec[i] + shaft[i, 1], None]
            zs += [dist[i], dist[i] + shaft[i, 2], None]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color="#737373", width=4),
            name="velocity",
            showlegend=False,
            hoverinfo="skip",
        ))
        heads = _arrowhead_mesh3d(
            np.column_stack([ra, dec, dist]), dpos, span, "#737373",
            head_size=head_size, head_width=head_width,
        )
        if heads is not None:
            traces.append(heads)
    return traces, rms, span, v0


def _traceback_times(traceback, n_frames):
    if traceback is None:
        return [0.0]
    if np.isscalar(traceback):
        return [float(traceback)]
    seq = tuple(traceback)
    if len(seq) == 0:
        return [0.0]
    if len(seq) == 1:
        return [float(seq[0])]
    t0, t1 = float(seq[0]), float(seq[1])
    n = max(int(n_frames), 2)
    return list(np.linspace(t0, t1, n))


def plot_association_3d(
    df, result, residual=True, arrow_frac=0.15, title=None,
    show_vectors=True, head_size=0.02, view=(90, 210), traceback=None,
    n_frames=9, figsize=None, cmap=None, outlier_color=None,
):
    """Interactive 3D sky plot: :math:`(\\alpha, \\delta, 1/\\pi)` with velocity arrows.

    Arrow length is a visual scale, not a physical displacement. Residual
    velocities (default) subtract the member mean so bulk motion does not
    dominate. ``traceback`` is lookback time in Myr: a scalar shows one
    epoch; ``(t0, t1)`` adds a slider so the association can be coasted
    at constant ICRS velocity, :math:`r \\to r - v t`.

    Parameters
    ----------
    df : DataFrame
    result : dict
        Clustering ``best`` result.
    residual : bool
        Subtract the member-mean velocity before drawing arrows.
    arrow_frac : float
        Arrow length as a fraction of the axis span at the RMS speed.
    traceback : float or tuple of float, optional
        Lookback time (Myr).
    n_frames : int
        Number of slider frames when ``traceback`` is a range.
    view : tuple
        Initial camera ``(elev, azim)`` in degrees (matplotlib convention).
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    times = _traceback_times(traceback, n_frames)
    camera = _plotly_camera_from_view(view)
    if figsize is None:
        figsize = (6.5, 7.0)

    # Fixed axis span and residual zero-point from the present-day frame
    # so the slider does not jump the camera or residual arrows.
    traces0, rms0, span0, v0 = _association_3d_traces(
        df, result, residual=residual, arrow_frac=arrow_frac,
        show_vectors=show_vectors, head_size=head_size, cmap=cmap,
        outlier_color=outlier_color,
    )

    frame_traces = []
    for t in times:
        dft = coast_dataframe(df, t) if t else df
        traces, rms, _, _ = _association_3d_traces(
            dft, result, residual=residual, arrow_frac=arrow_frac,
            show_vectors=show_vectors, head_size=head_size, cmap=cmap,
            outlier_color=outlier_color, global_span=span0, v0=v0,
        )
        frame_traces.append((t, traces, rms))

    fig = _Association3dFigure(data=frame_traces[0][1])

    if title is None:
        title = "Association on the sky"
        if show_vectors:
            title += " with space-velocity vectors"
            if residual:
                title += " (residual velocities)"
        if len(times) == 1 and times[0]:
            t = times[0]
            if t > 0:
                title += ", {:g} Myr ago".format(t)
            else:
                title += ", {:g} Myr from now".format(-t)

    scene = dict(
        xaxis_title="α [deg]",
        yaxis_title="δ [deg]",
        zaxis_title="d = 1/π [kpc]",
        xaxis=dict(autorange="reversed"),
        aspectmode="cube",
        domain=dict(x=[0.0, 1.0], y=[0.12, 1.0]),
    )
    if camera is not None:
        scene["camera"] = camera
    width_in, height_in = figsize
    layout = dict(
        title=title,
        scene=scene,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, t=50, b=0),
        width=int(round(float(width_in) * 100)),
        height=int(round(float(height_in) * 100)),
        autosize=False,
    )
    if show_vectors:
        r0 = _camera_distance(camera)
        v_ref = _nice_scale_value(float(rms0[0]))
        vis_frac = arrow_frac * v_ref / max(float(rms0[0]), 1e-15)
        vis_frac = float(np.clip(vis_frac, 1e-6, 0.40))
        fig._scale_js = _velocity_scale_zoom_js(vis_frac, r0, "{:g} km/s".format(v_ref))

    if len(times) > 1:
        fig.frames = [
            go.Frame(data=traces, name="{:.2f}".format(t))
            for t, traces, _ in frame_traces
        ]
        steps = []
        for t, _, _ in frame_traces:
            label = "{:g}".format(t)
            steps.append(dict(
                args=[["{:.2f}".format(t)], dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                )],
                label=label,
                method="animate",
            ))
        layout["sliders"] = [dict(
            active=0,
            currentvalue=dict(prefix="lookback ", suffix=" Myr", xanchor="right"),
            pad=dict(t=30, b=10),
            steps=steps,
            x=0.12, y=0.0, len=0.70,
        )]
        layout["updatemenus"] = [dict(
            type="buttons",
            showactive=False,
            x=0.02, y=0.02,
            xanchor="left", yanchor="bottom",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=400, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))],
                ),
            ],
        )]
        layout["margin"] = dict(l=0, r=0, t=50, b=80)

    fig.update_layout(**layout)
    return fig


def _format_sigma_label(nsigma):
    nsigma = float(nsigma)
    if nsigma.is_integer():
        return "{:.0f}σ".format(nsigma)
    return "{:g}σ".format(nsigma)


def _axis_aligned_error_bars_mesh(x, y, z, dx, dy, dz, radius):
    x, y, z = (np.asarray(a, dtype=float).ravel() for a in (x, y, z))
    dx, dy, dz = (np.asarray(a, dtype=float).ravel() for a in (dx, dy, dz))
    r = float(radius)
    bars = (
        (x - dx, x + dx, y, y, z, z, np.array([0.0, r, 0.0]), np.array([0.0, 0.0, r])),
        (x, x, y - dy, y + dy, z, z, np.array([r, 0.0, 0.0]), np.array([0.0, 0.0, r])),
        (x, x, y, y, z - dz, z + dz, np.array([r, 0.0, 0.0]), np.array([0.0, r, 0.0])),
    )
    n = int(x.size)
    n_bars = 3 * n
    verts = np.empty((n_bars * 8, 3), dtype=float)
    faces = np.empty((n_bars * 12, 3), dtype=int)
    local_faces = np.array(
        [
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        ],
        dtype=int,
    )
    b = 0
    for x0, x1, y0, y1, z0, z1, u, v in bars:
        p0 = np.column_stack([x0, y0, z0])
        p1 = np.column_stack([x1, y1, z1])
        for i in range(n):
            c0, c1 = p0[i], p1[i]
            base = 8 * b
            verts[base: base + 8] = (
                c0 + u + v, c0 + u - v, c0 - u - v, c0 - u + v,
                c1 + u + v, c1 + u - v, c1 - u - v, c1 - u + v,
            )
            faces[12 * b: 12 * (b + 1)] = local_faces + base
            b += 1
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="black", opacity=1.0, flatshading=True,
        lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1, fresnel=0),
        lightposition=dict(x=0, y=0, z=1e5),
        hoverinfo="skip", name="1σ errors", showlegend=True, showscale=False,
    )


def plot_velocity_gaussian(result, nsigma=(1, 2, 3), n_mesh=40):
    """Interactive ellipsoid of the fitted velocity Gaussian.

    Member velocities are overplotted. If the fit is 2D (no RVs), the
    :math:`v_r` axis is shown only for display.

    Parameters
    ----------
    result : dict
        Output of :func:`~gaia_clustering.velocity.fit_velocity_dispersion`.
    nsigma : float or sequence of float
        Mahalanobis contours of :math:`N(\\mu, \\Sigma)`.
    n_mesh : int
        Ellipsoid mesh resolution.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    v_obs = np.asarray(result["v_obs"], dtype=float)
    n_vel = int(result.get("n_vel", v_obs.shape[1]))
    mean = np.asarray(result["mean"], dtype=float)
    cov_pop = np.asarray(result["cov"], dtype=float)
    if n_vel == 2:
        v_obs3 = np.column_stack([v_obs, np.zeros(len(v_obs))])
        mean3 = np.array([mean[0], mean[1], 0.0])
        cov3 = np.eye(3) * 1e-6
        cov3[:2, :2] = cov_pop
        labels = ("vα [km/s]", "vδ [km/s]", "vr (unconstrained)")
    else:
        v_obs3 = np.where(np.isfinite(v_obs), v_obs, np.nan)
        # Stars without RV: plot at the posterior mean vr so they sit in the vα–vδ plane of the ellipsoid.
        if v_obs3.shape[1] == 3:
            missing = ~np.isfinite(v_obs3[:, 2])
            v_obs3[missing, 2] = mean[2]
        mean3 = mean
        cov3 = cov_pop
        labels = ("vα [km/s]", "vδ [km/s]", "vr [km/s]")

    xyz = v_obs3
    levels = np.unique(np.atleast_1d(np.asarray(nsigma, dtype=float)).ravel())
    if levels.size == 1:
        opacities = {float(levels[0]): 0.35}
    else:
        lo, hi = float(levels.min()), float(levels.max())
        opacities = {
            float(s): 0.12 + 0.43 * (hi - float(s)) / (hi - lo)
            for s in levels
        }
    evals, evecs = np.linalg.eigh(cov3)
    axis_1sigma = np.sqrt(np.clip(evals, 0.0, None))
    u = np.linspace(0.0, 2.0 * np.pi, n_mesh)
    v = np.linspace(0.0, np.pi, n_mesh)
    sphere = np.stack(
        [
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        ],
        axis=0,
    )
    sphere_flat = sphere.reshape(3, -1)

    def _ellipsoid_at(ns):
        radii = ns * axis_1sigma
        ell = ((evecs * radii) @ sphere_flat).reshape(3, n_mesh, n_mesh)
        ell += mean3[:, None, None]
        return ell

    ellipsoids = [_ellipsoid_at(float(ns)) for ns in np.sort(levels)[::-1]]
    x, y, z = xyz.T
    pts = np.vstack([xyz, mean3] + [ell.reshape(3, -1).T for ell in ellipsoids])
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = float(np.max(hi - lo))
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    mid = 0.5 * (hi + lo)
    lo_lim = mid - 0.55 * span
    hi_lim = mid + 0.55 * span

    fig = go.Figure()
    for ns, ellipsoid in zip(np.sort(levels)[::-1], ellipsoids):
        fig.add_trace(go.Surface(
            x=ellipsoid[0], y=ellipsoid[1], z=ellipsoid[2],
            opacity=opacities[float(ns)], showscale=False,
            colorscale=[[0, "royalblue"], [1, "royalblue"]],
            name="{} ellipsoid".format(_format_sigma_label(ns)),
            hoverinfo="skip", showlegend=True,
        ))
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=5, color="black"),
        name="Member stars",
    ))
    fig.add_trace(go.Scatter3d(
        x=[mean3[0]], y=[mean3[1]], z=[mean3[2]],
        mode="markers",
        marker=dict(size=7, color="royalblue"),
        name="Posterior mean μ",
    ))
    sigma_title = ", ".join(_format_sigma_label(s) for s in np.sort(levels))
    fig.update_layout(
        title="Association velocity Gaussian ({} ellipsoids)".format(sigma_title),
        scene=dict(
            xaxis_title=labels[0],
            yaxis_title=labels[1],
            zaxis_title=labels[2],
            aspectmode="cube",
            xaxis=dict(range=[lo_lim[0], hi_lim[0]]),
            yaxis=dict(range=[lo_lim[1], hi_lim[1]]),
            zaxis=dict(range=[lo_lim[2], hi_lim[2]]),
        ),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, t=50, b=0),
        height=700,
    )
    return fig


def plot_velocity_corner(result, include_std=True, include_corr=True, **corner_kwargs):
    """Corner plot of the population velocity posterior.

    Always includes :math:`(v_\\alpha, v_\\delta[, v_r])`. Optional panels
    append component dispersions and unique correlations.

    Parameters
    ----------
    result : dict
        Must contain ``trace`` from NUTS sampling.
    include_std, include_corr : bool
        Extra :math:`\\sigma` and :math:`\\rho` panels.
    **corner_kwargs
        Forwarded to :func:`corner.corner` (e.g. ``truths``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import corner

    if "trace" not in result:
        raise ValueError("result has no 'trace'; sample with return_inferencedata=True")
    post = result["trace"].posterior
    n_vel = int(result.get("n_vel", 3))
    mu = np.asarray(post["Venv"].values).reshape(-1, n_vel)
    std = np.asarray(post["stds"].values).reshape(-1, n_vel)
    corr = np.asarray(post["corr"].values).reshape(-1, n_vel, n_vel)

    samples = [mu]
    labels = [r"$v_\alpha$", r"$v_\delta$"]
    if n_vel == 3:
        labels.append(r"$v_r$")
    if include_std:
        samples.append(std)
        labels.extend([r"$\sigma_\alpha$", r"$\sigma_\delta$"][:n_vel])
        if n_vel == 3:
            labels.append(r"$\sigma_r$")
    if include_corr:
        samples.append(corr[:, 0, 1])
        labels.append(r"$\rho_{\alpha\delta}$")
        if n_vel == 3:
            samples.append(corr[:, 0, 2])
            samples.append(corr[:, 1, 2])
            labels.extend([r"$\rho_{\alpha r}$", r"$\rho_{\delta r}$"])
    samples = np.column_stack(samples)
    kwargs = dict(
        labels=labels,
        show_titles=True,
        title_fmt=".2f",
        quantiles=(0.16, 0.5, 0.84),
        title_kwargs={"fontsize": fs_text},
        label_kwargs={"fontsize": fs_label},
    )
    kwargs.update(corner_kwargs)
    fig = corner.corner(samples, **kwargs)
    dim = "3D" if n_vel == 3 else "2D (tangential)"
    fig.suptitle("Association velocity Gaussian posterior ({})".format(dim), fontsize=fs_title, y=1.02)
    return fig
