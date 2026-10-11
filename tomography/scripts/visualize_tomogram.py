#!/usr/bin/env python3
"""Visualize a saved tomogram (rsc/tomogram/<stem>.pickle).

No ROS and no GPU required — only NumPy + Matplotlib (Open3D optional, for --pcd).

The tomogram is a 5-channel tensor of stacked horizontal slices:

    data[0] = trav      traversability cost   (>= cost_barrier  -> impassable)
    data[1] = trav_gx   cost gradient in x
    data[2] = trav_gy   cost gradient in y
    data[3] = elev_g    ground/floor height   (NaN where no floor)
    data[4] = elev_c    ceiling height        (NaN where no ceiling)

Usage
-----
    # 3D view (default): all layers drawn at their floor height, colored by cost
    python3 visualize_tomogram.py --tomo clinic

    # include the saved trajectory if rsc/clinic_traj.npy exists
    python3 visualize_tomogram.py --tomo clinic --traj rsc/clinic_traj.npy

    # classic 2D per-layer panels (cost / floor / clearance)
    python3 visualize_tomogram.py --tomo clinic --2d

    # also dump a colored point cloud (PLY) for CloudCompare/RViz
    python3 visualize_tomogram.py --tomo clinic --pcd
"""

import argparse
import os
import pickle
import sys

import numpy as np

RSG_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/../..'
DEFAULT_DIR = RSG_ROOT + '/rsc/tomogram'


# ─── matplotlib 3D compatibility ────────────────────────────────────────────

def _import_mpl3d():
    """Import `mpl_toolkits.mplot3d`, working around a Debian packaging quirk.

    Debian's apt `python3-matplotlib` ships `/usr/lib/python3/dist-packages/
    mpl_toolkits.pth`, which pre-registers the *apt* `mpl_toolkits` in
    `sys.modules` at interpreter startup. If a newer pip matplotlib shadows
    `matplotlib` itself (e.g. 3.10 in ~/.local), then `mpl_toolkits.mplot3d`
    still comes from the old apt install and fails to import with
    `cannot import name 'docstring' from 'matplotlib'`.

    Fix: pin `mpl_toolkits` to the directory sitting next to the matplotlib
    that is actually loaded. A .pth cannot be renamed without sudo, and
    sys.path order alone cannot win because the apt copy is a regular package.

    The clean environment fix is to remove the apt matplotlib:
        sudo apt remove python3-matplotlib
    """
    import importlib
    import types

    import matplotlib

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(matplotlib.__file__)))
    cand = os.path.join(pkg_root, 'mpl_toolkits')
    if os.path.isfile(os.path.join(cand, 'mplot3d', 'axes3d.py')):
        mod = types.ModuleType('mpl_toolkits')
        mod.__path__ = [cand]
        mod.__file__ = os.path.join(cand, '__init__.py')
        sys.modules['mpl_toolkits'] = mod
    return importlib.import_module('mpl_toolkits.mplot3d').Axes3D


# ─── loading ────────────────────────────────────────────────────────────────

def load_tomogram(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    data = np.asarray(d['data'], dtype=np.float32)
    if data.ndim != 4 or data.shape[0] != 5:
        raise ValueError(
            f'Expected tomogram shape (5, n_layer, dim_x, dim_y), got {data.shape}'
        )
    return {
        'data': data,
        'resolution': float(d['resolution']),
        'center': np.asarray(d['center'], dtype=np.float64),
        'slice_h0': float(d['slice_h0']),
        'slice_dh': float(d['slice_dh']),
    }


def world_extent(tomo):
    """Return (xmin, xmax, ymin, ymax) of the grid in map coordinates."""
    _, n_slice, dim_x, dim_y = tomo['data'].shape
    res = tomo['resolution']
    cx, cy = tomo['center'][0], tomo['center'][1]
    return (cx - dim_x * res / 2, cx + dim_x * res / 2,
            cy - dim_y * res / 2, cy + dim_y * res / 2)


def layer_height(tomo, s):
    """Nominal slice plane of a *non-simplified* layer index.

    NOTE: the exported tomogram only keeps the simplified layers, and their
    original slice indices are not stored. So this is meaningful only for the
    uniform input slicing, NOT for indexing the exported layers — a cell's
    height is `elev_g[3]`, and that is what the planner and RViz use for z.
    Prefer `layer_floor_range()` for reporting the z of an exported layer.
    """
    return tomo['slice_h0'] + s * tomo['slice_dh']


def layer_floor_range(tomo, s):
    """(min, max) of the finite floor heights of exported layer `s`, or None."""
    g = tomo['data'][3][s]
    fin = np.isfinite(g)
    if not fin.any():
        return None
    return float(g[fin].min()), float(g[fin].max())


def data_z_range(tomo):
    """(min, max) z actually stored across all layers (from elev_g), or None."""
    rngs = [r for r in (layer_floor_range(tomo, s)
                        for s in range(tomo['data'].shape[1])) if r]
    if not rngs:
        return None
    return min(r[0] for r in rngs), max(r[1] for r in rngs)


def grid_axes(tomo):
    """1D arrays of the world x / y coordinate of each grid cell centre."""
    _, n_slice, dim_x, dim_y = tomo['data'].shape
    res = tomo['resolution']
    xmin, _, ymin, _ = world_extent(tomo)
    xs = xmin + (np.arange(dim_x) + 0.5) * res
    ys = ymin + (np.arange(dim_y) + 0.5) * res
    return xs, ys


# ─── console summary ────────────────────────────────────────────────────────

def print_summary(tomo, name):
    data = tomo['data']
    _, n_slice, dim_x, dim_y = data.shape
    res, dh = tomo['resolution'], tomo['slice_dh']
    xmin, xmax, ymin, ymax = world_extent(tomo)

    print(f'\n{"=" * 66}')
    print(f' Tomogram: {name}')
    print(f'{"=" * 66}')
    print(f'  shape           : {data.shape}   (channels, layers, dim_x, dim_y)')
    print(f'  resolution      : {res} m/cell')
    print(f'  center          : ({tomo["center"][0]:.3f}, {tomo["center"][1]:.3f})')
    print(f'  extent          : x [{xmin:.1f}, {xmax:.1f}]  y [{ymin:.1f}, {ymax:.1f}]')
    print(f'  slice_h0/dh     : {tomo["slice_h0"]} / {dh}')
    print(f'  layers          : {n_slice}')
    zr = data_z_range(tomo)
    if zr:
        print(f'  floor z (from elev_g): {zr[0]:.2f} .. {zr[1]:.2f} m')

    trav, g, c = data[0], data[3], data[4]
    print(f'\n  NOTE: the exported layers are the *simplified* ones — their original')
    print(f'  slice index is not stored, so `z` below is the floor height from elev_g,')
    print(f'  which is what the planner and RViz use as z.\n')
    print(f'  {"layer":>5} {"floor z [m]":>15} {"cost[min,max]":>16} {"barrier%":>9} '
          f'{"floor%":>7} {"ceil%":>7} {"clearance[m] med":>18}')
    print(f'  {"-" * 82}')
    for s in range(n_slice):
        t = trav[s]
        finite = np.isfinite(t)
        # barrier cells: cost == max (50 typically) — round to avoid f16 fp noise
        barrier = finite & (t >= (np.nanmax(t) - 0.5)) & (t > 0)
        fg, fc = np.isfinite(g[s]), np.isfinite(c[s])
        interval = np.where(fg & fc, c[s] - g[s], np.nan)
        med = np.nanmedian(interval) if np.any(np.isfinite(interval)) else float('nan')
        cmin, cmax = (np.nanmin(t), np.nanmax(t)) if finite.any() else (0.0, 0.0)
        r = layer_floor_range(tomo, s)
        ztxt = f'[{r[0]:.2f},{r[1]:.2f}]' if r else '-- none --'
        print(f'  {s:>5} {ztxt:>15} '
              f'{f"[{cmin:.1f},{cmax:.1f}]":>16} {barrier.mean() * 100:>8.1f}% '
              f'{fg.mean() * 100:>6.1f}% {fc.mean() * 100:>6.1f}% {med:>18.2f}')

    bad = [s for s in range(n_slice) if not np.isfinite(g[s]).any()]
    if bad:
        print(f'\n  ! layers with no floor cells: {bad} — their slice plane lies')
        print(f'    below the whole cloud, so the planner uses the -100 height')
        print(f'    sentinel here. Planning with start/end on such a layer yields a')
        print(f'    path at z=-99.5. Pick a layer with floor data instead.')
    print(f'{"=" * 66}\n')


# ─── 3D view (default) ──────────────────────────────────────────────────────

def _load_traj(tomo, traj_path, log=print):
    """Load a trajectory .npy and sanity-check that it lives near the tomogram.

    Returns (N,3) array or None. Skipped when the height is clearly off the map
    (e.g. a path planned from a click that landed on a floor-less layer)."""
    if not traj_path or not os.path.isfile(traj_path):
        return None
    traj = np.asarray(np.load(traj_path), dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] < 3:
        log(f'[WARN] {traj_path}: expected (N,3), got {traj.shape} — skipped')
        return None
    z = traj[:, 2]
    zr = data_z_range(tomo)
    if zr is None:
        return None
    z_lo, z_hi = zr[0] - 2.0, zr[1] + 2.0
    if not np.any((z >= z_lo) & (z <= z_hi)):
        log(f'[WARN] {traj_path}: z range [{z.min():.2f}, {z.max():.2f}] is far '
            f'from the tomogram floor z [{zr[0]:.2f}, {zr[1]:.2f}] — not overlaid. '
            f'The planner defaults to slice 0; a bottom-most layer has no floor '
            f'cells, so its height is the -100 sentinel (path at z=-99.5). '
            f'Re-plan with the start/end on a layer that has floor data.')
        return None
    return traj


def render_3d(tomo, name, out_path, dpi, stride, elev, azim, z_scale,
              layers=None, traj=None, show=False):
    _import_mpl3d()
    import matplotlib.pyplot as plt

    data = tomo['data']
    _, n_slice, dim_x, dim_y = data.shape
    xs, ys = grid_axes(tomo)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')

    cos_max = float(np.nanmax(data[0])) if np.isfinite(data[0]).any() else 1.0
    cos_max = max(cos_max, 1.0)

    layers = layers if layers is not None else list(range(n_slice))
    step = max(1, int(stride))
    sl = slice(None, None, step)

    px, py, pz, pc = [], [], [], []
    drawn = 0
    for s in layers:
        g = data[3][s]
        mask = np.isfinite(g)
        if not mask.any():
            continue
        keep = np.zeros_like(mask)
        keep[sl, sl] = True
        mask &= keep
        if not mask.any():
            continue
        drawn += int(mask.sum())
        px.append(gx[mask]); py.append(gy[mask]); pz.append(g[mask])
        pc.append(np.clip(data[0][s][mask], 0.0, cos_max))

    if not drawn:
        print('[WARN] no floor cells to draw in 3D')
        return

    px = np.concatenate(px); py = np.concatenate(py)
    pz = np.concatenate(pz); pc = np.concatenate(pc)

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    # green = free, red = barrier (same convention as RViz + the PLY export)
    cmap = plt.get_cmap('turbo')
    sc = ax.scatter(px, py, pz, c=pc, cmap=cmap, vmin=0.0, vmax=cos_max,
                    s=1.0, marker='.', linewidths=0, depthshade=False)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08, label='traversability cost')

    if traj is not None:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='lime', lw=2.0,
                label='trajectory')
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='lime', s=45,
                   marker='o', label='start')
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=45,
                   marker='X', label='goal')
        ax.legend(loc='upper right')

    xmin, xmax, ymin, ymax = world_extent(tomo)
    dx, dy = xmax - xmin, ymax - ymin
    z_lo, z_hi = float(pz.min()), float(pz.max())
    dz = max(z_hi - z_lo, 1e-3)
    # vertical exaggeration: e.g. a 5 m tall map over 56 m would look like paper
    ax.set_box_aspect((dx, dy, dz * z_scale))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(z_lo, z_hi)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_zlabel(f'z [m]  (x{z_scale:g} exaggerated)')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f'{name} — tomogram 3D  ({len(layers)} layers, '
                 f'{drawn} cells, stride {step})', fontsize=13)

    fig.savefig(out_path, dpi=dpi)
    if not show:
        plt.close(fig)


# ─── 2D per-layer panels ────────────────────────────────────────────────────

def render_layer(tomo, s, out_path, name, dpi, show=False):
    import matplotlib.pyplot as plt

    data = tomo['data']
    trav, g, c = data[0][s], data[3][s], data[4][s]
    xmin, xmax, ymin, ymax = world_extent(tomo)
    extent = [ymin, ymax, xmin, xmax]   # note: rows=y, cols=x after transpose

    interval = np.where(np.isfinite(g) & np.isfinite(c), c - g, np.nan)

    # cells are stored [idx_x, idx_y]; transpose so x is horizontal, y vertical
    panels = [
        (trav.T, 'Traversability cost', 'turbo', 0.0, max(1.0, float(np.nanmax(trav)))),
        (g.T, 'Floor height [m]', 'terrain', None, None),
        (interval.T, 'Clearance (ceil - floor) [m]', 'Blues', 0.0, None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes, panels):
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='0.85')
        masked = np.ma.masked_invalid(arr)
        im = ax.imshow(masked, origin='lower', extent=extent, cmap=cmap_obj,
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal')

    r = layer_floor_range(tomo, s)
    ztxt = f'floor z = {r[0]:.2f}..{r[1]:.2f} m' if r else 'no floor cells'
    fig.suptitle(f'{name} — layer {s}  ({ztxt}, '
                 f'{xmax - xmin:.1f} x {ymax - ymin:.1f} m)', fontsize=13)
    fig.savefig(out_path, dpi=dpi)
    if not show:
        plt.close(fig)


# ─── point cloud export ─────────────────────────────────────────────────────

def export_pcd(tomo, out_path, downsample):
    import open3d as o3d

    data = tomo['data']
    _, n_slice, dim_x, dim_y = data.shape
    xs, ys = grid_axes(tomo)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')

    cos = float(np.nanmax(data[0])) if np.isfinite(data[0]).any() else 1.0
    cos = max(cos, 1e-6)
    pts, cols = [], []
    for s in range(n_slice):
        g = data[3][s]
        mask = np.isfinite(g)
        if not mask.any():
            continue
        if downsample > 1:
            keep = np.zeros_like(mask)
            keep[::downsample, ::downsample] = True
            mask &= keep
        pts.append(np.stack([gx[mask], gy[mask], g[mask]], axis=1))
        # color by cost: green (free) -> red (barrier)
        t = np.clip(data[0][s][mask] / cos, 0.0, 1.0)
        cols.append(np.stack([t, 1.0 - t, np.zeros_like(t)], axis=1))

    if not pts:
        print('[WARN] no floor cells to export')
        return
    points = np.concatenate(pts, axis=0).astype(np.float64)
    colors = np.concatenate(cols, axis=0).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)
    print(f'[OK] point cloud: {out_path}  ({len(points)} points, '
          f'green=free -> red=barrier)')


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tomo', default='clinic', help='tomogram stem (default: clinic)')
    ap.add_argument('--dir', default=DEFAULT_DIR, help='tomogram directory')
    ap.add_argument('--out', default=None, help='output dir (default: <dir>/vis_<stem>)')
    ap.add_argument('--layer', type=int, action='append',
                    help='render only these layer indices (repeatable)')
    ap.add_argument('--2d', dest='two_d', action='store_true',
                    help='use the classic per-layer panel view instead of 3D')
    ap.add_argument('--dpi', type=int, default=150)
    ap.add_argument('--pcd', action='store_true', help='also export a colored PLY')
    ap.add_argument('--pcd-downsample', type=int, default=2, metavar='N',
                    help='keep every N-th cell in the PLY (default 2)')
    ap.add_argument('--show', action='store_true', help='open interactive windows instead')

    g3 = ap.add_argument_group('3D view options')
    g3.add_argument('--stride', type=int, default=4,
                    help='draw every N-th cell (default 4)')
    g3.add_argument('--elev', type=float, default=32.0, help='camera elevation deg')
    g3.add_argument('--azim', type=float, default=-60.0, help='camera azimuth deg')
    g3.add_argument('--z-scale', type=float, default=8.0, metavar='F',
                    help='vertical exaggeration of the z axis (default 8)')
    g3.add_argument('--traj', default=None,
                    help='trajectory .npy to overlay (default: rsc/<stem>_traj.npy)')
    g3.add_argument('--no-traj', action='store_true', help='do not overlay a trajectory')
    args = ap.parse_args()

    path = os.path.join(args.dir, args.tomo + '.pickle')
    if not os.path.isfile(path):
        sys.exit(f'error: tomogram not found: {path}\n'
                 f'       run: cd tomography/scripts && '
                 f'python3 run_standalone.py --scene <Scene>')

    tomo = load_tomogram(path)
    print_summary(tomo, args.tomo)

    n_layer = tomo['data'].shape[1]
    layers = args.layer or list(range(n_layer))
    layers = [s for s in layers if 0 <= s < n_layer]
    if not layers:
        sys.exit('error: no valid --layer given')

    import matplotlib
    matplotlib.use('TkAgg' if args.show else 'Agg')

    out_dir = args.out or os.path.join(args.dir, f'vis_{args.tomo}')
    os.makedirs(out_dir, exist_ok=True)

    if args.two_d:
        for s in layers:
            out_path = os.path.join(out_dir, f'layer_{s:02d}.png')
            render_layer(tomo, s, out_path, args.tomo, args.dpi, show=args.show)
            print(f'[OK] {out_path}')
    else:
        traj = None
        if not args.no_traj:
            traj_path = args.traj or os.path.join(RSG_ROOT, 'rsc',
                                                  f'{args.tomo}_traj.npy')
            traj = _load_traj(tomo, traj_path)
            if traj is not None:
                print(f'[OK] overlaying trajectory: {traj_path} '
                      f'({traj.shape[0]} waypoints)')
        out_path = os.path.join(out_dir, f'{args.tomo}_3d.png')
        render_3d(tomo, args.tomo, out_path, args.dpi, args.stride,
                  args.elev, args.azim, args.z_scale, layers=layers, traj=traj,
                  show=args.show)
        print(f'[OK] {out_path}')

    if args.pcd:
        export_pcd(tomo, os.path.join(out_dir, f'{args.tomo}_tomogram.ply'),
                   max(1, args.pcd_downsample))

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
