#!/usr/bin/env python3
"""Standalone planner runner (no ROS required)."""
import sys
import os
import argparse
import ctypes
import numpy as np

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
_gtsam = root + '/lib/3rdparty/gtsam-4.1.1/install/lib'

# Preload the *bundled* GTSAM 4.1.1 (and TBB 2022) by absolute path with
# RTLD_GLOBAL before importing the pybind11 modules. If ROS is sourced,
# /opt/ros/humble/lib/x86_64-linux-gnu is on LD_LIBRARY_PATH *ahead* of RUNPATH
# and ships a different libgtsam.so.4 (4.2.0) under the same SONAME; mixing the
# two in one process corrupts the LM optimizer (SIGSEGV). Preloading pins 4.1.1.
# (os.environ edits below don't affect this process's already-built search path.)
for _lib in [
    '/usr/local/lib/libtbb.so.12',
    '/usr/local/lib/libtbbmalloc.so.2',
    _gtsam + '/libmetis-gtsam.so',
    _gtsam + '/libgtsam.so.4',
    root + '/lib/build/src/common/smoothing/libcommon_smoothing.so',
]:
    ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)

sys.path.append(root)
sys.path.insert(0, root + '/lib')
from config import Config
from planner_wrapper import TomogramPlanner

parser = argparse.ArgumentParser()
parser.add_argument('--tomo', type=str, default='clinic', help='Tomogram file stem in rsc/tomogram/')
parser.add_argument('--start', type=float, nargs=2, default=[0.0, 0.0], metavar=('X', 'Y'))
parser.add_argument('--end',   type=float, nargs=2, default=[10.0, 5.0], metavar=('X', 'Y'))
args = parser.parse_args()

# Keep child processes (if any) pointed at the bundled GTSAM too.
for _p in [_gtsam, root + '/lib/build/src/common/smoothing']:
    _ld = os.environ.get('LD_LIBRARY_PATH', '')
    if _p not in _ld:
        os.environ['LD_LIBRARY_PATH'] = _ld + ':' + _p

cfg = Config()
planner = TomogramPlanner(cfg)
planner.loadTomogram(args.tomo)

start_pos = np.array(args.start, dtype=np.float32)
end_pos   = np.array(args.end,   dtype=np.float32)

print(f"[INFO] Planning from {start_pos} to {end_pos}")
traj_3d = planner.plan(start_pos, end_pos)

if traj_3d is None:
    print("[WARN] No path found between the given start and end positions.")
else:
    print(f"[INFO] Trajectory found: {traj_3d.shape[0]} waypoints")
    print("[INFO] First waypoint:", traj_3d[0])
    print("[INFO] Last  waypoint:", traj_3d[-1])
    out = os.path.dirname(os.path.abspath(__file__)) + '/../../rsc/clinic_traj.npy'
    np.save(out, traj_3d)
    print(f"[INFO] Trajectory saved to {out}")
