from __future__ import annotations
import os
import math
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

#!/usr/bin/env python3
"""
cif_visualizer.py

Given a CIF file (TmFeO3.cif in the same directory), this script:
1. Parses the crystal structure with pymatgen.
2. For each Tm and Fe crystallographic site:
    - Finds its local ligand environment (neighboring O, Tm, Fe within cutoff).
    - Determines the site's point group using SiteSymmetryAnalyzer.
    - Extracts approximate symmetry elements (rotation axes and mirror planes).
    - Generates a 3D plot of the local coordination with symmetry elements overlaid.
3. Saves figures as PNG files: site_<label>.png
4. Prints a concise textual summary.

Requirements:
  pip install pymatgen matplotlib numpy

Note: Symmetry element extraction is heuristic (proper rotations and mirrors only).
"""

import matplotlib.pyplot as plt

CIF_FILENAME = "TmFeO3.cif"
NEIGHBOR_CUTOFF = 3.2  # Angstrom; adjust if needed
OUTPUT_DIR = "site_plots"
TOL = 1e-4

# Coloring by element
COLOR_MAP = {
     "Tm": "#8e44ad",
     "Fe": "#e74c3c",
     "O": "#2c3e50"
}

def load_structure(path: str) -> Structure:
     parser = CifParser(path)
     structures = parser.get_structures(primitive=False)
     if not structures:
          raise ValueError("No structure parsed from CIF.")
     return structures[0]

def get_site_label(structure: Structure, idx: int) -> str:
     # Pymatgen often stores labels in site_properties under 'label'
     if "label" in structure.site_properties:
          return structure.site_properties["label"][idx]
     # Fallback: element+index
     return f"{structure[idx].specie.symbol}{idx}"

def angle_from_rotation_matrix(R: np.ndarray) -> float:
     # Proper rotation angle (0..pi)
     cos_theta = (np.trace(R) - 1.0) / 2.0
     cos_theta = min(1.0, max(-1.0, cos_theta))
     return math.degrees(math.acos(cos_theta))

def is_identity(R: np.ndarray) -> bool:
     return np.allclose(R, np.eye(3), atol=1e-6)

def classify_operation(op) -> str:
     R = np.array(op.rotation_matrix)
     det = round(np.linalg.det(R), 6)
     if is_identity(R):
          return "identity"
     # Check inversion
     if np.allclose(R, -np.eye(3), atol=1e-6):
          return "inversion"
     # Reflection: det ~ -1 and eigenvalues ~ (1,1,-1)
     if abs(det + 1) < 1e-5:
          eigvals, _ = np.linalg.eig(R)
          neg_ones = sum(abs(ev + 1) < 1e-4 for ev in eigvals)
          pos_ones = sum(abs(ev - 1) < 1e-4 for ev in eigvals)
          if neg_ones == 1 and pos_ones == 2:
                return "mirror"
          # Could be rotoinversion; ignore for now
          return "improper"
     # Proper rotation det ~ +1
     if abs(det - 1) < 1e-5:
          return "rotation"
     return "other"

def extract_rotation_axis(R: np.ndarray) -> np.ndarray | None:
     # Axis is eigenvector with eigenvalue 1
     eigvals, eigvecs = np.linalg.eig(R)
     for i, ev in enumerate(eigvals):
          if abs(ev - 1) < 1e-4:
                v = np.real(eigvecs[:, i])
                if np.linalg.norm(v) > 1e-8:
                     return v / np.linalg.norm(v)
     return None

def extract_mirror_normal(R: np.ndarray) -> np.ndarray | None:
     eigvals, eigvecs = np.linalg.eig(R)
     for i, ev in enumerate(eigvals):
          if abs(ev + 1) < 1e-4:
                n = np.real(eigvecs[:, i])
                if np.linalg.norm(n) > 1e-8:
                     return n / np.linalg.norm(n)
     return None

def deduplicate_vectors(vecs: list[np.ndarray], tol=5e-2) -> list[np.ndarray]:
     uniq = []
     for v in vecs:
          if v is None:
                continue
          v = v / np.linalg.norm(v)
          keep = True
          for u in uniq:
                if abs(np.dot(u, v)) > 1 - tol:
                     keep = False
                     break
          if keep:
                uniq.append(v)
     return uniq

def make_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
     # Choose vector not parallel to normal
     trial = np.array([1.0, 0.0, 0.0])
     if abs(np.dot(trial, normal)) > 0.9:
          trial = np.array([0.0, 1.0, 0.0])
     v1 = trial - np.dot(trial, normal) * normal
     v1 /= np.linalg.norm(v1)
     v2 = np.cross(normal, v1)
     v2 /= np.linalg.norm(v2)
     return v1, v2

def plot_site_environment(structure: Structure,
                                  idx: int,
                                  rotations: list[np.ndarray],
                                  mirrors: list[np.ndarray],
                                  neighbors_info: list[tuple[str, np.ndarray]],
                                  point_group: str,
                                  label: str,
                                  local_axes: dict | None = None):
     os.makedirs(OUTPUT_DIR, exist_ok=True)
     fig = plt.figure(figsize=(5.5, 5.0), dpi=160)
     ax = fig.add_subplot(111, projection='3d')
     # Central site at origin
     center_cart = structure[idx].coords
     for specie, vec_cart in neighbors_info:
          rel = vec_cart - center_cart
          ax.scatter(rel[0], rel[1], rel[2],
                         color=COLOR_MAP.get(specie, "gray"),
                         s=60, depthshade=True, label=specie)
          ax.text(rel[0], rel[1], rel[2], specie, fontsize=7)
     ax.scatter(0, 0, 0, color="gold", s=120, edgecolors="k")
     # Draw local frame if provided
     if local_axes:
          axis_draw_len = 2.0
          color_map = {"x": "red", "y": "blue", "z": "black"}
          for name in ("x", "y", "z"):
               vec = np.array(local_axes.get(name))
               if vec is None or np.linalg.norm(vec) < 1e-8:
                    continue
               v = vec / max(1e-12, np.linalg.norm(vec)) * axis_draw_len
               ax.quiver(0, 0, 0, v[0], v[1], v[2], length=1.0, color=color_map[name], arrow_length_ratio=0.12)
               ax.text(v[0]*1.05, v[1]*1.05, v[2]*1.05, name, color=color_map[name], fontsize=8)
     # Draw rotation axes
     axis_len = 2.4
     for axis in rotations:
          a = axis * axis_len
          ax.plot([-a[0], a[0]], [-a[1], a[1]], [-a[2], a[2]],
                     color="green", linewidth=2, alpha=0.8)
     # Draw mirror planes
     plane_size = 2.2
     for normal in mirrors:
          v1, v2 = make_plane_basis(normal)
          # Create square in plane
          corners = []
          for sx in (-1, 1):
                for sy in (-1, 1):
                     corners.append((sx * plane_size * v1 + sy * plane_size * v2))
          # Reorder for polygon
          order = [0, 1, 3, 2]
          poly = np.array([corners[i] for i in order])
          ax.plot_trisurf(poly[:, 0], poly[:, 1], poly[:, 2],
                                color="cyan", alpha=0.25, linewidth=0)
          # Normal arrow
          ax.quiver(0, 0, 0, normal[0], normal[1], normal[2],
                        length=1.5, color="cyan", arrow_length_ratio=0.1)
     ax.set_title(f"{label}  PG: {point_group}", fontsize=9)
     # Formatting
     max_range = 0
     all_pts = [np.zeros(3)] + [vec_cart - center_cart for _, vec_cart in neighbors_info]
     if all_pts:
          pts = np.array(all_pts)
          for i in range(3):
                span = pts[:, i].max() - pts[:, i].min()
                max_range = max(max_range, span)
     max_range = max(max_range, 2.5)
     for axis_i, axis_name in enumerate(['x', 'y', 'z']):
          getattr(ax, f"set_{axis_name}lim")((-max_range/2, max_range/2))
     ax.set_box_aspect((1, 1, 1))
     ax.set_xlabel("x (Å)")
     ax.set_ylabel("y (Å)")
     ax.set_zlabel("z (Å)")
     # Avoid duplicate legend entries
     handles, labels = ax.get_legend_handles_labels()
     uniq = {}
     for h, l in zip(handles, labels):
          uniq.setdefault(l, h)
     if uniq:
          ax.legend(uniq.values(), uniq.keys(), fontsize=6, loc='upper right')
     out_path = os.path.join(OUTPUT_DIR, f"site_{label}.png")
     plt.tight_layout()
     plt.savefig(out_path)
     plt.close(fig)
     return out_path

def analyze_site(structure: Structure, idx: int):
     label = get_site_label(structure, idx)
     site = structure[idx]
     # Gather neighbors first (needed to build local cluster for point group analysis)
     neighbors_raw = structure.get_neighbors(site, NEIGHBOR_CUTOFF)
     neighbors_info: list[tuple[str, np.ndarray]] = []
     species = [site.specie.symbol]
     coords = [np.array([0.0, 0.0, 0.0])]  # central at origin
     center_cart = site.coords
     for nn in neighbors_raw:
          specie = nn.specie.symbol
          if specie not in ("O", "Tm", "Fe"):
               continue
          neighbors_info.append((specie, nn.coords))
          species.append(specie)
          coords.append(nn.coords - center_cart)
     # Build a Molecule of the local cluster (central + neighbors translated to origin)
     try:
          mol = Molecule(species, coords)
          pga = PointGroupAnalyzer(mol, tolerance=0.3)
          # PointGroupAnalyzer.get_pointgroup() returns PointGroup object with symbol attribute
          pg_obj = pga.get_pointgroup()
          point_group = getattr(pg_obj, 'sch_symbol', getattr(pg_obj, 'symbol', str(pg_obj)))
          # Obtain symmetry operations; attribute names may vary across versions
          ops = getattr(pga, 'symmops', None)
          if ops is None:
               ops = getattr(pga, 'get_symmetry_operations')()
     except Exception:
          point_group = "Unknown"
          ops = []
     rotation_axes = []
     mirror_normals = []
     for op in ops:
          kind = classify_operation(op)
          R = np.array(op.rotation_matrix)
          if kind == "rotation":
                theta = angle_from_rotation_matrix(R)
                if theta > 1e-2:
                     axis = extract_rotation_axis(R)
                     rotation_axes.append(axis)
          elif kind == "mirror":
                n = extract_mirror_normal(R)
                mirror_normals.append(n)
     rotation_axes = deduplicate_vectors(rotation_axes)
     mirror_normals = deduplicate_vectors(mirror_normals)
     data = {
          "label": label,
          "index": idx,
          "element": structure[idx].specie.symbol,
          "point_group": point_group,
          "rotation_axes": rotation_axes,
          "mirror_normals": mirror_normals,
          "neighbors": neighbors_info
     }
     # Compute local frame if applicable
     data["local_axes"] = compute_local_frame(structure, idx, data)
     return data

# --- New code for local coordinate frame construction ---

def _normalize(v: np.ndarray) -> np.ndarray:
     n = np.linalg.norm(v)
     if n < 1e-8:
          return v * 0
     return v / n

def compute_local_frame(structure: Structure, idx: int, data: dict):
     """
     Build local orthonormal axes (x,y,z) for specified symmetry cases.
     Cs (mirror, symbol variants: 'm','Cs'): z = mirror normal; x = in-plane Tm-O bond; y = z x x.
     Ci (inversion only, variants: '-1','Ci','i'): z = Fe-O bond most aligned with lattice c; x,y from remaining bonds.
     Returns dict {x: [...], y: [...], z: [...]} or None.
     """
     pg = (data.get("point_group") or "").lower()
     elem = data.get("element")
     neighbors = data.get("neighbors", [])
     center = structure[idx].coords
     # Standardize synonyms
     is_cs = pg in {"m", "cs"}
     is_ci = pg in {"-1", "ci", "i"}
     if is_cs and elem == "Tm":
          mirrors = data.get("mirror_normals", [])
          if not mirrors:
               return None
          z = _normalize(mirrors[0])
          candidate = None
          score_best = None
          for specie, coord in neighbors:
               if specie != "O":
                    continue
               v = coord - center
               v_plane = v - np.dot(v, z) * z
               plane_norm = np.linalg.norm(v_plane)
               if plane_norm < 1e-5:
                    continue
               score = (abs(np.dot(v, z)), -plane_norm)
               if score_best is None or score < score_best:
                    score_best = score
                    candidate = v_plane
          if candidate is None:
               return None
          x = _normalize(candidate)
          y = _normalize(np.cross(z, x))
          x = _normalize(np.cross(y, z))
          # Enforce right-handedness: x x y should align with z
          if np.dot(np.cross(x, y), z) < 0:
               y = -y
          return {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()}
     if is_ci and elem == "Fe":
          bonds = []
          for specie, coord in neighbors:
               if specie == "O":
                    v = coord - center
                    if np.linalg.norm(v) > 1e-5:
                         bonds.append(v)
          if len(bonds) < 1:
               return None
          # Lattice c direction
          c_vec = structure.lattice.matrix[2]
          c_dir = _normalize(c_vec)
          # Choose bond most aligned with c
          z = None
          best_align = -1.0
          for v in bonds:
               vn = _normalize(v)
               align = abs(np.dot(vn, c_dir))
               if align > best_align:
                    best_align = align
                    z = vn
          if z is None:
               return None
          # Choose x from remaining bonds: as perpendicular as possible to z, then largest norm
          x = None
          best_score = None
          for v in bonds:
               vn = _normalize(v)
               if abs(np.dot(vn, z)) > 0.95:  # too parallel
                    continue
               perp = abs(np.dot(vn, z))  # want this small
               score = (perp, -np.linalg.norm(v))
               if best_score is None or score < best_score:
                    best_score = score
                    x = vn
          if x is None:
               # Fallback: project c_dir onto plane perpendicular to z
               proj = c_dir - np.dot(c_dir, z)*z
               if np.linalg.norm(proj) < 1e-6:
                    # arbitrary perpendicular
                    trial = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(trial, z)) > 0.9:
                         trial = np.array([0.0, 1.0, 0.0])
                    proj = trial - np.dot(trial, z)*z
               x = _normalize(proj)
          y = _normalize(np.cross(z, x))
          x = _normalize(np.cross(y, z))
          # Enforce right-handedness: x x y should align with z
          if np.dot(np.cross(x, y), z) < 0:
               y = -y
          return {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()}
     return None

def main():
     if not os.path.isfile(CIF_FILENAME):
          raise SystemExit(f"CIF file {CIF_FILENAME} not found in current directory.")
     structure = load_structure(CIF_FILENAME)
     targets = [i for i, site in enumerate(structure) if site.specie.symbol in ("Tm", "Fe")]
     summaries = []
     for idx in targets:
          data = analyze_site(structure, idx)
          fig_path = plot_site_environment(
                structure,
                idx,
                data["rotation_axes"],
                data["mirror_normals"],
                data["neighbors"],
                data["point_group"],
                data["label"],
                data.get("local_axes")
          )
          data["figure"] = fig_path
          summaries.append(data)
     print("Site symmetry summary (Tm & Fe):")
     for s in summaries:
          print(f"- {s['label']} (index {s['index']}, element {s['element']}): Point Group = {s['point_group']}")
          print(f"  Neighbors counted: {len(s['neighbors'])}")
          print(f"  Rotation axes ({len(s['rotation_axes'])}): " +
                  ", ".join([f"[{v[0]:.2f} {v[1]:.2f} {v[2]:.2f}]" for v in s['rotation_axes']]) if s['rotation_axes'] else "  Rotation axes: none")
          print(f"  Mirror planes ({len(s['mirror_normals'])}): " +
                  ", ".join([f"n=[{v[0]:.2f} {v[1]:.2f} {v[2]:.2f}]" for v in s['mirror_normals']]) if s['mirror_normals'] else "  Mirror planes: none")
          if s.get("local_axes"):
               ax = s["local_axes"]
               print("  Local frame:")
               print(f"    x=[{ax['x'][0]:.3f} {ax['x'][1]:.3f} {ax['x'][2]:.3f}]")
               print(f"    y=[{ax['y'][0]:.3f} {ax['y'][1]:.3f} {ax['y'][2]:.3f}]")
               print(f"    z=[{ax['z'][0]:.3f} {ax['z'][1]:.3f} {ax['z'][2]:.3f}]")
          else:
               print("  Local frame: (not defined for this site)")
          print(f"  Figure: {s['figure']}")
     print(f"Plots saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
     main()