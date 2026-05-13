"""
MolVision3D — Interactive 3D Molecular & Protein Visualizer

A Streamlit web application for interactive 3D visualization of molecules and
proteins. Users can:
  * Enter a SMILES string or pick an example.
  * Upload structures in .pdb / .mol / .sdf / .mol2 / .xyz.
  * Fetch by name or CID from PubChem.
  * Fetch by PDB ID from the RCSB PDB.

Features include MMFF94 / UFF energy minimization, multiple display styles
(stick / ball-and-stick / wireframe / CPK space-filling / cartoon ribbon),
vector overlays (bond directions, dipole moment, principal axes), MEP-style
surface coloring by Gasteiger partial charge, geometric measurements
(distance, angle, torsion), and export to MOL / PDB / SDF / XYZ.

Run with:
    streamlit run "3d modelling.py"

Required libraries:
    rdkit, streamlit, py3Dmol, numpy, pandas
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolTransforms import GetDihedralDeg

# Draw needs native rendering libs (Cairo / Pillow + boost). On some hosts
# the import fails (Python 3.14 wheels, missing native deps). The 2D
# depiction is a "nice to have", so degrade gracefully.
try:
    from rdkit.Chem import Draw  # type: ignore
    _DRAW_AVAILABLE = True
except Exception:  # pragma: no cover — host-dependent
    Draw = None  # type: ignore
    _DRAW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MolVision3D",
    page_icon="🧬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MoleculeData:
    """A parsed molecule along with its source block and detected format."""
    mol: Optional[Chem.Mol]
    block: str
    fmt: str  # "mol", "pdb", "sdf", or "xyz"
    name: str = ""


# ---------------------------------------------------------------------------
# Molecule parsing / construction
# ---------------------------------------------------------------------------

def mol_from_smiles(smiles: str, add_hs: bool = True,
                    optimize: bool = True) -> MoleculeData:
    """Build a 3D molecule from a SMILES string using RDKit + MMFF94/UFF."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    if add_hs:
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)

    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, maxIters=400)

    block = Chem.MolToMolBlock(mol)
    return MoleculeData(mol=mol, block=block, fmt="mol", name=smiles)


def mol_from_upload(uploaded_file) -> MoleculeData:
    """Parse an uploaded molecular file (.pdb / .mol / .sdf / .mol2 / .xyz)."""
    name = uploaded_file.name
    suffix = name.rsplit(".", 1)[-1].lower()
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

    if suffix == "pdb":
        mol = Chem.MolFromPDBBlock(text, removeHs=False, sanitize=False)
        return MoleculeData(mol=mol, block=text, fmt="pdb", name=name)

    if suffix == "mol":
        mol = Chem.MolFromMolBlock(text, removeHs=False)
        return MoleculeData(mol=mol, block=text, fmt="mol", name=name)

    if suffix == "sdf":
        supplier = Chem.SDMolSupplier()
        supplier.SetData(text, removeHs=False)
        mol = next((m for m in supplier if m is not None), None)
        block = Chem.MolToMolBlock(mol) if mol is not None else text
        return MoleculeData(mol=mol, block=block, fmt="sdf", name=name)

    if suffix == "mol2":
        mol = Chem.MolFromMol2Block(text, removeHs=False, sanitize=False)
        block = Chem.MolToMolBlock(mol) if mol is not None else text
        return MoleculeData(mol=mol, block=block, fmt="mol", name=name)

    if suffix == "xyz":
        # py3Dmol can render xyz directly, RDKit can't parse it without a
        # third-party helper. Render-only fallback.
        return MoleculeData(mol=None, block=text, fmt="xyz", name=name)

    raise ValueError(f"Unsupported file extension: .{suffix}")


# ---------------------------------------------------------------------------
# Database integrations (PubChem, RCSB PDB)
# ---------------------------------------------------------------------------

_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
_RCSB = "https://files.rcsb.org/download/{pdb}.pdb"


def _http_get(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "MolVision3D/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


@st.cache_data(show_spinner=False)
def fetch_pubchem(query: str, by: str = "name") -> MoleculeData:
    """Fetch a 3D SDF for a compound from PubChem (by 'name' or 'cid')."""
    q = query.strip()
    if not q:
        raise ValueError("Empty PubChem query")
    if by == "name":
        url = f"{_PUBCHEM}/name/{urllib.parse.quote(q)}/SDF?record_type=3d"
    else:
        url = f"{_PUBCHEM}/cid/{q}/SDF?record_type=3d"
    try:
        text = _http_get(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"Not found on PubChem: {q!r}") from exc
        raise

    supplier = Chem.SDMolSupplier()
    supplier.SetData(text, removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    block = Chem.MolToMolBlock(mol) if mol is not None else text
    return MoleculeData(mol=mol, block=block, fmt="sdf", name=q)


@st.cache_data(show_spinner=False)
def fetch_rcsb(pdb_id: str) -> MoleculeData:
    """Download a PDB file from the RCSB Protein Data Bank."""
    pid = pdb_id.strip().lower()
    if len(pid) != 4 or not pid.isalnum():
        raise ValueError("PDB IDs are 4-character alphanumeric codes (e.g. 1CRN).")
    try:
        text = _http_get(_RCSB.format(pdb=pid))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"PDB ID {pid.upper()} not found on RCSB.") from exc
        raise

    mol = Chem.MolFromPDBBlock(text, removeHs=False, sanitize=False)
    return MoleculeData(mol=mol, block=text, fmt="pdb", name=pid.upper())


# ---------------------------------------------------------------------------
# Molecular descriptors
# ---------------------------------------------------------------------------

def compute_descriptors(mol: Chem.Mol) -> pd.DataFrame:
    """Return a small table of common molecular descriptors."""
    if mol is None:
        return pd.DataFrame()

    try:
        no_h = Chem.RemoveHs(mol)
    except Exception:
        no_h = mol

    rows = [
        ("Formula", rdMolDescriptors.CalcMolFormula(mol)),
        ("Molecular weight", f"{Descriptors.MolWt(mol):.3f} g/mol"),
        ("Exact mass", f"{Descriptors.ExactMolWt(mol):.4f}"),
        ("Heavy atoms", no_h.GetNumHeavyAtoms()),
        ("Total atoms", mol.GetNumAtoms()),
        ("Bonds", mol.GetNumBonds()),
        ("Rings", rdMolDescriptors.CalcNumRings(mol)),
        ("Aromatic rings", rdMolDescriptors.CalcNumAromaticRings(mol)),
        ("Rotatable bonds", rdMolDescriptors.CalcNumRotatableBonds(mol)),
        ("H-bond donors", rdMolDescriptors.CalcNumHBD(mol)),
        ("H-bond acceptors", rdMolDescriptors.CalcNumHBA(mol)),
        ("TPSA", f"{rdMolDescriptors.CalcTPSA(mol):.2f} Å²"),
        ("LogP (Crippen)", f"{Descriptors.MolLogP(mol):.3f}"),
    ]

    # Stereochemistry summary.
    try:
        stereo = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                           useLegacyImplementation=False)
        if stereo:
            tag = ", ".join(f"{i}:{lbl}" for i, lbl in stereo)
        else:
            tag = "—"
        rows.append(("Chiral centers", tag))
    except Exception:
        pass

    return pd.DataFrame(rows, columns=["Property", "Value"])


def atom_table(mol: Chem.Mol) -> pd.DataFrame:
    """Per-atom positions, partial charge, hybridization."""
    if mol is None or mol.GetNumConformers() == 0:
        return pd.DataFrame()
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    rows = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        try:
            q = float(atom.GetProp("_GasteigerCharge"))
            if not np.isfinite(q):
                q = 0.0
        except Exception:
            q = 0.0
        rows.append({
            "idx": atom.GetIdx(),
            "element": atom.GetSymbol(),
            "x": round(pos.x, 3),
            "y": round(pos.y, 3),
            "z": round(pos.z, 3),
            "formal_q": atom.GetFormalCharge(),
            "partial_q": round(q, 3),
            "hybridization": str(atom.GetHybridization()),
            "aromatic": atom.GetIsAromatic(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Geometric measurements
# ---------------------------------------------------------------------------

def _pos(mol: Chem.Mol, i: int) -> np.ndarray:
    p = mol.GetConformer().GetAtomPosition(i)
    return np.array([p.x, p.y, p.z])


def measure_distance(mol: Chem.Mol, i: int, j: int) -> float:
    return float(np.linalg.norm(_pos(mol, i) - _pos(mol, j)))


def measure_angle(mol: Chem.Mol, i: int, j: int, k: int) -> float:
    a, b, c = _pos(mol, i), _pos(mol, j), _pos(mol, k)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def measure_torsion(mol: Chem.Mol, i: int, j: int, k: int, l: int) -> float:
    return float(GetDihedralDeg(mol.GetConformer(), i, j, k, l))


# ---------------------------------------------------------------------------
# Vector calculations
# ---------------------------------------------------------------------------

def _conf_coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )


def gasteiger_charges(mol: Chem.Mol) -> np.ndarray:
    AllChem.ComputeGasteigerCharges(mol)
    q = np.array([
        float(atom.GetProp("_GasteigerCharge"))
        if atom.HasProp("_GasteigerCharge") else 0.0
        for atom in mol.GetAtoms()
    ])
    return np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)


def dipole_vector(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray, float]:
    """Approximate dipole vector from Gasteiger partial charges (units: e·Å)."""
    coords = _conf_coords(mol)
    charges = gasteiger_charges(mol)
    centroid = coords.mean(axis=0)
    dipole = (charges[:, None] * coords).sum(axis=0)
    return centroid, dipole, float(np.linalg.norm(dipole))


def principal_axes(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = _conf_coords(mol)
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return coords.mean(axis=0), eigvecs[:, order].T, eigvals[order]


# ---------------------------------------------------------------------------
# 3D rendering with py3Dmol
# ---------------------------------------------------------------------------

STYLE_MAP = {
    "Stick": {"stick": {"radius": 0.15}},
    "Ball & Stick": {
        "stick": {"radius": 0.12},
        "sphere": {"scale": 0.25},
    },
    "Wireframe": {"line": {"linewidth": 2.5}},
    "CPK (space-filling)": {"sphere": {}},  # default scale = VDW radius
    "Cartoon (protein)": {"cartoon": {"color": "spectrum"}},
    "Cartoon + Stick": {
        "cartoon": {"color": "spectrum"},
        "stick": {"radius": 0.15},
    },
}


def _add_arrow(view: py3Dmol.view, start: np.ndarray, end: np.ndarray,
               color: str = "red", radius: float = 0.08) -> None:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return
    unit = direction / length
    shaft_end = start + unit * length * 0.8
    view.addCylinder({
        "start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
        "end": {"x": float(shaft_end[0]), "y": float(shaft_end[1]), "z": float(shaft_end[2])},
        "radius": radius,
        "color": color,
        "fromCap": 1, "toCap": 1,
    })
    view.addArrow({
        "start": {"x": float(shaft_end[0]), "y": float(shaft_end[1]), "z": float(shaft_end[2])},
        "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
        "radius": radius * 2.2,
        "color": color,
    })


def _charge_to_color(q: float, vmax: float) -> str:
    """Map a partial charge to a red-white-blue hex color.

    Negative charges -> red, positive -> blue, near-zero -> white.
    Mirrors the MEP convention used by MolView and PyMOL.
    """
    t = max(-1.0, min(1.0, q / max(vmax, 1e-6)))
    if t >= 0:
        # White to blue.
        r = int(255 * (1 - t))
        g = int(255 * (1 - t))
        b = 255
    else:
        # White to red.
        r = 255
        g = int(255 * (1 + t))
        b = int(255 * (1 + t))
    return f"#{r:02x}{g:02x}{b:02x}"


def _apply_mep_coloring(view: py3Dmol.view, mol: Chem.Mol,
                        base_style: dict) -> None:
    """Recolor atoms by Gasteiger partial charge for an MEP-style look."""
    charges = gasteiger_charges(mol)
    vmax = max(float(np.max(np.abs(charges))), 0.2)
    for atom, q in zip(mol.GetAtoms(), charges):
        color = _charge_to_color(q, vmax)
        style = {k: {**v, "color": color} for k, v in base_style.items()}
        view.setStyle({"index": atom.GetIdx()}, style)


def build_view(
    data: MoleculeData,
    style_name: str,
    *,
    show_labels: bool = False,
    show_surface: bool = False,
    show_mep_surface: bool = False,
    show_bond_vectors: bool = False,
    show_dipole: bool = False,
    show_axes: bool = False,
    spin: bool = False,
    bg_color: str = "white",
    surface_opacity: float = 0.65,
    width: int = 800,
    height: int = 580,
) -> str:
    """Build a py3Dmol view and return embeddable HTML."""
    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor(bg_color)

    block_fmt = "pdb" if data.fmt == "pdb" else (
        "xyz" if data.fmt == "xyz" else "mol"
    )
    view.addModel(data.block, block_fmt)

    base_style = STYLE_MAP.get(style_name, STYLE_MAP["Ball & Stick"])
    view.setStyle({}, base_style)

    if show_mep_surface and data.mol is not None:
        # Per-atom recolor; surface inherits those colors when no color is
        # specified in the surface style.
        _apply_mep_coloring(view, data.mol, base_style)
        view.addSurface(py3Dmol.VDW, {"opacity": surface_opacity})
    elif show_surface:
        view.addSurface(py3Dmol.VDW, {
            "opacity": surface_opacity,
            "color": "white",
        })

    if show_labels and data.mol is not None:
        conf = data.mol.GetConformer()
        for atom in data.mol.GetAtoms():
            if atom.GetSymbol() == "H":
                continue
            pos = conf.GetAtomPosition(atom.GetIdx())
            view.addLabel(
                f"{atom.GetSymbol()}{atom.GetIdx()}",
                {
                    "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                    "backgroundColor": "black",
                    "backgroundOpacity": 0.55,
                    "fontColor": "white",
                    "fontSize": 11,
                    "inFront": True,
                },
            )

    if show_bond_vectors and data.mol is not None:
        conf = data.mol.GetConformer()
        for bond in data.mol.GetBonds():
            a = conf.GetAtomPosition(bond.GetBeginAtomIdx())
            b = conf.GetAtomPosition(bond.GetEndAtomIdx())
            start = np.array([a.x, a.y, a.z])
            end = np.array([b.x, b.y, b.z])
            _add_arrow(view, start, end, color="orange", radius=0.05)

    if show_dipole and data.mol is not None:
        try:
            centroid, dipole, magnitude = dipole_vector(data.mol)
            if magnitude > 1e-3:
                tip = centroid + dipole * (2.0 / max(magnitude, 0.1))
                _add_arrow(view, centroid, tip, color="red", radius=0.12)
                view.addLabel(
                    f"μ ≈ {magnitude:.2f} e·Å",
                    {
                        "position": {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])},
                        "backgroundColor": "red",
                        "backgroundOpacity": 0.7,
                        "fontColor": "white",
                        "fontSize": 12,
                    },
                )
        except Exception:
            pass

    if show_axes and data.mol is not None:
        try:
            centroid, axes, eigvals = principal_axes(data.mol)
            colors = ["#1f77b4", "#2ca02c", "#9467bd"]
            scales = np.sqrt(np.maximum(eigvals, 1e-3)) * 1.5 + 1.0
            for axis, color, scale in zip(axes, colors, scales):
                _add_arrow(view, centroid, centroid + axis * scale,
                           color=color, radius=0.07)
        except Exception:
            pass

    view.zoomTo()
    if spin:
        view.spin(True)

    return view._make_html()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def mol_to_xyz(mol: Chem.Mol, comment: str = "MolVision3D") -> str:
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), comment]
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():2s} {p.x: .6f} {p.y: .6f} {p.z: .6f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Example molecules
# ---------------------------------------------------------------------------

EXAMPLES = {
    "Ethanol": "CCO",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "Benzene": "c1ccccc1",
    "Water": "O",
    "Methane": "C",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Cholesterol": "CC(C)CCC[C@@H](C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
}


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------

def sidebar_inputs() -> tuple[Optional[MoleculeData], dict]:
    """Render sidebar controls. Returns (molecule, render_options)."""
    st.sidebar.title("🧬 MolVision3D")
    st.sidebar.caption("Interactive 3D molecular & protein visualizer")

    source = st.sidebar.radio(
        "Input source",
        ["SMILES string", "Example molecule", "Upload file",
         "PubChem lookup", "RCSB PDB lookup"],
        index=0,
    )

    data: Optional[MoleculeData] = None
    error: Optional[str] = None

    if source == "SMILES string":
        smiles = st.sidebar.text_input(
            "SMILES", value="CCO",
            help="Enter a SMILES string (e.g. CCO for ethanol).",
        )
        add_hs = st.sidebar.checkbox("Add explicit hydrogens", value=True)
        optimize = st.sidebar.checkbox(
            "Energy minimization (MMFF94 / UFF)", value=True,
            help="Runs MMFF94 with UFF fallback so the 3D model is in a "
                 "stable, realistic conformation.",
        )
        if smiles.strip():
            try:
                data = mol_from_smiles(smiles.strip(), add_hs=add_hs,
                                       optimize=optimize)
            except Exception as exc:
                error = str(exc)

    elif source == "Example molecule":
        choice = st.sidebar.selectbox("Pick an example", list(EXAMPLES))
        try:
            data = mol_from_smiles(EXAMPLES[choice])
            data.name = choice
        except Exception as exc:
            error = str(exc)

    elif source == "Upload file":
        uploaded = st.sidebar.file_uploader(
            "Upload structure",
            type=["pdb", "mol", "sdf", "mol2", "xyz"],
            help="Supported formats: PDB, MOL, SDF, MOL2, XYZ.",
        )
        if uploaded is not None:
            try:
                data = mol_from_upload(uploaded)
            except Exception as exc:
                error = str(exc)

    elif source == "PubChem lookup":
        by = st.sidebar.radio("Look up by", ["Name", "CID"], horizontal=True)
        query = st.sidebar.text_input(
            "Query", value="aspirin" if by == "Name" else "2244",
            help="PubChem fetches a 3D-optimized SDF for the compound.",
        )
        if st.sidebar.button("Fetch from PubChem"):
            with st.spinner("Querying PubChem…"):
                try:
                    data = fetch_pubchem(query, by="name" if by == "Name" else "cid")
                except Exception as exc:
                    error = str(exc)

    else:  # RCSB PDB lookup
        pdb_id = st.sidebar.text_input(
            "PDB ID (4 chars)", value="1CRN",
            help="Examples: 1CRN (crambin), 1AKI (lysozyme), 4HHB (hemoglobin).",
        )
        if st.sidebar.button("Fetch from RCSB"):
            with st.spinner("Downloading from RCSB…"):
                try:
                    data = fetch_rcsb(pdb_id)
                except Exception as exc:
                    error = str(exc)

    st.sidebar.divider()
    st.sidebar.subheader("Render style")
    default_style = "Cartoon (protein)" if (data and data.fmt == "pdb") else "Ball & Stick"
    style_name = st.sidebar.selectbox(
        "Display style",
        list(STYLE_MAP.keys()),
        index=list(STYLE_MAP.keys()).index(default_style),
    )
    bg_color = st.sidebar.selectbox(
        "Background", ["white", "black", "0xeeeeee", "0x0e1117"], index=0,
    )

    st.sidebar.subheader("Surfaces & overlays")
    show_surface = st.sidebar.checkbox("Van der Waals surface", value=False)
    show_mep_surface = st.sidebar.checkbox(
        "MEP surface (partial-charge gradient)", value=False,
        help="Recolors atoms by Gasteiger partial charge and renders a "
             "translucent VDW surface. Red = δ−, blue = δ+.",
    )
    surface_opacity = st.sidebar.slider(
        "Surface opacity", 0.10, 1.00, 0.65, 0.05,
        disabled=not (show_surface or show_mep_surface),
    )
    show_labels = st.sidebar.checkbox("Atom labels", value=False)
    show_bond_vectors = st.sidebar.checkbox("Bond direction arrows", value=False)
    show_dipole = st.sidebar.checkbox("Dipole moment vector", value=False)
    show_axes = st.sidebar.checkbox("Principal axes", value=False)
    spin = st.sidebar.checkbox("Auto-rotate", value=False)

    options = {
        "style_name": style_name,
        "bg_color": bg_color,
        "show_labels": show_labels,
        "show_surface": show_surface,
        "show_mep_surface": show_mep_surface,
        "surface_opacity": surface_opacity,
        "show_bond_vectors": show_bond_vectors,
        "show_dipole": show_dipole,
        "show_axes": show_axes,
        "spin": spin,
    }

    if error:
        st.sidebar.error(error)

    return data, options


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def measurement_panel(mol: Chem.Mol) -> None:
    """Distance / angle / torsion calculator using atom indices."""
    n = mol.GetNumAtoms()
    st.markdown(
        f"Use atom indices from **0 to {n - 1}** (see *Atom table* below or "
        "enable *Atom labels* in the sidebar)."
    )
    mode = st.radio(
        "Measurement",
        ["Distance (2 atoms)", "Angle (3 atoms)", "Torsion (4 atoms)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode.startswith("Distance"):
        c1, c2 = st.columns(2)
        i = c1.number_input("Atom i", 0, n - 1, 0, key="dist_i")
        j = c2.number_input("Atom j", 0, n - 1, min(1, n - 1), key="dist_j")
        if i != j:
            st.metric(
                f"d({i},{j})", f"{measure_distance(mol, int(i), int(j)):.4f} Å"
            )
    elif mode.startswith("Angle"):
        c1, c2, c3 = st.columns(3)
        i = c1.number_input("Atom i", 0, n - 1, 0, key="ang_i")
        j = c2.number_input("Atom j (vertex)", 0, n - 1, min(1, n - 1), key="ang_j")
        k = c3.number_input("Atom k", 0, n - 1, min(2, n - 1), key="ang_k")
        if len({int(i), int(j), int(k)}) == 3:
            st.metric(
                f"∠({i},{j},{k})",
                f"{measure_angle(mol, int(i), int(j), int(k)):.2f}°",
            )
    else:
        c1, c2, c3, c4 = st.columns(4)
        i = c1.number_input("i", 0, n - 1, 0, key="tor_i")
        j = c2.number_input("j", 0, n - 1, min(1, n - 1), key="tor_j")
        k = c3.number_input("k", 0, n - 1, min(2, n - 1), key="tor_k")
        l = c4.number_input("l", 0, n - 1, min(3, n - 1), key="tor_l")
        if len({int(i), int(j), int(k), int(l)}) == 4:
            st.metric(
                f"τ({i},{j},{k},{l})",
                f"{measure_torsion(mol, int(i), int(j), int(k), int(l)):.2f}°",
            )


def downloads_panel(data: MoleculeData) -> None:
    base = (data.name or "molecule").split()[0].replace("/", "_")
    mol = data.mol
    if mol is not None:
        try:
            st.download_button("Download .mol", Chem.MolToMolBlock(mol),
                               file_name=f"{base}.mol",
                               mime="chemical/x-mdl-molfile")
        except Exception:
            pass
        try:
            st.download_button("Download .pdb", Chem.MolToPDBBlock(mol),
                               file_name=f"{base}.pdb",
                               mime="chemical/x-pdb")
        except Exception:
            pass
        try:
            # An SDF is one or more MOL blocks each terminated by "$$$$".
            sdf = Chem.MolToMolBlock(mol).rstrip() + "\n$$$$\n"
            st.download_button("Download .sdf", sdf, file_name=f"{base}.sdf",
                               mime="chemical/x-mdl-sdfile")
        except Exception:
            pass
        try:
            st.download_button("Download .xyz", mol_to_xyz(mol, base),
                               file_name=f"{base}.xyz", mime="chemical/x-xyz")
        except Exception:
            pass
    else:
        # For raw PDB / XYZ uploads, re-offer the original block.
        ext = data.fmt
        st.download_button(
            f"Download .{ext}", data.block,
            file_name=f"{base}.{ext}",
            mime="text/plain",
        )


def render_main(data: MoleculeData, options: dict) -> None:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader(f"3D view — {data.name or 'molecule'}")
        html = build_view(data, **options)
        components.html(html, height=600, scrolling=False)
        st.caption(
            "Drag to rotate · scroll to zoom · right-drag to pan · "
            "double-click to recenter."
        )

    with right:
        tabs = st.tabs(["Statistics", "Measure", "Atoms", "Downloads"])

        with tabs[0]:
            if data.mol is not None:
                st.dataframe(compute_descriptors(data.mol),
                             hide_index=True, use_container_width=True)
            else:
                st.info(
                    "Descriptors unavailable — RDKit could not build a full "
                    "chemistry model from this file (common for raw protein "
                    "PDBs). 3D rendering still works."
                )

            if _DRAW_AVAILABLE and data.mol is not None:
                with st.expander("2D depiction"):
                    try:
                        img = Draw.MolToImage(Chem.RemoveHs(data.mol),
                                              size=(360, 280))
                        st.image(img, use_column_width=True)
                    except Exception as exc:
                        st.write(f"2D depiction unavailable: {exc}")

        with tabs[1]:
            if data.mol is not None and data.mol.GetNumConformers() > 0:
                measurement_panel(data.mol)
            else:
                st.info("Measurements require a 3D conformer.")

        with tabs[2]:
            if data.mol is not None:
                st.dataframe(atom_table(data.mol), hide_index=True,
                             use_container_width=True, height=420)
            else:
                st.info("Atom table unavailable for this input.")

        with tabs[3]:
            downloads_panel(data)


def render_welcome() -> None:
    st.title("🧬 MolVision3D")
    st.markdown(
        """
        **Interactive 3D Molecular & Protein Visualizer**

        Explore molecules and proteins directly in your browser.

        **Inputs.** SMILES strings, example molecules, file uploads
        (`.pdb`, `.mol`, `.sdf`, `.mol2`, `.xyz`), live **PubChem** lookups by
        name or CID, and live **RCSB PDB** lookups by 4-letter ID.

        **Chemistry.** RDKit handles 2D→3D conversion, ETKDGv3 embedding, and
        MMFF94 / UFF energy minimization for stable conformers.

        **Visualization.** py3Dmol renders stick, ball-and-stick, wireframe,
        CPK space-filling, and cartoon ribbon styles, plus van der Waals and
        **MEP-style** surfaces (red = δ−, blue = δ+).

        **Vectors.** Overlay bond direction arrows, the dipole moment vector,
        and principal inertia axes.

        **Analysis.** Real-time bond length / angle / torsion measurements,
        full descriptor table (MW, formula, TPSA, LogP, rings, donors /
        acceptors, chiral centers), atom-level partial charges, and downloads
        in MOL / PDB / SDF / XYZ.

        Pick an input source from the sidebar to get started.
        """
    )


def main() -> None:
    data, options = sidebar_inputs()
    if data is None:
        render_welcome()
        return
    render_main(data, options)


if __name__ == "__main__":
    main()
