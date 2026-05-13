"""
MolVision3D — Interactive 3D Molecular & Protein Visualizer

A Streamlit web application for interactive 3D visualization of molecules and
proteins. Users can upload structures (.pdb / .mol / .sdf) or enter a SMILES
string. The app generates 3D coordinates with RDKit, renders the structure
with py3Dmol, and overlays scientific vectors (bond directions, dipole
moment, principal axes).

Run with:
    streamlit run "3d modelling.py"

Required libraries:
    rdkit, streamlit, py3Dmol, numpy, pandas, plotly, scipy, biopython
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# Draw needs native rendering libs (Cairo / Pillow + boost). On some hosts
# (e.g. brand-new Python builds without matching rdkit wheels) the import
# fails. The 2D depiction is a "nice to have", so degrade gracefully.
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
    """Build a 3D molecule from a SMILES string using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    if add_hs:
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Fall back to random coords if ETKDG fails.
        AllChem.EmbedMolecule(mol, useRandomCoords=True)

    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, maxIters=400)

    block = Chem.MolToMolBlock(mol)
    return MoleculeData(mol=mol, block=block, fmt="mol", name=smiles)


def mol_from_upload(uploaded_file) -> MoleculeData:
    """Parse an uploaded molecular file (.pdb / .mol / .sdf / .xyz)."""
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
        # First molecule from the SDF.
        supplier = Chem.SDMolSupplier()
        supplier.SetData(text, removeHs=False)
        mol = next((m for m in supplier if m is not None), None)
        block = Chem.MolToMolBlock(mol) if mol is not None else text
        return MoleculeData(mol=mol, block=block, fmt="sdf", name=name)

    if suffix == "xyz":
        return MoleculeData(mol=None, block=text, fmt="xyz", name=name)

    raise ValueError(f"Unsupported file extension: .{suffix}")


# ---------------------------------------------------------------------------
# Molecular descriptors
# ---------------------------------------------------------------------------

def compute_descriptors(mol: Chem.Mol) -> pd.DataFrame:
    """Return a small table of common molecular descriptors."""
    if mol is None:
        return pd.DataFrame()

    # A no-H copy for "heavy" formula / counts that match typical drawings.
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
    return pd.DataFrame(rows, columns=["Property", "Value"])


def atom_table(mol: Chem.Mol) -> pd.DataFrame:
    """Per-atom positions and basic info (works on the 3D conformer)."""
    if mol is None or mol.GetNumConformers() == 0:
        return pd.DataFrame()
    conf = mol.GetConformer()
    rows = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        rows.append({
            "idx": atom.GetIdx(),
            "element": atom.GetSymbol(),
            "x": round(pos.x, 3),
            "y": round(pos.y, 3),
            "z": round(pos.z, 3),
            "charge": atom.GetFormalCharge(),
            "aromatic": atom.GetIsAromatic(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Vector calculations
# ---------------------------------------------------------------------------

def _conf_coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )


def dipole_vector(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray, float]:
    """Approximate dipole moment using Gasteiger partial charges.

    Returns (origin, vector, magnitude_in_eA). Direction goes from the
    centroid toward the negative-charge weighted center (chemistry convention
    points from + to - is also fine; we just visualize an arrow either way).
    """
    AllChem.ComputeGasteigerCharges(mol)
    coords = _conf_coords(mol)
    charges = np.array([
        float(atom.GetProp("_GasteigerCharge"))
        if atom.HasProp("_GasteigerCharge") else 0.0
        for atom in mol.GetAtoms()
    ])
    charges = np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0)
    centroid = coords.mean(axis=0)
    dipole = (charges[:, None] * coords).sum(axis=0)
    magnitude = float(np.linalg.norm(dipole))
    return centroid, dipole, magnitude


def principal_axes(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centroid plus the three principal inertia axes (unit vectors)."""
    coords = _conf_coords(mol)
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]  # largest variance first
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
    "Sphere": {"sphere": {"scale": 0.35}},
    "Line": {"line": {"linewidth": 2}},
    "Cartoon (protein)": {"cartoon": {"color": "spectrum"}},
}


def _add_arrow(view: py3Dmol.view, start: np.ndarray, end: np.ndarray,
               color: str = "red", radius: float = 0.08) -> None:
    """Add a cylinder + cone arrow between two 3D points."""
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return
    unit = direction / length
    # Shaft = 80%, head = 20%.
    shaft_end = start + unit * length * 0.8
    view.addCylinder({
        "start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
        "end": {"x": float(shaft_end[0]), "y": float(shaft_end[1]), "z": float(shaft_end[2])},
        "radius": radius,
        "color": color,
        "fromCap": 1,
        "toCap": 1,
    })
    view.addArrow({
        "start": {"x": float(shaft_end[0]), "y": float(shaft_end[1]), "z": float(shaft_end[2])},
        "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
        "radius": radius * 2.2,
        "color": color,
    })


def build_view(
    data: MoleculeData,
    style_name: str,
    *,
    show_labels: bool = False,
    show_surface: bool = False,
    show_bond_vectors: bool = False,
    show_dipole: bool = False,
    show_axes: bool = False,
    spin: bool = False,
    bg_color: str = "white",
    width: int = 800,
    height: int = 560,
) -> str:
    """Build a py3Dmol view and return embeddable HTML."""
    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor(bg_color)

    # Detect format for py3Dmol.
    block_fmt = "pdb" if data.fmt == "pdb" else (
        "xyz" if data.fmt == "xyz" else "mol"
    )
    view.addModel(data.block, block_fmt)

    view.setStyle({}, STYLE_MAP.get(style_name, STYLE_MAP["Stick"]))

    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.6, "color": "white"})

    if show_labels and data.mol is not None:
        conf = data.mol.GetConformer()
        for atom in data.mol.GetAtoms():
            if atom.GetSymbol() == "H":
                continue  # avoid crowding
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
}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def sidebar_inputs() -> tuple[Optional[MoleculeData], dict]:
    """Render sidebar controls. Returns (molecule, render_options)."""
    st.sidebar.title("🧬 MolVision3D")
    st.sidebar.caption("Interactive 3D molecular & protein visualizer")

    source = st.sidebar.radio(
        "Input source",
        ["SMILES string", "Example molecule", "Upload file"],
        index=0,
    )

    data: Optional[MoleculeData] = None
    error: Optional[str] = None

    if source == "SMILES string":
        smiles = st.sidebar.text_input(
            "SMILES",
            value="CCO",
            help="Enter a SMILES string (e.g. CCO for ethanol).",
        )
        add_hs = st.sidebar.checkbox("Add explicit hydrogens", value=True)
        optimize = st.sidebar.checkbox("MMFF/UFF optimization", value=True)
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

    else:  # Upload file
        uploaded = st.sidebar.file_uploader(
            "Upload structure",
            type=["pdb", "mol", "sdf", "xyz"],
            help="Supported formats: PDB, MOL, SDF, XYZ.",
        )
        if uploaded is not None:
            try:
                data = mol_from_upload(uploaded)
            except Exception as exc:
                error = str(exc)

    st.sidebar.divider()
    st.sidebar.subheader("Render style")
    style_name = st.sidebar.selectbox(
        "Visualization mode",
        list(STYLE_MAP.keys()),
        index=1,
    )
    bg_color = st.sidebar.selectbox(
        "Background",
        ["white", "black", "0xeeeeee", "0x0e1117"],
        index=0,
    )

    st.sidebar.subheader("Overlays")
    show_labels = st.sidebar.checkbox("Atom labels", value=False)
    show_surface = st.sidebar.checkbox("VDW surface", value=False)
    show_bond_vectors = st.sidebar.checkbox("Bond direction arrows", value=False)
    show_dipole = st.sidebar.checkbox("Dipole moment vector", value=False)
    show_axes = st.sidebar.checkbox("Principal axes", value=False)
    spin = st.sidebar.checkbox("Auto-rotate", value=False)

    options = {
        "style_name": style_name,
        "bg_color": bg_color,
        "show_labels": show_labels,
        "show_surface": show_surface,
        "show_bond_vectors": show_bond_vectors,
        "show_dipole": show_dipole,
        "show_axes": show_axes,
        "spin": spin,
    }

    if error:
        st.sidebar.error(error)

    return data, options


def render_main(data: MoleculeData, options: dict) -> None:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader(f"3D view — {data.name or 'molecule'}")
        html = build_view(data, **options)
        components.html(html, height=580, scrolling=False)
        st.caption(
            "Drag to rotate · scroll to zoom · right-drag to pan · "
            "double-click to recenter."
        )

    with right:
        st.subheader("Molecular statistics")
        if data.mol is not None:
            desc = compute_descriptors(data.mol)
            st.dataframe(desc, hide_index=True, use_container_width=True)
        else:
            st.info(
                "Descriptors unavailable for this input (RDKit could not parse "
                "a full chemistry model — common for raw PDB protein files). "
                "3D rendering still works."
            )

        if data.mol is not None:
            with st.expander("Atom table"):
                st.dataframe(atom_table(data.mol), hide_index=True,
                             use_container_width=True)

            with st.expander("2D depiction"):
                if not _DRAW_AVAILABLE:
                    st.write(
                        "2D depiction unavailable on this host "
                        "(rdkit.Chem.Draw could not be imported)."
                    )
                else:
                    try:
                        img = Draw.MolToImage(
                            Chem.RemoveHs(data.mol), size=(360, 280)
                        )
                        st.image(img, use_column_width=True)
                    except Exception as exc:
                        st.write(f"2D depiction unavailable: {exc}")

            with st.expander("Download"):
                mol_block = Chem.MolToMolBlock(data.mol)
                st.download_button(
                    "Download .mol",
                    data=mol_block,
                    file_name=f"{(data.name or 'molecule').split()[0]}.mol",
                    mime="chemical/x-mdl-molfile",
                )
                try:
                    pdb_block = Chem.MolToPDBBlock(data.mol)
                    st.download_button(
                        "Download .pdb",
                        data=pdb_block,
                        file_name=f"{(data.name or 'molecule').split()[0]}.pdb",
                        mime="chemical/x-pdb",
                    )
                except Exception:
                    pass


def render_welcome() -> None:
    st.title("🧬 MolVision3D")
    st.markdown(
        """
        **Interactive 3D Molecular & Protein Visualizer**

        Explore molecules and proteins directly in your browser. Upload a
        structure or enter a SMILES string in the sidebar to get started.

        - Powered by **RDKit** for chemistry, **py3Dmol** for 3D graphics,
          and **Streamlit** for the UI.
        - Toggle overlays for **bond direction vectors**, the **dipole
          moment**, and the **principal inertia axes**.
        - Supported uploads: `.pdb`, `.mol`, `.sdf`, `.xyz`.
        """
    )
    st.info(
        "Tip — try the example molecules (caffeine, aspirin, glucose) from "
        "the sidebar to see vector overlays in action."
    )


def main() -> None:
    data, options = sidebar_inputs()
    if data is None:
        render_welcome()
        return
    render_main(data, options)


if __name__ == "__main__":
    main()
