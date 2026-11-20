"""
docking.py -- ligand preparation and AutoDock Vina calls.

prepare_ligand_3d / mol_to_pdbqt_string   SMILES -> protonated 3D molecule -> pdbqt text (Meeko)
dock_ligand                               runs vina.exe on one ligand and returns the scores and the pose pdbqt
unsupported_elements / protonate_ph7_rules / _net_charge_after_protonation   the pH 7 rules and the element filter used for every docked ligand

dock_generated.py imports this module; set docking.VINA_EXECUTABLE_PATH to use another vina.exe.
PDBQTMolecule and RDKitMolCreate are imported here so that docking.PDBQTMolecule / docking.RDKitMolCreate are available to callers.
"""
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate

import config as cfg

# Default vina.exe location, resolved by config.py; callers override it by assigning docking.VINA_EXECUTABLE_PATH.
VINA_EXECUTABLE_PATH = cfg.DEFAULT_VINA

# Vina defaults used when a caller does not pass its own: search effort, number of poses kept, and the search box edge lengths in Angstrom.
DOCKING_EXHAUSTIVENESS = 8
DOCKING_NUM_POSES = 9
DOCKING_BOX_SIZE = (20.0, 20.0, 20.0)

# 'ph7_rules' applies the rule-based charge state below to every docked ligand (so all scores stay comparable); 'none' docks molecules as written.
# Elements Vina/Meeko can type are listed in VINA_SUPPORTED_ELEMENTS; _PROTONATION_SELFTEST remembers whether the one-off rule self-test has run and passed.
LIGAND_PROTONATION_MODE = 'ph7_rules'
VINA_SUPPORTED_ELEMENTS = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}
_PROTONATION_SELFTEST = {'done': False, 'ok': True}


# Builds one energy-minimised 3D conformer of a SMILES for docking.  The molecule first gets the rule-based pH 7 charge state and explicit hydrogens, then several conformers are
# embedded (ETKDGv3, seeded) and each is minimised with MMFF94 (UFF when MMFF has no parameters); the lowest-energy one is kept.
# Returns an RDKit Mol with exactly one conformer, or None if the SMILES is invalid or no conformer could be embedded.
def prepare_ligand_3d(smiles: str, num_conformers: int = 10, random_seed: int = 42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # charge state first, hydrogens second, so the added hydrogens match the charged atoms
    mol = _apply_ph7_protonation(mol)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
    if len(conf_ids) == 0:
        return None

    # minimise every conformer and score it by its force-field energy
    scored = []
    for conf_id in conf_ids:
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id)
            if ff is None:
                raise ValueError("no MMFF parameters for this molecule")
        except Exception:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        ff.Minimize(maxIts=2000)
        scored.append((ff.CalcEnergy(), conf_id))

    # keep only the lowest-energy conformer
    scored.sort(key=lambda x: x[0])
    best_conf_id = scored[0][1]

    best_mol = Chem.Mol(mol)
    best_mol.RemoveAllConformers()
    best_mol.AddConformer(mol.GetConformer(best_conf_id), assignId=True)
    return best_mol


# Converts the 3D molecule from prepare_ligand_3d into PDBQT text with Meeko (rotatable bonds and atom types included).
# Raises RuntimeError if Meeko cannot prepare or write the ligand.
def mol_to_pdbqt_string(mol) -> str:
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    if not mol_setups:
        raise RuntimeError("Meeko produced no molecule setups for this ligand.")
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    if not is_ok:
        raise RuntimeError(f"Meeko PDBQT conversion failed: {error_msg}")
    return pdbqt_string


# Reads the affinity (kcal/mol) of each pose from Vina's 'REMARK VINA RESULT:' lines, in file order; Vina writes the best pose first, so element 0 is the best score.
def _parse_vina_scores(pdbqt_text: str) -> list:
    scores = []
    for line in pdbqt_text.splitlines():
        if line.startswith('REMARK VINA RESULT:'):
            scores.append(float(line.split()[3]))
    return scores


# Docks one SMILES into a receptor and returns {'best_score', 'all_scores', 'poses_pdbqt'}, or None if the ligand cannot be prepared or Vina returns no pose.
# The receptor is a pdbqt file; the search box is given by its centre and edge lengths (Angstrom).  vina_seed makes a run reproducible.  Raises RuntimeError if vina.exe exits with an error.
def dock_ligand(smiles: str, receptor_pdbqt_path: str, box_center: tuple,
                 box_size: tuple = DOCKING_BOX_SIZE,
                 exhaustiveness: int = DOCKING_EXHAUSTIVENESS,
                 n_poses: int = DOCKING_NUM_POSES,
                 vina_seed: int = 42) -> dict:
    mol_3d = prepare_ligand_3d(smiles)
    if mol_3d is None:
        return None
    ligand_pdbqt = mol_to_pdbqt_string(mol_3d)

    # ligand and Vina output live in a temporary folder that is deleted when the block ends
    with tempfile.TemporaryDirectory() as tmp_dir:
        ligand_path = os.path.join(tmp_dir, 'ligand.pdbqt')
        out_path = os.path.join(tmp_dir, 'ligand_out.pdbqt')
        with open(ligand_path, 'w') as f:
            f.write(ligand_pdbqt)

        # Vina command line (the receptor, ligand and output are files; everything else is a plain option)
        cmd = [
            VINA_EXECUTABLE_PATH,
            '--receptor', receptor_pdbqt_path,
            '--ligand', ligand_path,
            '--out', out_path,
            '--center_x', str(box_center[0]), '--center_y', str(box_center[1]), '--center_z', str(box_center[2]),
            '--size_x', str(box_size[0]), '--size_y', str(box_size[1]), '--size_z', str(box_size[2]),
            '--exhaustiveness', str(exhaustiveness),
            '--num_modes', str(n_poses),
            '--seed', str(vina_seed),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"vina.exe failed (exit {result.returncode}):\n{result.stderr}")

        with open(out_path) as f:
            poses_pdbqt = f.read()

    scores = _parse_vina_scores(poses_pdbqt)
    if not scores:
        return None

    return {
        'best_score': scores[0],
        'all_scores': scores,
        'poses_pdbqt': poses_pdbqt,
    }


# Returns the set of elements in the molecule that Vina/Meeko cannot type (boron, silicon, selenium, arsenic, metals ...); an empty set means the molecule can be docked.
def unsupported_elements(mol) -> set:
    return {a.GetSymbol() for a in mol.GetAtoms()} - VINA_SUPPORTED_ELEMENTS


# Rule-based approximation of the pH 7 charge state (not a pKa calculation).  It protonates ONE basic aliphatic amine and deprotonates carboxylic acids; amidines, guanidines, phosphates, sulfonates and
# anything else are left as written.  Returns (new_mol, number of amines protonated, number of acids deprotonated).
def protonate_ph7_rules(mol):
    m = Chem.Mol(mol)
    n_prot = n_deprot = 0
    amine_idx = None
    # find the first basic aliphatic amine: a neutral, non-aromatic, trivalent N with single bonds only, whose neighbours are all non-aromatic carbons or hydrogens
    # and none of those carbons carries a C=N / C=O / C=S / C=C (this excludes amides, anilines, enamines, sulfonamides and N-heteroatom bonds)
    for atom in m.GetAtoms():
        if (atom.GetSymbol() != 'N' or atom.GetIsAromatic() or atom.GetFormalCharge() != 0
                or atom.GetTotalValence() != 3):
            continue
        ok = True
        for b in atom.GetBonds():
            if b.GetBondTypeAsDouble() != 1.0:
                ok = False
                break
            nbr = b.GetOtherAtom(atom)
            if nbr.GetSymbol() == 'H':
                continue
            if nbr.GetSymbol() != 'C' or nbr.GetIsAromatic():
                ok = False
                break
            for b2 in nbr.GetBonds():
                other = b2.GetOtherAtom(nbr)
                if other.GetIdx() == atom.GetIdx():
                    continue
                if b2.GetBondTypeAsDouble() >= 2.0 and other.GetSymbol() in ('N', 'O', 'S', 'C'):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            amine_idx = atom.GetIdx()
            break
    # give that nitrogen a +1 charge and one extra hydrogen
    if amine_idx is not None:
        a = m.GetAtomWithIdx(amine_idx)
        nh = a.GetTotalNumHs()
        a.SetFormalCharge(1)
        a.SetNumExplicitHs(nh + 1)
        a.SetNoImplicit(True)
        n_prot = 1
    # carboxylic acids: a hydroxyl oxygen on a carbon that also carries a C=O becomes a carboxylate (-1, no hydrogen)
    for atom in m.GetAtoms():
        if (atom.GetSymbol() != 'O' or atom.GetFormalCharge() != 0 or atom.GetDegree() != 1
                or atom.GetTotalNumHs() != 1):
            continue
        nbr = atom.GetNeighbors()[0]
        if nbr.GetSymbol() != 'C':
            continue
        if any(b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(nbr).GetSymbol() == 'O'
               for b in nbr.GetBonds()):
            atom.SetFormalCharge(-1)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(True)
            n_deprot += 1
    Chem.SanitizeMol(m)
    return m, n_prot, n_deprot


# Runs the protonation rules once on a handful of molecules with known net charge (amines, acids, amides, anilines, sulfonamides, nitriles ...).  Any mismatch or error switches protonation off for the run,
# with a message, rather than silently mis-charging ligands.
def _run_protonation_selftest():
    cases = [('CN(C)CCc1ccccc1', 1), ('CCN', 1), ('CC(=O)O', -1), ('NCC(=O)O', 0), ('CC(=O)NC', 0),
             ('Nc1ccccc1', 0), ('c1ccncc1', 0), ('CS(=O)(=O)NC', 0), ('C[NH+](C)C', 1), ('CC#N', 0),
             ('CN(C)C=O', 0)]
    ok = True
    try:
        for smi, expected in cases:
            m, _, _ = protonate_ph7_rules(Chem.MolFromSmiles(smi))
            net = sum(a.GetFormalCharge() for a in m.GetAtoms())
            if net != expected:
                ok = False
                print(f"[Protonation] self-test mismatch for {smi}: net charge {net}, expected {expected}")
    except Exception as e:
        ok = False
        print(f"[Protonation] self-test error: {e}")
    _PROTONATION_SELFTEST['done'] = True
    _PROTONATION_SELFTEST['ok'] = ok
    print(f"[Protonation] self-test {'passed' if ok else 'FAILED -- pH 7 protonation disabled for this run'}")


# Applies the pH 7 rules to a molecule if the mode is 'ph7_rules' and the self-test passed (running it on first use); otherwise, or if the rules raise, returns the molecule unchanged.
def _apply_ph7_protonation(mol):
    if LIGAND_PROTONATION_MODE != 'ph7_rules':
        return mol
    if not _PROTONATION_SELFTEST['done']:
        _run_protonation_selftest()
    if not _PROTONATION_SELFTEST['ok']:
        return mol
    try:
        return protonate_ph7_rules(mol)[0]
    except Exception:
        return mol


# Net formal charge of the molecule after the pH 7 rules; used to put a ligand into a charge class (-, 0, +) when matching decoys to actives.
def _net_charge_after_protonation(mol) -> int:
    m = _apply_ph7_protonation(mol)
    return int(sum(a.GetFormalCharge() for a in m.GetAtoms()))
