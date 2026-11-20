#!/usr/bin/env python3
"""
prep_receptors.py -- prepare EXTRA receptor structures for dock_generated.py (--extra-receptors / --ensemble-structures).

For every entry in ENTRIES (edit it, or use --only) the script
  1. gets the PDB file (local <ID>.pdb if present, else downloads https://files.rcsb.org/download/<ID>.pdb),
  2. writes <label>_clean.pdb : first model, the chosen protein chain(s), alternate location A only, no waters, no ligands, no ions,
  3. writes <label>_ligand.pdb : the co-crystal ligand (used for the box and for the redocking check),
  4. writes <label>.box.txt : box = ligand bounding box + 5 A on every side, centred on it (exactly the rule of the existing boxes),
  5. converts the clean PDB to rec_<label>.pdbqt with the BUILT-IN converter (no external tool needed).  It reproduces the existing rec_*.pdbqt files:
     the same atom typing and polar hydrogens, residues with missing heavy atoms left out; checked atom for atom on rec_depression.pdbqt and
     rec_diabetes.pdbqt, and Vina scores of 25 poses were identical.  Alternatives: --converter prepare_receptor | obabel | none (clean PDB only),
  6. appends a sodium ion line (atom type 'Na') for entries with keep_ions=['NA'] (not chloride: Vina would type it as a hydrophobic halogen),
  7. writes extra_receptors.json (read by  dock_generated.py --extra-receptors extra_receptors.json).

  python prep_receptors.py --list                      show the entries
  python prep_receptors.py                             prepare all entries
  python prep_receptors.py --only DPP4-2RGU,DPP4-2ONC
  python prep_receptors.py --controls                  also re-prepare the ORIGINAL structures (1X70, 5I6X) with this pipeline and compare
                                                       their atom-type counts with rec_diabetes.pdbqt / rec_depression.pdbqt (a check that the
                                                       conversion tool matches the one used for the originals)
  python prep_receptors.py --pdb-dir <folder>          look for <ID>.pdb there first

The ligand of each entry is found by residue name (ligand=...) or, if that is None, automatically: the largest hetero group of the chain
that is not water, an ion, a sugar or a lipid/detergent.  The groups found are always printed: CHECK that the right one was chosen.
Nothing here decides anything scientific: it only builds files.  ASCII only.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.request

# the extra receptor structures to prepare.  Fields: label, the case key in dock_generated.py, PDB id, protein chain(s), ligand residue name (None = automatic), the SMILES of the
# co-crystal ligand (None = the case's own reference; for another ligand it MUST match the crystal ligand, because the redocking check uses it), in_ensemble (True = part of the
# best-score ensemble, False = calibrated on its own only) and keep_ions (residue names to append).  The ligands are: DPP4-2RGU linagliptin, DPP4-2ONC an alogliptin-series
# quinazolinone, DPP4-3VJK teneligliptin, SERT-5I71 S-citalopram (central site), SERT-7LWD imipramine (cryo-EM 3.65 A, wild type, no ions listed)
ENTRIES = [
    dict(label='DPP4-2RGU', case='diabetes', pdb='2RGU', chains='A', ligand='356',
         ref='CC#CCn1c(nc2c1c(=O)n(c(=O)n2C)Cc1nc(C)c2ccccc2n1)N1CCCC(N)C1', in_ensemble=True, keep_ions=[]),
    dict(label='DPP4-2ONC', case='diabetes', pdb='2ONC', chains='A', ligand='SY1',
         ref='N#Cc1ccccc1Cn1c(nc2ccccc2c1=O)N1CCCC(N)C1', in_ensemble=True, keep_ions=[]),
    dict(label='DPP4-3VJK', case='diabetes', pdb='3VJK', chains='A', ligand='M51',
         ref='Cc1cc(n(n1)c1ccccc1)N1CCN(CC1)C1CNC(C1)C(=O)N1CCSC1', in_ensemble=True, keep_ions=[]),
    dict(label='SERT-5I71', case='depression', pdb='5I71', chains='A', ligand=None,
         ref='CN(C)CCCC1(OCc2cc(ccc12)C#N)c1ccc(F)cc1', in_ensemble=True, keep_ions=['NA']),
    dict(label='SERT-7LWD', case='depression', pdb='7LWD', chains='A', ligand='IXX',
         ref='CN(C)CCCN1c2ccccc2CCc2ccccc12', in_ensemble=False, keep_ions=[]),
]
# the original structures, re-prepared only with --controls and never part of an ensemble: the same fields as ENTRIES plus the original pdbqt and box files they are compared with
CONTROLS = [
    dict(label='DPP-4-ctrl', case='diabetes', pdb='1X70', chains='AB', ligand=None, ref=None, in_ensemble=False, keep_ions=[],
         orig_pdbqt='rec_diabetes.pdbqt', orig_box='rec_diabetes.box.txt'),
    dict(label='SERT-ctrl', case='depression', pdb='5I6X', chains='A', ligand='8PR', ref=None, in_ensemble=False, keep_ions=['NA'],
         orig_pdbqt='rec_depression.pdbqt', orig_box='rec_depression.box.txt'),
]
# residue names never taken as the co-crystal ligand in automatic mode: water, sugars, lipids and detergents, buffer and crystallisation additives, ions
NOT_LIGAND = {'HOH', 'WAT', 'NAG', 'NDG', 'BMA', 'MAN', 'FUC', 'GAL', 'GLC', 'BGC', 'SIA', 'CLR', 'LMT', 'BOG', 'OLC', 'OLA', 'OLB', 'PLM', 'PEG', 'PGE', 'P6G',
              '1PE', '12P', 'LDA', 'DMU', 'DDM', 'C8E', 'GOL', 'EDO', 'SO4', 'PO4', 'ACT', 'ACY', 'TRS', 'MES', 'DMS', 'CIT', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN',
              'MN', 'FE', 'CD', 'CO', 'NI', 'CU', 'IOD', 'BR', 'PC', 'PCW', 'POPC', 'PX4', 'PEE', 'LPP', 'HEX', 'OCT', 'LI1', 'D10', 'DAO', 'UND'}
# water residue names
WATER = {'HOH', 'WAT', 'DOD'}
# the 20 standard amino-acid residue names (a hetero group with one of these names is not taken as the ligand)
AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}


# print() that flushes at once, so progress shows while the script runs.
def say(*x):
    print(*x, flush=True)


# Finds the PDB file for an id: the first folder in dirs that holds <ID>.pdb (any letter case); otherwise downloads it from files.rcsb.org into the first folder.  Exits with
# instructions if the download fails.  Returns the file path.
def get_pdb(pid, dirs):
    names = [f"{pid}.pdb", f"{pid.lower()}.pdb", f"{pid.upper()}.pdb"]
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    dest = os.path.join(dirs[0], f"{pid.upper()}.pdb")
    say(f"   downloading {pid} ...")
    try:
        req = urllib.request.Request(f"https://files.rcsb.org/download/{pid.upper()}.pdb", headers={'User-Agent': 'prep_receptors/1.0'})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except Exception as e:
        raise SystemExit(f"   cannot download {pid} ({type(e).__name__}: {e}).  Save the file from rcsb.org as {dest} and run again.")
    with open(dest, 'wb') as fh:
        fh.write(data)
    return dest


# Reads a PDB file, first model only.  Returns (atoms, hets): the ATOM records and the HETATM records, each a dict with the raw line, atom name, residue name, chain, residue
# number, insertion code, coordinates and element.  Alternate locations other than A (or 1) are skipped.
def parse(path):
    atoms, hets = [], []
    for ln in open(path, encoding='latin-1'):
        if ln.startswith('ENDMDL'):
            break
        if not ln.startswith(('ATOM  ', 'HETATM')) or len(ln) < 54:
            continue
        alt = ln[16]
        if alt not in (' ', 'A', '1'):
            continue
        rec = dict(line=ln.rstrip('\n'), name=ln[12:16].strip(), res=ln[17:20].strip(), chain=ln[21], num=ln[22:26].strip(), icode=ln[26],
                   xyz=(float(ln[30:38]), float(ln[38:46]), float(ln[46:54])), el=(ln[76:78].strip() if len(ln) >= 78 else '').upper())
        (atoms if ln.startswith('ATOM  ') else hets).append(rec)
    return atoms, hets


# Groups hetero records by residue (name, chain, number, insertion code), optionally for one chain only.
def groups(hets, chain=None):
    g = {}
    for h in hets:
        if chain is not None and h['chain'] != chain:
            continue
        g.setdefault((h['res'], h['chain'], h['num'], h['icode']), []).append(h)
    return g


# Number of protein atoms within 4.5 A of any heavy atom of the ligand (20.25 is 4.5 squared).  A buried, binding-site copy has many, a crystal-contact copy few.
def contacts(lig, prot):
    n = 0
    for p_ in prot:
        if any(sum((p_['xyz'][i] - l['xyz'][i]) ** 2 for i in range(3)) < 20.25 for l in lig if l['el'] != 'H'):
            n += 1
    return n


# Chooses the co-crystal ligand.  Returns (group key, atom records).  With a residue name it takes the hetero group of that name; otherwise it takes the groups in the given
# chains that are not on the NOT_LIGAND list, are not amino acids and have at least 8 atoms.  Among several copies it prefers the one with the most protein contacts (named
# ligand) or the largest, then the most contacts (automatic).  It always prints the hetero groups found so that the choice can be checked, and exits if nothing fits.
def pick_ligand(hets, chains, resname, prot=()):
    g = groups(hets)
    mine = {k: v for k, v in g.items() if k[1] in chains}
    pool = mine if mine else g
    say("   hetero groups found (resname chain number: heavy atoms): " + ', '.join(f"{k[0]} {k[1]}{k[2]}:{len([a for a in v if a['el'] != 'H'])}" for k, v in sorted(pool.items(), key=lambda t: -len(t[1]))
                                                                               if k[0] not in WATER))
    if resname:
        cands = {k: v for k, v in g.items() if k[0] == resname}
        if not cands:
            raise SystemExit(f"   no hetero group named {resname} in this file: pick one of the names above and set ligand=... in ENTRIES")
        cands = {k: v for k, v in cands.items() if k[1] in chains} or cands
    else:
        cands = {k: v for k, v in pool.items() if k[0] not in NOT_LIGAND and k[0] not in AA and len(v) >= 8}
        if not cands:
            raise SystemExit("   no ligand candidate found automatically: set ligand=<residue name> in ENTRIES")
    cn = {k_: contacts(v, prot) for k_, v in cands.items()} if prot else {k_: 0 for k_ in cands}
    k = max(cands, key=lambda t: (len(cands[t]) if not resname else 0, cn[t]))
    if len(cands) > 1 and prot:
        say("   candidate copies / protein atoms within 4.5 A: " + ', '.join(f"{x[0]} {x[1]}{x[2]}:{cn[x]}" for x in sorted(cands)) + f"  -> took {k[0]} {k[1]}{k[2]} (most contacts)")
    if not resname and len(cands) > 1:
        say(f"   WARNING: {len(cands)} ligand candidates ({', '.join(sorted(x[0] for x in cands))}); took the largest, {k[0]}: set ligand=... if that is wrong")
    return k, cands[k]


# Docking box from a ligand: centred on the bounding box of its heavy atoms, each side the extent plus 10 A (5 A of margin on every side).  Returns (center, size).
def box_from(lig):
    xs = [a['xyz'] for a in lig if a['el'] != 'H']
    lo = [min(x[i] for x in xs) for i in range(3)]
    hi = [max(x[i] for x in xs) for i in range(3)]
    center = [(lo[i] + hi[i]) / 2 for i in range(3)]
    size = [hi[i] - lo[i] + 10.0 for i in range(3)]
    return center, size


# Writes a box file in the 'center_x = ...' / 'size_x = ...' text format.
def write_box(path, center, size):
    with open(path, 'w', newline='\n') as fh:
        for n, v in zip('xyz', center):
            fh.write(f"center_{n} = {v:.3f}\n")
        for n, v in zip('xyz', size):
            fh.write(f"size_{n} = {v:.3f}\n")


# Reads a box file in that format.  Returns ([cx, cy, cz], [sx, sy, sz]).
def read_box(path):
    d = {}
    for ln in open(path):
        if '=' in ln:
            k, v = ln.split('=', 1)
            d[k.strip()] = float(v)
    return [d['center_x'], d['center_y'], d['center_z']], [d['size_x'], d['size_y'], d['size_z']]


# atom types, mean partial charges and polar hydrogens per standard residue, taken from the existing rec_*.pdbqt files so that structures prepared with the built-in converter are
# typed the same way.  RES_TYPE: residue -> {atom name: [AutoDock type, charge]}; names that start with @ are polar hydrogens (type HD).  RES_HYD: residue -> {parent atom: [hydrogen names]}
RES_TYPE = {"ALA":{"@H":["HD",0.164],"C":["C",0.24],"CA":["C",0.167],"CB":["C",0.041],"N":["N",-0.344],"O":["OA",-0.276],"OXT":["OA",-0.532]},"ARG":{"@H":["HD",0.164],"@HE":["HD",0.26],"@HH11":["HD",0.256],"@HH12":["HD",0.256],"@HH21":["HD",0.256],"@HH22":["HD",0.256],"C":["C",0.242],"CA":["C",0.171],"CB":["C",0.038],"CD":["C",0.197],"CG":["C",0.04],"CZ":["C",0.339],"N":["N",-0.344],"NE":["N",-0.278],"NH1":["N",-0.29],"NH2":["N",-0.29],"O":["OA",-0.273]},"ASN":{"@H":["HD",0.164],"@HD21":["HD",0.159],"@HD22":["HD",0.159],"C":["C",0.242],"CA":["C",0.18],"CB":["C",0.134],"CG":["C",0.22],"N":["N",-0.342],"ND2":["N",-0.368],"O":["OA",-0.273],"OD1":["OA",-0.275]},"ASP":{"@H":["HD",0.164],"C":["C",0.242],"CA":["C",0.176],"CB":["C",0.087],"CG":["C",0.044],"N":["N",-0.343],"O":["OA",-0.273],"OD1":["OA",-0.548],"OD2":["OA",-0.549]},"CYS":{"@H":["HD",0.164],"@HG":["HD",0.102],"C":["C",0.243],"CA":["C",0.182],"CB":["C",0.106],"N":["N",-0.342],"O":["OA",-0.272],"SG":["SA",-0.149]},"GLN":{"@H":["HD",0.164],"@HE21":["HD",0.159],"@HE22":["HD",0.159],"C":["C",0.242],"CA":["C",0.171],"CB":["C",0.045],"CD":["C",0.217],"CG":["C",0.102],"N":["N",-0.343],"NE2":["N",-0.368],"O":["OA",-0.273],"OE1":["OA",-0.275]},"GLU":{"@H":["HD",0.164],"C":["C",0.242],"CA":["C",0.171],"CB":["C",0.041],"CD":["C",0.041],"CG":["C",0.055],"N":["N",-0.344],"O":["OA",-0.273],"OE1":["OA",-0.548],"OE2":["OA",-0.549]},"GLY":{"@H":["HD",0.164],"C":["C",0.239],"CA":["C",0.216],"N":["N",-0.346],"O":["OA",-0.273]},"HIS":{"@H":["HD",0.164],"@HE2":["HD",0.167],"C":["C",0.242],"CA":["C",0.177],"CB":["C",0.093],"CD2":["A",0.107],"CE1":["A",0.196],"CG":["A",0.061],"N":["N",-0.343],"ND1":["NA",-0.242],"NE2":["N",-0.35],"O":["OA",-0.272]},"ILE":{"@H":["HD",0.164],"C":["C",0.243],"CA":["C",0.174],"CB":["C",0.018],"CD1":["C",0.004],"CG1":["C",0.002],"CG2":["C",0.01],"N":["N",-0.343],"O":["OA",-0.272]},"LEU":{"@H":["HD",0.165],"C":["C",0.243],"CA":["C",0.172],"CB":["C",0.038],"CD1":["C",0.007],"CD2":["C",0.007],"CG":["C",-0.015],"N":["N",-0.345],"O":["OA",-0.272]},"LYS":{"@H":["HD",0.164],"@HZ1":["HD",0.345],"@HZ2":["HD",0.345],"@HZ3":["HD",0.345],"C":["C",0.242],"CA":["C",0.171],"CB":["C",0.035],"CD":["C",0.061],"CE":["C",0.258],"CG":["C",0.008],"N":["N",-0.344],"NZ":["N",-0.357],"O":["OA",-0.273]},"MET":{"@H":["HD",0.164],"C":["C",0.242],"CA":["C",0.172],"CB":["C",0.046],"CE":["C",0.085],"CG":["C",0.072],"N":["N",-0.344],"O":["OA",-0.272],"SD":["SA",-0.165]},"PHE":{"@H":["HD",0.164],"C":["C",0.243],"CA":["C",0.175],"CB":["C",0.07],"CD1":["A",0.004],"CD2":["A",0.004],"CE1":["A",0.0],"CE2":["A",0.0],"CG":["A",-0.045],"CZ":["A",0.0],"N":["N",-0.343],"O":["OA",-0.272]},"PRO":{"C":["C",0.24],"CA":["C",0.174],"CB":["C",0.037],"CD":["C",0.121],"CG":["C",0.024],"N":["N",-0.329],"O":["OA",-0.276],"OXT":["OA",-0.53]},"SER":{"@H":["HD",0.164],"@HG":["HD",0.21],"C":["C",0.244],"CA":["C",0.197],"CB":["C",0.191],"N":["N",-0.342],"O":["OA",-0.272],"OG":["OA",-0.392]},"THR":{"@H":["HD",0.164],"@HG1":["HD",0.211],"C":["C",0.245],"CA":["C",0.199],"CB":["C",0.143],"CG2":["C",0.041],"N":["N",-0.341],"O":["OA",-0.273],"OG1":["OA",-0.389]},"TRP":{"@H":["HD",0.164],"@HE1":["HD",0.166],"C":["C",0.242],"CA":["C",0.175],"CB":["C",0.072],"CD1":["A",0.087],"CD2":["A",0.003],"CE2":["A",0.046],"CE3":["A",0.01],"CG":["A",-0.02],"CH2":["A",0.002],"CZ2":["A",0.026],"CZ3":["A",0.001],"N":["N",-0.343],"NE1":["N",-0.359],"O":["OA",-0.272]},"TYR":{"@H":["HD",0.164],"@HH":["HD",0.294],"C":["C",0.243],"CA":["C",0.175],"CB":["C",0.07],"CD1":["A",0.007],"CD2":["A",0.007],"CE1":["A",0.046],"CE2":["A",0.046],"CG":["A",-0.045],"CZ":["A",0.115],"N":["N",-0.343],"O":["OA",-0.272],"OH":["OA",-0.507]},"VAL":{"@H":["HD",0.164],"C":["C",0.243],"CA":["C",0.173],"CB":["C",0.015],"CG1":["C",0.01],"CG2":["C",0.01],"N":["N",-0.343],"O":["OA",-0.272]}}
RES_HYD = {"ALA":{"N":["H"]},"ARG":{"N":["H"],"NE":["HE"],"NH1":["HH11","HH12"],"NH2":["HH21","HH22"]},"ASN":{"N":["H"],"ND2":["HD21","HD22"]},"ASP":{"N":["H"]},"CYS":{"N":["H"],"SG":["HG"]},"GLN":{"N":["H"],"NE2":["HE21","HE22"]},"GLU":{"N":["H"]},"GLY":{"N":["H"]},"HIS":{"N":["H"],"NE2":["HE2"]},"ILE":{"N":["H"]},"LEU":{"N":["H"]},"LYS":{"N":["H"],"NZ":["HZ1","HZ2","HZ3"]},"MET":{"N":["H"]},"PHE":{"N":["H"]},"PRO":{},"SER":{"N":["H"],"OG":["HG"]},"THR":{"N":["H"],"OG1":["HG1"]},"TRP":{"N":["H"],"NE1":["HE1"]},"TYR":{"N":["H"],"OH":["HH"]},"VAL":{"N":["H"]}}


# Unit vector of v (the x axis for a zero-length vector).
def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v) if n > 1e-9 else (1.0, 0.0, 0.0)


# Vector difference a - b.
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


# Vector sum a + b.
def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


# Vector a scaled by k.
def _mul(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


# Cross product of two vectors.
def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


# Distance between two points.
def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


# Two unit vectors perpendicular to u and to each other.
def _perp(u):
    t = (1.0, 0.0, 0.0) if abs(u[0]) < 0.9 else (0.0, 1.0, 0.0)
    e1 = _unit(_cross(u, t))
    return e1, _unit(_cross(u, e1))


# Positions of n hydrogens 1.0 A from atom x.  nbrs are the coordinates of the bonded heavy atoms and nb_of_first is a neighbour of nbrs[0] (for planarity).  Vina ignores
# hydrogens except to mark the parent atom as a donor, so the geometry only has to be sensible (1.0 A, pointing away from the neighbours).
def _place_h(x, nbrs, nb_of_first, n):
    if nbrs:
        u = _unit(tuple(sum(_sub(x, b)[i] for b in nbrs) for i in range(3)))
    else:
        u = (0.0, 0.0, 1.0)
    if n == 1:
        return [_add(x, u)]
    # planar NH2: two hydrogens at +-60 degrees from the away direction, in the plane of the neighbours
    if n == 2:
        pn = _cross(_sub(x, nbrs[0]), _sub(nbrs[0], nb_of_first)) if nbrs and nb_of_first else (0.0, 0.0, 0.0)
        q = _unit(_cross(pn, u)) if any(abs(c) > 1e-6 for c in pn) else _perp(u)[0]
        return [_add(x, _add(_mul(u, 0.5), _mul(q, s * 0.866))) for s in (1, -1)]
    # n = 3: tetrahedral NH3, three hydrogens 120 degrees apart
    e1, e2 = _perp(u)
    out = []
    for k in range(3):
        a = 2.0943951 * k
        w = _add(_mul(e1, math.cos(a)), _mul(e2, math.sin(a)))
        out.append(_add(x, _add(_mul(u, 0.334), _mul(w, 0.943))))
    return out


# Converts the ATOM records of a cleaned PDB into a receptor pdbqt typed like the existing rec_*.pdbqt files, with no external tool: atom types and charges from RES_TYPE, polar
# hydrogens added at sensible positions.  Hydrogens already in the file, residues without a typing rule and residues with missing heavy atoms are left out, and cystines get
# no SG hydrogen.  Returns (number of atoms written, atom-type counts, warnings dict).
def builtin_receptor(clean_pdb, out_pdbqt):
    atoms = []
    for ln in open(clean_pdb, encoding='latin-1'):
        if not ln.startswith('ATOM  '):
            continue
        nm = ln[12:16].strip()
        ec = ln[76:78].strip().upper() if len(ln) >= 78 else ''
        # hydrogens already in the file are dropped; the polar ones are added again below from RES_HYD
        if ec == 'H' or (not ec and nm.lstrip('0123456789').startswith('H')):
            continue
        el = ec or nm[:1]
        atoms.append(dict(name=nm, res=ln[17:20].strip(), chain=ln[21], num=ln[22:26].strip(), icode=ln[26], el=el.upper(),
                          xyz=(float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))))
    if not atoms:
        raise ValueError('no ATOM records in ' + clean_pdb)
    # group the atoms into residues, in file order
    residues, order = {}, []
    for a in atoms:
        k = (a['chain'], a['num'], a['icode'], a['res'])
        if k not in residues:
            residues[k] = []
            order.append(k)
        residues[k].append(a)
    # disulfide bonds: two cysteine SG atoms closer than 2.4 A are bonded and get no hydrogen
    sg = [(k, a) for k in order if k[3] == 'CYS' for a in residues[k] if a['name'] == 'SG']
    ss = set()
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            if _dist(sg[i][1]['xyz'], sg[j][1]['xyz']) < 2.4:
                ss.add(sg[i][0])
                ss.add(sg[j][0])
    out, warn, counts, serial = [], {}, {}, 0

    # Appends one atom line to the pdbqt text (serial, name, residue, coordinates, partial charge, AutoDock type) and counts its type.
    def emit(name, res, chain, num, xyz, q, t):
        nonlocal serial
        serial += 1
        nm4 = (' ' + name.ljust(3)) if len(name) < 4 else name[:4]
        out.append("ATOM  %5d %s %3s %s%4s    %8.3f%8.3f%8.3f  1.00  0.00    %+6.3f %-2s\n" % (serial, nm4, res, chain, num, xyz[0], xyz[1], xyz[2], q, t))
        counts[t] = counts.get(t, 0) + 1

    # coordinates of the previous residue's carbonyl C, needed for the backbone N-H geometry; None across a gap or a dropped residue
    prev_c = None
    for idx, k in enumerate(order):
        chain, num, icode, res = k
        rl = residues[k]
        byname = {a['name']: a for a in rl}
        tbl = RES_TYPE.get(res)
        if tbl is None:
            warn[res] = warn.get(res, 0) + len(rl)
            prev_c = None
            continue
        missing = [n for n in tbl if not n.startswith('@') and n != 'OXT' and n not in byname]
        # like the original preparation, a residue with missing heavy atoms is dropped (OXT is optional)
        if missing:
            warn['dropped incomplete ' + res] = warn.get('dropped incomplete ' + res, 0) + 1
            prev_c = None
            continue
        for a in rl:
            ty = tbl.get(a['name'])
            # an atom without a rule gets a type from its element and charge 0, and is counted in the warnings
            if ty is None:
                t = {'C': 'C', 'N': 'N', 'O': 'OA', 'S': 'SA'}.get(a['el'], a['el'].capitalize())
                ty = [t, 0.0]
                warn[res + ':' + a['name']] = warn.get(res + ':' + a['name'], 0) + 1
            emit(a['name'], res, chain, num, a['xyz'], ty[1], ty[0])
            hyd = RES_HYD.get(res, {}).get(a['name'])
            if not hyd:
                continue
            # a cystine sulfur has no hydrogen
            if res == 'CYS' and a['name'] == 'SG' and k in ss:
                continue
            # bonded heavy atoms (closer than 1.9 A), plus the previous residue's C for a backbone N
            nbrs = [b['xyz'] for b in rl if b is not a and _dist(a['xyz'], b['xyz']) < 1.9]
            if a['name'] == 'N' and prev_c is not None and _dist(a['xyz'], prev_c) < 1.9:
                nbrs.append(prev_c)
            nbf = None
            if nbrs:
                for b in rl:
                    if b is not a and _dist(b['xyz'], nbrs[0]) < 0.01:
                        for c in rl:
                            if c is not a and c is not b and _dist(c['xyz'], b['xyz']) < 1.9:
                                nbf = c['xyz']
                                break
            pos = _place_h(a['xyz'], nbrs, nbf, len(hyd))
            for hn, hp in zip(hyd, pos):
                emit(hn, res, chain, num, hp, tbl.get('@' + hn, ['HD', 0.16])[1], 'HD')
        c_atom = byname.get('C')
        prev_c = c_atom['xyz'] if c_atom else None
    with open(out_pdbqt, 'w', newline='\n') as fh:
        fh.writelines(out)
    return serial, counts, warn


# Converts the cleaned PDB into a receptor pdbqt.  'builtin' (also what 'auto' means) uses builtin_receptor, typed like the existing files (checked atom for atom and by Vina scores
# on the two original receptors); 'prepare_receptor' (ADFR) and 'obabel' are optional external alternatives; 'none' writes only the clean PDB.  Returns the name of the converter
# that was used, or None if no conversion was done.
def convert(clean, out, which):
    if which in ('auto', 'builtin'):
        try:
            n, counts, warn = builtin_receptor(clean, out)
        except Exception as e:
            say(f"   built-in conversion FAILED: {type(e).__name__}: {e}")
            return None
        drop = {k: v for k, v in warn.items() if k.startswith('dropped incomplete')}
        other = {k: v for k, v in warn.items() if not k.startswith('dropped incomplete')}
        if drop:
            say(f"   {sum(drop.values())} residues with missing heavy atoms left out (as the original preparation did); first: " + ', '.join(sorted(drop)[:6]))
        if other:
            say(f"   WARNING: residues/atoms without a typing rule: {dict(list(other.items())[:10])}")
        return 'builtin'
    exe = {t: shutil.which(t) for t in ('prepare_receptor', 'obabel')}
    use = which
    if use == 'none' or not exe.get(use):
        return None
    if use == 'prepare_receptor':
        cmd = [exe[use], '-r', clean, '-o', out, '-A', 'hydrogens']
    else:
        cmd = [exe[use], clean, '-xr', '-h', '-p', '7.4', '-opdbqt', '-O', out]
    say(f"   converting with {use}: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        say(f"   conversion FAILED ({use}): {(r.stderr or r.stdout).strip()[:300]}")
        return None
    return use


# Counts the atoms of a pdbqt file by AutoDock atom type.  Returns (number of atoms, {type: count}).
def pdbqt_hist(path):
    h = {}
    n = 0
    for ln in open(path, encoding='latin-1'):
        if ln.startswith(('ATOM', 'HETATM')):
            t = ln[77:79].strip() if len(ln) >= 79 else ln.split()[-1]
            h[t] = h.get(t, 0) + 1
            n += 1
    return n, h


# Appends sodium ions (AutoDock type Na, charge +1) at the given positions to a receptor pdbqt as HETATM lines numbered from 9000.
def append_ions(pdbqt, ions):
    with open(pdbqt, 'a', newline='\n') as fh:
        num = 9000
        for i in ions:
            fh.write(f"HETATM{num:5d} NA   NA  A{900 + num - 9000:4d}    {i['xyz'][0]:8.3f}{i['xyz'][1]:8.3f}{i['xyz'][2]:8.3f}  1.00  0.00    +1.000 Na\n")
            num += 1


# Prepares one entry: gets the PDB, picks the protein chain(s) and the ligand, stops if the ligand does not sit in those chains, writes the clean protein PDB, the ligand PDB and
# the box file, converts to pdbqt, appends the kept ions and, for control entries, compares the result with the original receptor and box files.  Returns (the entry dict for
# the JSON file, whether a pdbqt was made).
def prep(e, a, dirs):
    say(f"\n[{e['label']}] PDB {e['pdb']}  case {e['case']}  chains {e['chains']}")
    src = get_pdb(e['pdb'], dirs)
    atoms, hets = parse(src)
    chains = set(e['chains'])
    prot = [x for x in atoms if x['chain'] in chains]
    if not prot:
        raise SystemExit(f"   no protein atoms in chain(s) {e['chains']}: chains present: {sorted({x['chain'] for x in atoms})}")
    key, lig = pick_ligand(hets, chains, e.get('ligand'), prot)
    say(f"   ligand: {key[0]} chain {key[1]} number {key[2]} ({len(lig)} atoms)")
    n_con = contacts(lig, prot)
    nres = len({(x['chain'], x['num']) for x in prot})
    say(f"   protein chain(s) {e['chains']}: {nres} residues; {n_con} protein heavy atoms within 4.5 A of the ligand")
    # sanity check: a ligand in the wrong chain touches almost no protein atoms
    if n_con < 25:
        raise SystemExit(f"   STOP: only {n_con} protein atoms touch the ligand, so chain(s) {e['chains']} are probably not the protein this ligand sits in "
                         f"(chains present: {sorted({x['chain'] for x in atoms})}).  Set chains=... in ENTRIES and run again.")
    center, size = box_from(lig)
    lab = e['label']
    with open(os.path.join(a.out_dir, f"{lab}_clean.pdb"), 'w', newline='\n') as fh:
        for x in prot:
            fh.write(x['line'] + '\n')
        fh.write('END\n')
    with open(os.path.join(a.out_dir, f"{lab}_ligand.pdb"), 'w', newline='\n') as fh:
        for x in lig:
            fh.write(x['line'] + '\n')
        fh.write('END\n')
    boxfile = os.path.join(a.out_dir, f"{lab}.box.txt")
    write_box(boxfile, center, size)
    say(f"   box center {', '.join('%.3f' % c for c in center)}  size {', '.join('%.3f' % s for s in size)}  -> {os.path.basename(boxfile)}")
    # ions to keep: the residue names listed in keep_ions, in the chosen chains
    ions = [h for h in hets if h['res'] in (e.get('keep_ions') or []) and h['chain'] in chains]
    # an ion only matters if it lies inside the box or within 4 A of it
    ions = [i for i in ions if all(abs(i['xyz'][k] - center[k]) <= size[k] / 2 + 4.0 for k in range(3))]
    out = os.path.join(a.out_dir, f"rec_{lab}.pdbqt")
    used = convert(os.path.join(a.out_dir, f"{lab}_clean.pdb"), out, a.converter)
    if used is None:
        say(f"   NO conversion done: convert {lab}_clean.pdb to rec_{lab}.pdbqt with the same tool you used for rec_depression.pdbqt etc., then run dock_generated.py")
        converted = False
    else:
        converted = True
        n, h = pdbqt_hist(out)
        say(f"   rec_{lab}.pdbqt: {n} atoms, types {dict(sorted(h.items()))}")
        if ions:
            append_ions(out, ions)
            say(f"   appended {len(ions)} sodium ion(s) near the site: " + '; '.join('(%.1f, %.1f, %.1f)' % i['xyz'] for i in ions))
    # control entries: compare the atom-type counts and the box with the original receptor and box files
    cmp_note = None
    if e.get('orig_pdbqt') and converted:
        op = None
        for d in dirs + ['.']:
            if os.path.exists(os.path.join(d, e['orig_pdbqt'])):
                op = os.path.join(d, e['orig_pdbqt'])
                break
        if op:
            n0, h0 = pdbqt_hist(op)
            n1, h1 = pdbqt_hist(out)
            say(f"   CONTROL: original {e['orig_pdbqt']}: {n0} atoms {dict(sorted(h0.items()))}")
            say(f"            this pipeline:              {n1} atoms {dict(sorted(h1.items()))}")
            diff = sum(abs(h0.get(t, 0) - h1.get(t, 0)) for t in set(h0) | set(h1))
            cmp_note = f"{diff} atoms differ in type count ({100.0 * diff / max(1, n0):.1f}% of {n0})"
            say(f"            -> {cmp_note}")
        ob = None
        for d in dirs + ['.']:
            if e.get('orig_box') and os.path.exists(os.path.join(d, e['orig_box'])):
                ob = os.path.join(d, e['orig_box'])
        if ob:
            c0, s0 = read_box(ob)
            dev = max(abs(c0[i] - center[i]) for i in range(3)), max(abs(s0[i] - size[i]) for i in range(3))
            say(f"   CONTROL box: max deviation from the existing box file {dev[0]:.3f} A (centre), {dev[1]:.3f} A (size)")
    # the record that dock_generated.py reads from the JSON file (--extra-receptors)
    entry = dict(label=lab, case=e['case'], pdbqt=f"rec_{lab}.pdbqt", center=[round(c, 3) for c in center], size=[round(s, 3) for s in size],
                 crystal_ligand=f"{lab}_ligand.pdb", in_ensemble=bool(e.get('in_ensemble')))
    if e.get('ref'):
        entry['reference'] = e['ref']
    return entry, converted


# Prepares the chosen entries (all of them, or those in --only; --controls adds the two control entries), writes the JSON file for dock_generated.py --extra-receptors and
# prints the next command.  --list only shows the entries.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', default='', help='comma list of labels')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--controls', action='store_true', help='also re-prepare 1X70 and 5I6X and compare with the existing receptor files')
    ap.add_argument('--converter', default='auto', choices=['auto', 'builtin', 'prepare_receptor', 'obabel', 'none'])
    ap.add_argument('--pdb-dir', default='.', help='folder with <ID>.pdb files (downloads go to the first of --pdb-dir)')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--json', default='extra_receptors.json')
    a = ap.parse_args()
    entries = list(ENTRIES) + (CONTROLS if a.controls else [])
    if a.list:
        for e in entries:
            say(f"{e['label']:<12} {e['case']:<11} {e['pdb']} chains {e['chains']} ligand {e.get('ligand')} in_ensemble {e.get('in_ensemble')} ions {e.get('keep_ions')}")
        return
    if a.only:
        want = {x.strip() for x in a.only.split(',') if x.strip()}
        entries = [e for e in list(ENTRIES) + CONTROLS if e['label'] in want]
        if not entries:
            raise SystemExit("no entry matches --only; see --list")
    os.makedirs(a.out_dir, exist_ok=True)
    # folders searched for <ID>.pdb, in order; a download goes to the first one
    dirs = [a.pdb_dir, '.', os.path.dirname(os.path.abspath('.'))]
    out, notconv = [], []
    for e in entries:
        ent, ok = prep(e, a, dirs)
        out.append(ent)
        if not ok:
            notconv.append(e['label'])
    with open(a.json, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    say(f"\n[done] {len(out)} entries -> {a.json}" + (f";  NOT converted yet: {notconv}" if notconv else ''))
    say("next:  python dock_generated.py --cases <case> --potency --extra-receptors " + a.json + " --ensemble-structures --diagnose --cache dock_cache_ens.jsonl --out dock_results_ens.csv --summary dock_summary_ens.json")


if __name__ == '__main__':
    main()
