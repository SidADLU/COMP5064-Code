"""
targets.py -- the study diseases / targets with their candidate receptors and curated actives.

DISEASE_TARGET_CONFIGS   the study table that dock_generated.py reads by entry name (receptor labels, docking boxes, reference ligands, curated actives)
resolve_vocab_entry      finds a vocabulary entry from keywords

Pure data plus one helper, so importing it needs neither torch nor RDKit.
"""
import re


# One entry per study case.  Each entry names the disease and target (keywords as they appear in the ligand library), says whether the disease is CNS-relevant, and lists candidate receptors.
# A receptor has: a label; the prepared receptor file (pdbqt) and the raw pdb it came from; the docking box (centre and edge lengths in Angstrom, in receptor coordinates);
# the SMILES of the co-crystal ligand (used for redocking); optionally 'dataset_target_codes' (library target codes whose ligands are its known actives) and 'calibration_active_smiles' (hand-picked actives).
# dock_generated.py uses the entries 'Malaria - T89515', 'Type-2 Diabetes' and 'Depression'; the first two Malaria entries also carry a Falcipain-2 receptor that the docking study does not use.
DISEASE_TARGET_CONFIGS = [
    {
        # Malaria at target T23761; the docking study uses the T89515 entry below instead.
        'name': 'Malaria',
        'disease_keywords': ['Malaria'],
        'target_keywords': ['T23761'],
        'cns_relevant': False,
        'candidate_receptors': [
            {
                'label': 'PfDHFR',
                # ten known antifolates (diaminopyrimidines, pteridines and a dihydrotriazine) used as curated calibration actives for PfDHFR
                'calibration_active_smiles': [
                    'CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1',
                    'CC1(C)N=C(N)N=C(N)N1c1ccc(Cl)cc1',
                    'COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC',
                    'COc1ccc(Cc2cnc(N)nc2N)cc1OC',
                    'COc1cc(Cc2cnc(N)nc2N)cc(OC)c1Br',
                    'CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O',
                    'Nc1nc(N)c2nc(CNc3ccc(cc3)C(=O)NC(CCC(=O)O)C(=O)O)cnc2n1',
                    'C#CCC(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O',
                    'COc1cc(NCc2ccc3nc(N)nc(N)c3c2C)cc(OC)c1OC',
                    'CC1(C)N=C(N)N=C(N)N1OCCCOc1cc(Cl)c(Cl)cc1Cl',
                ],
                # prepared receptor (pdbqt), the raw structure it came from, the docking box and the co-crystal ligand used for redocking
                'receptor_pdbqt_path': 'rec_malaria.pdbqt',
                'receptor_pdb_path': 'malaria_receptor_raw.pdb',
                'box_center': (27.287, 6.942, 58.547),
                'box_size': (19.007, 17.152, 17.963),
                'reference_smiles': 'CC1(N=C(N=C(N1OCCCOC2=CC(=C(C=C2Cl)Cl)Cl)N)N)C',
            },
            {
                # second candidate receptor of this entry; not used by dock_generated.py
                'label': 'Falcipain-2',
                'receptor_pdbqt_path': 'rec_falcipain2.pdbqt',
                'receptor_pdb_path': 'falcipain2_receptor_raw.pdb',
                'box_center': (-58.244, -1.431, -16.899),
                'box_size': (21.920, 15.869, 18.166),
                'reference_smiles': 'CC(C)C[C@@H](C(=O)NCCCCNC(=[NH2+])N)NC(=O)[C@H](CC(=O)O)O',
            },
        ],
    },
    {
        # Malaria at target T89515, the PfDHFR docking case; the receptor block below repeats the PfDHFR one above.
        'name': 'Malaria - T89515',
        'disease_keywords': ['Malaria'],
        'target_keywords': ['T89515'],
        'cns_relevant': False,
        'candidate_receptors': [
    {
        'label': 'PfDHFR',
        'calibration_active_smiles': [
            'CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1',
            'CC1(C)N=C(N)N=C(N)N1c1ccc(Cl)cc1',
            'COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC',
            'COc1ccc(Cc2cnc(N)nc2N)cc1OC',
            'COc1cc(Cc2cnc(N)nc2N)cc(OC)c1Br',
            'CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O',
            'Nc1nc(N)c2nc(CNc3ccc(cc3)C(=O)NC(CCC(=O)O)C(=O)O)cnc2n1',
            'C#CCC(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O',
            'COc1cc(NCc2ccc3nc(N)nc(N)c3c2C)cc(OC)c1OC',
            'CC1(C)N=C(N)N=C(N)N1OCCCOc1cc(Cl)c(Cl)cc1Cl',
        ],
        'receptor_pdbqt_path': 'rec_malaria.pdbqt',
        'receptor_pdb_path': 'malaria_receptor_raw.pdb',
        'box_center': (27.287, 6.942, 58.547),
        'box_size': (19.007, 17.152, 17.963),
        'reference_smiles': 'CC1(N=C(N=C(N1OCCCOC2=CC(=C(C=C2Cl)Cl)Cl)N)N)C',
    },
    {
        'label': 'Falcipain-2',
        'receptor_pdbqt_path': 'rec_falcipain2.pdbqt',
        'receptor_pdb_path': 'falcipain2_receptor_raw.pdb',
        'box_center': (-58.244, -1.431, -16.899),
        'box_size': (21.920, 15.869, 18.166),
        'reference_smiles': 'CC(C)C[C@@H](C(=O)NCCCCNC(=[NH2+])N)NC(=O)[C@H](CC(=O)O)O',
    },
],
    },
    {
        'name': 'Type-2 Diabetes',
        'disease_keywords': ['Type-2 diabetes'],
        'target_keywords': ['T31391'],
        'cns_relevant': False,
        'candidate_receptors': [
            {
                'label': 'DPP-4',
                # library target code whose ligands (the gliptins) are the known actives for calibration
                'dataset_target_codes': ['T31391'],
                'receptor_pdbqt_path': 'rec_diabetes.pdbqt',
                'receptor_pdb_path': 'diabetes_receptor_raw.pdb',
                'box_center': (40.926, 50.522, 34.843),
                'box_size': (25.394, 16.817, 14.622),
                'reference_smiles': 'C1CN2C(=NN=C2C(F)(F)F)CN1C(=O)C[C@@H](CC3=CC(=C(C=C3F)F)F)N',
            },
        ],
    },
    {
        'name': 'Depression',
        'disease_keywords': ['Major depressive disorder'],
        'target_keywords': ['T27812'],
        'cns_relevant': True,
        'candidate_receptors': [
            {
                'label': 'SERT',
                # library target code whose ligands (SSRIs, zimelidine, tianeptine) are the known actives for calibration
                'dataset_target_codes': ['T27812'],
                'receptor_pdbqt_path': 'rec_depression.pdbqt',
                'receptor_pdb_path': 'depression_receptor_raw.pdb',
                'box_center': (-31.847, -20.608, 2.119),
                'box_size': (17.642, 18.107, 16.787),
                'reference_smiles': 'C1CNC[C@H]([C@@H]1C2=CC=C(C=C2)F)COC3=CC4=C(C=C3)OCO4',
            },
        ],
    },
]


# Finds the entry of a vocabulary dict that matches any of the given keywords: first an exact match after normalising (lower case, punctuation and spaces collapsed), then a substring match.
# Unlike a forgiving lookup it raises ValueError when nothing matches, because a request is meant to name a specific disease or target.
def resolve_vocab_entry(vocab: dict, keywords: list) -> str:
    # lower case, runs of anything other than letters and digits collapsed to one space
    def normalize(value):
        value = str(value).lower().strip()
        value = re.sub(r'[^a-z0-9]+', ' ', value)
        return re.sub(r'\s+', ' ', value).strip()

    # vocabulary entries without the four special tokens
    entries = [k for k in vocab.keys() if k not in ('<PAD>', '<UNK>', '<BOS>', '<EOS>')]
    normalized_map = {normalize(e): e for e in entries}

    for kw in keywords:
        nkw = normalize(kw)
        if nkw in normalized_map:
            return normalized_map[nkw]
    for kw in keywords:
        nkw = normalize(kw)
        if not nkw:
            continue
        for norm_entry, original in normalized_map.items():
            if nkw in norm_entry:
                return original

    # no keyword matched: report which ones and where to look for the exact spelling
    raise ValueError(
        f"No vocabulary entry matched any of {keywords}. "
        f"Check the printed vocabulary sample (or preprocessor.disease_vocab / "
        f".target_vocab directly) for the exact spelling/format used in this checkpoint."
    )
