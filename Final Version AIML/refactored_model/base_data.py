"""
base_data.py -- vocabularies and integer encoding for the base language model.

DrugDataPreprocessor  builds and holds the vocabularies and encodes SMILES + context lists into integer arrays (the exemplar scripts rebuild it from a checkpoint
                      with DrugDataPreprocessor.from_saved_vocabularies)
DrugDataset           torch Dataset over the encoded rows
preprocess_data       builds the preprocessor and dataset from a DataFrame
collate_fn            batches a list of dataset items
"""
import ast
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import SMOKE_TEST


# Builds vocabularies from a table of ligands and encodes rows as fixed-length integer arrays: SMILES characters (character-level, with BOS / EOS / PAD / UNK) and four
# context lists (diseases, targets, KEGG pathways, WikiPathways).  The exemplar generator uses only the SMILES half; the context vocabularies belong to the ID-conditioned
# base model.  A saved instance is rebuilt without the table by from_saved_vocabularies.
class DrugDataPreprocessor:
    # Stores the length limits (SMILES characters, and the numbers of targets, pathways and diseases kept per row), sets the special-token ids and builds all vocabularies from the DataFrame.
    def __init__(self, df: pd.DataFrame, max_smiles_len: int = 200, max_target_len: int = 10,
                 max_kegg_len: int = 20, max_wiki_len: int = 20, max_disease_len: int = 5):
        self.df = df
        self.max_smiles_len = max_smiles_len
        self.max_target_len = max_target_len
        self.max_kegg_len = max_kegg_len
        self.max_wiki_len = max_wiki_len
        self.max_disease_len = max_disease_len

        self.smiles_vocab = {}
        self.disease_vocab = {}
        self.target_vocab = {}
        self.kegg_vocab = {}
        self.wiki_vocab = {}

        self.smiles_inv_vocab = {}
        self.disease_inv_vocab = {}
        self.target_inv_vocab = {}
        self.kegg_inv_vocab = {}
        self.wiki_inv_vocab = {}

        # special-token ids: PAD 0, BOS 1, EOS 2, UNK 3 (BOS and EOS exist only in the SMILES vocabulary)
        self.PAD_IDX = 0
        self.UNK_IDX = 3
        self.BOS_IDX = 1
        self.EOS_IDX = 2

        self._build_vocabularies()
        self._build_smiles_vocab()

    # Turns a CSV cell holding a stringified list ("['A', 'B']") into a Python list.  Empty, missing or '[]' cells give [], a single literal is wrapped in a list, and a
    # malformed string is split on commas.
    def _parse_list_field(self, value):
        if pd.isna(value) or value == '' or value == '[]':
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except (ValueError, SyntaxError):
            if isinstance(value, str):
                return [item.strip().strip("'\"") for item in value.split(',') if item.strip()]
            return []

    # Builds the four context vocabularies (disease, target, KEGG pathway, WikiPathway) as {item: index} dicts with <PAD> = 0 and <UNK> = 3, plus their inverses.  Items are
    # sorted for a stable order and numbered from 2, so the second item shares index 3 with <UNK>.
    def _build_vocabularies(self):
        all_diseases = set()
        all_targets = set()
        all_kegg = set()
        all_wiki = set()

        for _, row in self.df.iterrows():
            diseases = self._parse_list_field(row.get('Disease Entry', []))
            targets = self._parse_list_field(row.get('Targets for Drug', []))
            kegg = self._parse_list_field(row.get('Kegg Pathway for Targets', []))
            wiki = self._parse_list_field(row.get('Wiki pathway for Targets', []))

            all_diseases.update(diseases)
            all_targets.update(targets)
            all_kegg.update(kegg)
            all_wiki.update(wiki)

        self.disease_vocab = {
            '<PAD>': self.PAD_IDX,
            '<UNK>': self.UNK_IDX
        }
        for idx, item in enumerate(sorted(all_diseases), start=2):
            self.disease_vocab[item] = idx

        self.target_vocab = {
            '<PAD>': self.PAD_IDX,
            '<UNK>': self.UNK_IDX
        }
        for idx, item in enumerate(sorted(all_targets), start=2):
            self.target_vocab[item] = idx

        self.kegg_vocab = {
            '<PAD>': self.PAD_IDX,
            '<UNK>': self.UNK_IDX
        }
        for idx, item in enumerate(sorted(all_kegg), start=2):
            self.kegg_vocab[item] = idx

        self.wiki_vocab = {
            '<PAD>': self.PAD_IDX,
            '<UNK>': self.UNK_IDX
        }
        for idx, item in enumerate(sorted(all_wiki), start=2):
            self.wiki_vocab[item] = idx

        self.disease_inv_vocab = {v: k for k, v in self.disease_vocab.items()}
        self.target_inv_vocab = {v: k for k, v in self.target_vocab.items()}
        self.kegg_inv_vocab = {v: k for k, v in self.kegg_vocab.items()}
        self.wiki_inv_vocab = {v: k for k, v in self.wiki_vocab.items()}

    # Builds the character-level SMILES vocabulary: <PAD>, <UNK>, <BOS>, <EOS>, then every character that occurs in the table, most frequent first (indices from 4).
    def _build_smiles_vocab(self):
        char_counter = Counter()
        for smiles in self.df['Canonical Smiles'].dropna():
            char_counter.update(smiles)

        self.smiles_vocab = {
            '<PAD>': self.PAD_IDX,
            '<UNK>': self.UNK_IDX,
            '<BOS>': self.BOS_IDX,
            '<EOS>': self.EOS_IDX
        }

        sorted_chars = sorted(char_counter.keys(), key=lambda x: char_counter[x], reverse=True)
        for idx, char in enumerate(sorted_chars, start=4):
            self.smiles_vocab[char] = idx

        self.smiles_inv_vocab = {v: k for k, v in self.smiles_vocab.items()}

    # Encodes a SMILES string as <BOS>, one index per character, <EOS>, padded with <PAD> to max_smiles_len.  A missing or empty string gives an all-PAD array and unknown
    # characters map to <UNK>.  A string longer than max_smiles_len - 1 is cut and loses its <EOS>.
    def encode_smiles(self, smiles: str) -> np.ndarray:
        if smiles is None or not isinstance(smiles, str) or smiles.strip() == '':
            return np.full(self.max_smiles_len, self.PAD_IDX, dtype=np.int64)

        tokens = ['<BOS>'] + list(smiles) + ['<EOS>']

        if len(tokens) > self.max_smiles_len:
            tokens = tokens[:self.max_smiles_len]

        indices = [self.smiles_vocab.get(token, self.UNK_IDX) for token in tokens]

        padded_indices = indices + [self.PAD_IDX] * (self.max_smiles_len - len(indices))

        return np.array(padded_indices, dtype=np.int64)

    # Encodes a list of items as vocabulary indices, cut or padded with <PAD> to max_len; items not in the vocabulary map to <UNK>.
    def encode_categorical_list(self, items: list, vocab: dict, max_len: int) -> np.ndarray:
        if not items:
            return np.full(max_len, self.PAD_IDX, dtype=np.int64)

        indices = [vocab.get(item, self.UNK_IDX) for item in items]

        if len(indices) > max_len:
            indices = indices[:max_len]

        padded_indices = indices + [self.PAD_IDX] * (max_len - len(indices))

        return np.array(padded_indices, dtype=np.int64)

    # Encodes every row of the DataFrame.  Returns five integer arrays (SMILES, disease, target, KEGG, WikiPathway), one row per DataFrame row.
    def get_encoded_data(self):
        smiles_encoded = []
        disease_encoded = []
        target_encoded = []
        kegg_encoded = []
        wiki_encoded = []

        for _, row in self.df.iterrows():
            s = self.encode_smiles(row.get('Canonical Smiles', ''))
            smiles_encoded.append(s)

            d = self.encode_categorical_list(
                self._parse_list_field(row.get('Disease Entry', [])),
                self.disease_vocab,
                self.max_disease_len
            )
            disease_encoded.append(d)

            t = self.encode_categorical_list(
                self._parse_list_field(row.get('Targets for Drug', [])),
                self.target_vocab,
                self.max_target_len
            )
            target_encoded.append(t)

            k = self.encode_categorical_list(
                self._parse_list_field(row.get('Kegg Pathway for Targets', [])),
                self.kegg_vocab,
                self.max_kegg_len
            )
            kegg_encoded.append(k)

            w = self.encode_categorical_list(
                self._parse_list_field(row.get('Wiki pathway for Targets', [])),
                self.wiki_vocab,
                self.max_wiki_len
            )
            wiki_encoded.append(w)

        smiles_array = np.array(smiles_encoded)
        disease_array = np.array(disease_encoded)
        target_array = np.array(target_encoded)
        kegg_array = np.array(kegg_encoded)
        wiki_array = np.array(wiki_encoded)

        return smiles_array, disease_array, target_array, kegg_array, wiki_array

    # Rebuilds a preprocessor from the 'preprocessor_data' dict stored in a checkpoint, without the original DataFrame (df is None).  This is how the exemplar scripts load a
    # checkpoint on a machine that does not have the ligand CSV.  The vocabularies, length limits and special-token ids are restored.
    @classmethod
    def from_saved_vocabularies(cls, vocab_data: dict) -> 'DrugDataPreprocessor':
        # skip __init__, which needs a DataFrame
        self = cls.__new__(cls)
        self.df = None

        self.max_smiles_len = vocab_data['max_smiles_len']
        self.max_target_len = vocab_data['max_target_len']
        self.max_kegg_len = vocab_data['max_kegg_len']
        self.max_wiki_len = vocab_data['max_wiki_len']
        self.max_disease_len = vocab_data['max_disease_len']

        self.smiles_vocab = vocab_data['smiles_vocab']
        self.disease_vocab = vocab_data['disease_vocab']
        self.target_vocab = vocab_data['target_vocab']
        self.kegg_vocab = vocab_data['kegg_vocab']
        self.wiki_vocab = vocab_data['wiki_vocab']

        self.smiles_inv_vocab = {v: k for k, v in self.smiles_vocab.items()}
        self.disease_inv_vocab = {v: k for k, v in self.disease_vocab.items()}
        self.target_inv_vocab = {v: k for k, v in self.target_vocab.items()}
        self.kegg_inv_vocab = {v: k for k, v in self.kegg_vocab.items()}
        self.wiki_inv_vocab = {v: k for k, v in self.wiki_vocab.items()}

        self.PAD_IDX = vocab_data['PAD_IDX']
        self.UNK_IDX = vocab_data['UNK_IDX']
        self.BOS_IDX = vocab_data['BOS_IDX']
        self.EOS_IDX = vocab_data['EOS_IDX']

        return self


# Torch dataset over the encoded arrays (stored as long tensors).  An item is a dict with the SMILES row and the four context rows.
class DrugDataset(Dataset):
    def __init__(self, smiles, diseases, targets, kegg, wiki):
        self.smiles = torch.tensor(smiles, dtype=torch.long)
        self.diseases = torch.tensor(diseases, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.kegg = torch.tensor(kegg, dtype=torch.long)
        self.wiki = torch.tensor(wiki, dtype=torch.long)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {
            'smiles': self.smiles[idx],
            'disease': self.diseases[idx],
            'target': self.targets[idx],
            'kegg': self.kegg[idx],
            'wiki': self.wiki[idx]
        }


# Builds the preprocessor and the dataset from a DataFrame of ligands.  The length limits are fixed here: SMILES 256 characters (30 in SMOKE_TEST mode) and per row at most 45
# targets, 50 KEGG pathways, 50 WikiPathways and 25 diseases.  Returns (dataset, preprocessor).
def preprocess_data(CleanedData: pd.DataFrame):
    preprocessor = DrugDataPreprocessor(
        df=CleanedData,
        max_smiles_len=30 if SMOKE_TEST else 256,
        max_target_len=45,
        max_kegg_len=50,
        max_wiki_len=50,
        max_disease_len=25
    )

    smiles_enc, disease_enc, target_enc, kegg_enc, wiki_enc = preprocessor.get_encoded_data()

    dataset = DrugDataset(
        smiles=smiles_enc,
        diseases=disease_enc,
        targets=target_enc,
        kegg=kegg_enc,
        wiki=wiki_enc
    )

    return dataset, preprocessor


# Collates dataset items into one batch: {'smiles': (batch, length) tensor, 'context_dict': {name: (batch, length) tensor}}.  An empty batch gives {}.
def collate_fn(batch):

    if not batch:
        return {}

    sample = batch[0]
    context_keys = [k for k in sample.keys() if k != 'smiles']

    smiles_list = []
    context_lists = {key: [] for key in context_keys}

    for sample in batch:
        smiles_tensor = sample['smiles'].clone().detach()
        smiles_list.append(smiles_tensor)

        for key in context_keys:
            if isinstance(sample[key], torch.Tensor):
                ctx_tensor = sample[key]
            else:
                ctx_tensor = torch.tensor(sample[key], dtype=torch.long)

            context_lists[key].append(ctx_tensor)


    # rows already have a fixed length, so they stack; if lengths ever differed they would be padded with 0 (<PAD>) to the longest
    try:
        smiles_batch = torch.stack(smiles_list)
    except RuntimeError:
        smiles_batch = torch.nn.utils.rnn.pad_sequence(
            smiles_list,
            batch_first=True,
            padding_value=0
        )

    context_dict = {}
    for key in context_keys:
        try:
            ctx_batch = torch.stack(context_lists[key])
        except RuntimeError:
            ctx_batch = torch.nn.utils.rnn.pad_sequence(
                context_lists[key],
                batch_first=True,
                padding_value=0
            )
        context_dict[key] = ctx_batch

    return {
        'smiles': smiles_batch,
        'context_dict': context_dict
    }
