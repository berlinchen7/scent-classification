import numpy as np
import pandas as pd
from tqdm import tqdm
from mol_utils import get_mol, get_tanimoto_K
from scent_data import LeffingwellGoodscentsDataset
from sklearn.preprocessing import OneHotEncoder
from vendi_score import vendi


def compute_molecular_vendi(smiles_list):
    moles_list = [get_mol(smile) for smile in smiles_list]
    assert None not in moles_list # Makes sure that there are no weird smiles that get_mol() is unable to process
    K = get_tanimoto_K(moles_list)
    vendi_score = vendi.score_K(K)
    return vendi_score

combined_dataset = LeffingwellGoodscentsDataset()
print("Finished loading the combined Leffingwell and Goodscents dataset.")

num_scents = combined_dataset.labels[0].shape[0]
vendi_scores = np.zeros((num_scents))
smell_smiles_map = {} # Maps a given odor (a str object) to the list of smiles that has that odor

for input, odor_labels in zip(combined_dataset.inputs, combined_dataset.odor_labels):
    curr_smiles = input.smiles
    for odor in odor_labels:
        if odor not in smell_smiles_map:
            smell_smiles_map[odor] = []
        smell_smiles_map[odor].append(curr_smiles)

scent_classes = pd.read_csv("scentClasses_combined.csv")
scent_classes = scent_classes["Scent"].tolist()
scent_classes = [[item] for item in scent_classes]
scent_onehot_encoder = OneHotEncoder(handle_unknown='ignore')
scent_onehot_encoder.fit(scent_classes)

count = 0
for odor in tqdm(smell_smiles_map):
    curr_onehot = scent_onehot_encoder.transform([[odor]]).toarray()
    if curr_onehot.any():
        curr_onehot_index = np.nonzero(curr_onehot)[1][0]
        vendi_scores[curr_onehot_index] = compute_molecular_vendi(smell_smiles_map[odor])
        count += 1
print(f"compute_molecular_vendi is called {count} times.")

with open('vendi_cache.npy', 'wb') as f:
    np.save(f, vendi_scores)
