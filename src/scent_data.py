import numpy as np
import pandas as pd
import pyrfume
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from torch_geometric.utils import smiles
from math import isnan
from rdkit import Chem


class GoodscentsDataset(Dataset):
    def __init__(self, verbose=False):
        self.scent_onehot_encoder = None

        df_goodscents_odor2ID = pyrfume.load_data("goodscents/data_rw_odor.csv", remote=True)
        df_goodscents_ID2OPL = pyrfume.load_data("goodscents/data_rw_opl.csv", remote=True)
        df_goodscents_OPL2SMILES = pyrfume.load_data("goodscents/opl.csv", remote=True)

        common_indices = list(set(df_goodscents_ID2OPL.index).intersection(set(df_goodscents_odor2ID.index))) # list of TGSC IDs shared by the odor and the mapping dataset

        odor_id_raw, odor_tags_raw = list(df_goodscents_odor2ID.index), list(df_goodscents_odor2ID['Tags'])

        # Multiple rows may correspond to the same molecule, so combine those rows:
        odor_id_processed, odor_tags_processed = [], []
        OPL_list_processed = []
        id2index = {} # Maps TGSC IDs to indices of odor_id_processed
        for curr_id, curr_tag in zip(odor_id_raw, odor_tags_raw):
            if curr_tag is np.nan:
                curr_tag = []
            else:
                # Adjust for pesky str such as "['goaty','fresh','goat's','milk','cheese']"
                curr_tag = curr_tag.replace("['", '["').replace("','", '","').replace("']", '"]')
                curr_tag = eval(curr_tag)
                
            if curr_id in id2index: # in case when we reach a duplicate row
                odor_tags_processed[id2index[curr_id]] = list(set(odor_tags_processed[id2index[curr_id]] + curr_tag))
            elif curr_id in common_indices:
                # Need to filter out TGSC ID with no valid TGSC OPL ID, and the
                # if-else clause is because type(df_goodscents_ID2OPL.loc[curr_id]['TGSC OPL ID'])
                # can be <class 'pandas.core.series.Series'>, <class 'str'>, or <class 'float'>
                if type(df_goodscents_ID2OPL.loc[curr_id]['TGSC OPL ID']) is np.nan:
                    list_of_avail_OPL = []
                elif type(df_goodscents_ID2OPL.loc[curr_id]['TGSC OPL ID']) is float and isnan(float(df_goodscents_ID2OPL.loc[curr_id]['TGSC OPL ID'])):
                    list_of_avail_OPL = []
                elif type(df_goodscents_ID2OPL.loc[curr_id]['TGSC OPL ID']) is pd.core.series.Series:
                    list_of_avail_OPL = list(df_goodscents_ID2OPL.loc[curr_id]['TGSC OPL ID'])
                    list_of_avail_OPL = [avail_OPL for avail_OPL in list_of_avail_OPL if avail_OPL is not np.nan]
                else:
                    list_of_avail_OPL = [df_goodscents_ID2OPL.loc[curr_id]['TGSC OPL ID']]

                if len(list_of_avail_OPL) > 0:
                    id2index[curr_id] = len(odor_tags_processed)
                    OPL_list_processed.append(list_of_avail_OPL)
                    odor_tags_processed.append(curr_tag)
                    odor_id_processed.append(curr_id)

        # Construct the corresponding list of SMILES representation, whilst further
        # filtering out molecules with no known TGSC OPL ID, invalid smiles, or no scent labels:
        smiles_final = []
        odor_id_final, odor_tags_final = [], []
        for i, curr_id in enumerate(odor_id_processed):
            list_of_avail_OPL = OPL_list_processed[i]
            smiles_candidates = []
            for curr_OPL in list_of_avail_OPL:
                if curr_OPL in df_goodscents_OPL2SMILES.index:
                    smiles_candidates.append(df_goodscents_OPL2SMILES.loc[curr_OPL]['SMILES'])
            if verbose and len(set(smiles_candidates)) > 1:
                print(f"WARNING: multiple smiles_candidates for the same OPL ID: {smiles_candidates}")
            curr_odor_tags = odor_tags_processed[i] # if curr_odor_tags is [], skip (note: a tag can be "odorless")
            if len(smiles_candidates) > 0 and len(curr_odor_tags) > 0:
                curr_smiles = smiles_candidates[0]
                # Check if curr_smiles is a valid smiles string
                # See: https://github.com/rdkit/rdkit/issues/2430
                if Chem.MolFromSmiles(curr_smiles) is not None:
                    smiles_final.append(smiles_candidates[0])
                    odor_id_final.append(curr_id)
                    odor_tags_final.append(curr_odor_tags)

        if verbose:
            print(f"There are {len(odor_id_final)} molecules with scent labels and SMILES representation. ")

        self.inputs = [smiles.from_smiles(smiles_str) for smiles_str in smiles_final]

        self.odor_labels = odor_tags_final
        self.labels = self.load_odor_labels_as_onehot(self.odor_labels, scentClasses_path='scentClasses_combined.csv')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        smiles = self.inputs[idx]
        return smiles, label
    
    def transform_multiclass(self, onehot_encoder: OneHotEncoder, inp_labels: list) -> np.array:
        inp_labels = [[inp_label] for inp_label in inp_labels]
        return onehot_encoder.transform(inp_labels).toarray().sum(axis=0)
    
    def load_odor_labels_as_onehot(self, odor_labels: list, scentClasses_path: str = "scentClasses.csv") -> list:
        if self.scent_onehot_encoder is None:
            # Build the one hot encoder instance:
            scent_classes = pd.read_csv(scentClasses_path)
            scent_classes = scent_classes["Scent"].tolist()
            scent_classes = [[item] for item in scent_classes]
            self.scent_onehot_encoder = OneHotEncoder(handle_unknown='ignore')
            self.scent_onehot_encoder.fit(scent_classes)

        # Convert odor_labels to a list of one hot tensors:
        ret = []
        for odor_label in odor_labels:
            curr_onehot_numpy = self.transform_multiclass(self.scent_onehot_encoder, odor_label)
            ret.append(torch.tensor(curr_onehot_numpy))
        return ret


class LeffingwellDataset(Dataset):
    def __init__(self):
        self.scent_onehot_encoder = None

        df_leffingwell = pyrfume.load_data("leffingwell/leffingwell_data.csv", remote=True)
        df_smiles = list(df_leffingwell['smiles'])
        self.inputs = [smiles.from_smiles(smiles_str) for smiles_str in df_smiles]

        self.odor_labels = list(df_leffingwell['odor_labels_filtered'])
        self.odor_labels = [eval(odor_label) for odor_label in self.odor_labels]
        self.labels = self.load_odor_labels_as_onehot(self.odor_labels, scentClasses_path='scentClasses_combined.csv')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        smiles = self.inputs[idx]
        return smiles, label
    
    def transform_multiclass(self, onehot_encoder: OneHotEncoder, inp_labels: list) -> np.array:
        inp_labels = [[inp_label] for inp_label in inp_labels]
        return onehot_encoder.transform(inp_labels).toarray().sum(axis=0)
    
    def load_odor_labels_as_onehot(self, odor_labels: list, scentClasses_path: str = "scentClasses.csv") -> list:
        if self.scent_onehot_encoder is None:
            # Build the one hot encoder instance:
            scent_classes = pd.read_csv(scentClasses_path)
            scent_classes = scent_classes["Scent"].tolist()
            scent_classes = [[item] for item in scent_classes]
            self.scent_onehot_encoder = OneHotEncoder(handle_unknown='ignore')
            self.scent_onehot_encoder.fit(scent_classes)

        # Convert odor_labels to a list of one hot tensors:
        ret = []
        for odor_label in odor_labels:
            curr_onehot_numpy = self.transform_multiclass(self.scent_onehot_encoder, odor_label)
            # curr_class_indices = np.nonzero(curr_onehot_numpy)[0]
            ret.append(torch.tensor(curr_onehot_numpy))
        return ret


class LeffingwellGoodscentsDataset(Dataset):
    def __init__(self, verbose=False):
        self.verbose = verbose

        if self.verbose:
            print("Loading the Leffingwell dataset")
        self.leffingwell = LeffingwellDataset()

        if self.verbose:
            print("Loading the Goodscents dataset")
        self.goodscents = GoodscentsDataset(verbose=self.verbose)

        self.odor_labels = self.leffingwell.odor_labels + self.goodscents.odor_labels
        self.inputs = self.leffingwell.inputs + self.goodscents.inputs
        self.labels = self.leffingwell.labels + self.goodscents.labels

        self.odor_labels, self.inputs, self.labels = self.combine_duplicates(self.odor_labels, self.inputs, self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        smiles = self.inputs[idx]
        return smiles, label

    def combine_duplicates(self, odor_labels, inputs, labels):
        """
        Combine duplicate molecules using their InchiKey representations.
        """
        inchikey2index_lookup = {} # index refers to the index of the processed lists
        odor_labels_processed, inputs_processed, labels_processed = [], [], []
        for i, input in enumerate(inputs): # inputs is a list of pytorch_geometric graph objects
            curr_smiles = input.smiles
            curr_mol = Chem.MolFromSmiles(curr_smiles)
            curr_inchikey = Chem.MolToInchiKey(curr_mol)
            if curr_inchikey in inchikey2index_lookup: # duplicates found!
                if self.verbose:
                    print(f"Duplicates found: smiles={curr_smiles}")
                curr_index = inchikey2index_lookup[curr_inchikey]
                odor_labels_processed[curr_index] = list(set(odor_labels_processed[curr_index] + odor_labels[i]))

                prev_onehot = np.array(labels_processed[curr_index])
                curr_onehot = np.array(labels[i])
                new_onehot = np.logical_or(prev_onehot, curr_onehot).astype(int)
                labels_processed[curr_index] = torch.tensor(new_onehot)
                continue

            inchikey2index_lookup[curr_inchikey] = len(inputs_processed)
            odor_labels_processed.append(odor_labels[i])
            inputs_processed.append(input)
            labels_processed.append(labels[i])
        
        return odor_labels_processed, inputs_processed, labels_processed

# combines = LeffingwellGoodscentsDataset()