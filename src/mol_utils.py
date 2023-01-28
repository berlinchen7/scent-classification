import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect


def get_mol(smiles_or_mol):
    """
    moses: Loads SMILES/molecule into RDKit's object
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def get_tanimoto_K(mols, fp="morgan"):
    N = len(mols)
    K = np.zeros((N, N))
    if fp == "rdk":
        fps = [Chem.RDKFingerprint(x) for x in mols]
    elif fp == "morgan":
        fps = [GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in mols]
    else:
        fps = [fp(x) for e in mols]
    for i in range(N):
        for j in range(i, N):
            K[i, j] = K[j, i] = DataStructs.FingerprintSimilarity(
                fps[i], fps[j]
            )
    return K