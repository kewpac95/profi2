import numpy as np
import torch
from configparser import ConfigParser
from profis.dataset import ProfisDataset, load_charset
from profis.net_RL import Profis
from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from torch.utils.data import DataLoader
import deepsmiles as ds

def initialize_profis(config_path):
    config = ConfigParser()
    config.read(config_path)
    model = Profis(
        fp_size=int(config["MODEL"]["fp_len"]),
        fc1_size=int(config["MODEL"]["fc1_size"]),
        fc2_size=int(config["MODEL"]["fc2_size"]),
        hidden_size=int(config["MODEL"]["hidden_size"]),
        latent_size=int(config["MODEL"]["latent_size"]),
        transformer_layers=int(config["MODEL"]["transformer_layers"]),
        nhead=int(config["MODEL"]["nhead"]),
        eps_coef=float(config["MODEL"]["eps_coef"]),
        dropout=float(config["MODEL"]["dropout"]),
        alphabet_size=len(
            load_charset(f'data/{config["RUN"]["out_encoding"].lower()}_alphabet.txt')
        ),
        max_seq_len=int(config["MODEL"]["max_seq_len"]),
    )
    return model

class ValidityChecker:
    def __init__(self, encoding):
        self.encoding = encoding
        if encoding == "deepsmiles":
            self.decoder = ds.Converter(rings=True, branches=True)
    def __call__(self, seq):
        if self.encoding == "selfies":
            return True
        if self.encoding == "deepsmiles":
            try:
                smiles = self.decoder.decode(seq)
            except ds.exceptions.DecodeError:
                return False
        else:
            smiles = seq
        if Chem.MolFromSmiles(smiles, sanitize=True) is None:
            return False
        else:
            return True

def KRFP_score(mol, fp: torch.Tensor):
    score = 0
    key = pd.read_csv("data/KlekFP_keys.txt", header=None)
    fp_len = fp.shape[0]
    for i in range(fp_len):
        if fp[i] == 1:
            frag = Chem.MolFromSmarts(key.iloc[i].values[0])
            score += mol.HasSubstructMatch(frag)
    return score / torch.sum(fp).item()

def ECFP_score(mol, fp: torch.Tensor):
    score = 0
    fp_len = fp.shape[0]
    ECFP_reconstructed = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_len)
    for i in range(fp_len):
        if ECFP_reconstructed[i] and fp[i]:
            score += 1
    return score / torch.sum(fp).item()

def try_QED(mol):
    try:
        qed = QED.qed(mol)
    except:
        qed = 0
    return qed

def smiles2sparse_KRFP(smiles):
    mol = Chem.MolFromSmiles(smiles)
    keys = "data/KlekFP_keys.txt"
    klek_keys = [line.strip() for line in open(keys)]
    klek_keys_mols = list(map(Chem.MolFromSmarts, klek_keys))
    fp_list = []
    for i, key in enumerate(klek_keys_mols):
        if mol.HasSubstructMatch(key):
            fp_list.append(1)
        else:
            fp_list.append(0)
    return np.array(fp_list)

def smiles2dense_KRFP(smiles):
    mol = Chem.MolFromSmiles(smiles)
    keys = "data/KlekFP_keys.txt"
    klek_keys = [line.strip() for line in open(keys)]
    klek_keys_mols = list(map(Chem.MolFromSmarts, klek_keys))
    fp_list = []
    for i, key in enumerate(klek_keys_mols):
        if mol.HasSubstructMatch(key):
            fp_list.append(i)
    return np.array(fp_list)

def sparse2dense(sparse, return_numpy=True):
    dense = []
    for idx, value in enumerate(sparse):
        if value == 1:
            dense.append(idx)
    if return_numpy:
        return np.array(dense)
    else:
        return dense

def dense2sparse(dense, fp_len=4860):
    sparse = np.zeros(fp_len, dtype=np.int8)
    for value in dense:
        sparse[value] = 1
    return np.array(sparse)

def encode(df, model, device, batch=1024):
    dataset = ProfisDataset(df, fp_len=model.fp_size)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)
    mus = []
    logvars = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in dataloader:
            X, _ = batch
            X = X.to(device)
            mu, logvar = model.encode(X)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
        mus = np.concatenate(mus, axis=0)
        logvars = np.concatenate(logvars, axis=0)
    return mus, logvars

def smiles2sparse_ECFP(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = np.array(GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    return np.array(fp)

def smiles2dense_ECFP(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = np.array(GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    return sparse2dense(fp)

def decode_seq_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()