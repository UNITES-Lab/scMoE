import anndata as ad
import numpy as np
import scanpy as sc
from sklearn import preprocessing
import sys
import argparse
import random
import datetime
import time
sys.path.append('.')

from src.utils.utils import setup_logger
from src.network.interface import scMoE
from src.network.configs import *
import torch
import os

from sklearn.model_selection import StratifiedKFold

def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='dyngen')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--task', type=str, default='unsupervised') # unsupervised, supervised
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=4e-3) # 1e-3
    parser.add_argument('--n_patches', type=int, default=64) # 1, 8, 16, 64
    parser.add_argument('--n_transformers', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=4)# 1,2,3,4
    parser.add_argument('--n_routers', type=int, default=1)
    parser.add_argument('--n_experts', type=int, default=32) # 4, 8, 16, 32
    parser.add_argument('--gate_loss_weight', type=float, default=1e-1) # 1e-1, 1e-2, 1e-3
    parser.add_argument('--decoder_loss_weight', type=float, default=1.) # 1e-1, 1e-2, 1e-3
    parser.add_argument('--top_k', type=int, default=4) 
    parser.add_argument('--patch', type=str2bool, default=True)

    return parser.parse_known_args()


def pre_ps(adata_list,sc_pre = None):
    adata_list_all = [ad_x.copy() for ad_x in adata_list]
    scalars = []
    assert (adata_list_all[0].X>=0).all(), "poluted input"
    for idx, mod in enumerate(adata_list_all):
        t_x = mod.X
        if sc_pre != None:
            scaler = sc_pre[idx]
        else:
            scaler = preprocessing.StandardScaler().fit(t_x)
        t_x = scaler.transform(t_x)
        mod.X = t_x
        adata_list_all[idx] = mod
        scalars.append(scaler)

    return adata_list_all, scalars

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def concat_adatas(adatas_train, adatas_test):
    return [ad.concat([adata_train, adata_test]) for adata_train, adata_test in zip(adatas_train, adatas_test)]

def partitions(celltype, n_partitions, seed=0):
    """
    adapted from https://github.com/AllenInstitute/coupledAE-patchseq
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Safe to ignore warning - there are celltypes with a low sample number that are not crucial for the analysis.
    with warnings.catch_warnings():
        skf = StratifiedKFold(n_splits=n_partitions, random_state=seed, shuffle=True)

    # Get all partition indices from the sklearn generator:
    ind_dict = [{'train': train_ind, 'val': val_ind} for train_ind, val_ind in
                skf.split(X=np.zeros(shape=celltype.shape), y=celltype)]
    return ind_dict

def bound_matric(adata):
    scaler = preprocessing.StandardScaler().fit(adata.X)
    adata.X = scaler.transform(adata.X)
    return adata

def patch_seq_pre_ps(adata_rna_raw, adata_ephys_raw, adata_morph_raw, cv, ind_dict,split=False):
    adata_rna, adata_ephys, adata_morph = adata_rna_raw.copy(), adata_ephys_raw.copy(), adata_morph_raw.copy()
    adatas_train, adatas_test = [], []
    for mod in [adata_rna, adata_ephys, adata_morph]:
        mod.obs['label'] = mod.obs['cell_type_TEM']
        if split:
            m_train = mod[ind_dict[cv]['train']]
            m_test = mod[ind_dict[cv]['val']]

        else:
            m_train = mod[ind_dict[cv]['train']]
            m_test = mod[ind_dict[cv]['val']]

        adatas_train.append(m_train)
        adatas_test.append(m_test)
    adatas_all = [ad.concat([m_train, m_test]) for m_train, m_test in zip(adatas_train, adatas_test)]
    return adatas_train, adatas_test, adatas_all


if __name__ == '__main__':
    technique = 'patchseq'
    data_path = f"UnitedNet_NatComm_data/patchseq"
    root_save_path = f"saved_results/{technique}"
    device=torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    view = 'mean'
    args, _ = parse_args()
    seed_everything(seed=0)

    model_kwargs = {
        "task": args.task,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "train_epochs": args.train_epochs,
        "lr": args.lr,
        "n_patches": args.n_patches,
        "n_transformers": args.n_transformers,
        "n_head": args.n_head,
        "n_routers": args.n_routers,
        "n_experts": args.n_experts,
        "gate_loss_weight": args.gate_loss_weight,
        "decoder_loss_weight": args.decoder_loss_weight,
        "top_k": args.top_k,
        "patch": args.patch,
    }

    adata_rna_raw = sc.read_h5ad(f'{data_path}/adata_RNA_TEM.h5ad')
    adata_ephys_raw = sc.read_h5ad(f'{data_path}/adata_Ephys_TEM.h5ad')
    adata_morph_raw = sc.read_h5ad(f'{data_path}/adata_Morph_TEM.h5ad')
    adata_rna_raw = bound_matric(adata_rna_raw)
    adata_ephys_raw = bound_matric(adata_ephys_raw)
    adata_morph_raw = bound_matric(adata_morph_raw)

    ind_dict = partitions(adata_rna_raw.obs['cell_type_TEM'], n_partitions=10, seed=0)
    seq_length = [adata_rna_raw.X.shape[1], adata_ephys_raw.X.shape[1], adata_morph_raw.X.shape[1]]
    
    n_labels = len(adata_rna_raw.obs['cell_type_TEM'].unique())

    dyngen_config = pathseq_config(sequence_length=seq_length, categories=n_labels, **model_kwargs)
    
    aris = []
    r2s = []
    accs = []
    logger = setup_logger('./logs', '-', f'{technique}.txt')

    cv=9
    adatas_train, adatas_test, all_data = patch_seq_pre_ps(adata_rna_raw,adata_ephys_raw,adata_morph_raw,cv,ind_dict,split=True)
    
    save_path = f"{root_save_path}/{cv}"

    model = scMoE(save_path, device=device, technique=dyngen_config)
    metrics = model.train(adatas_train, all_data, verbose=True)

    aris.append(metrics['ari'])
    r2s.append(metrics['r2_off_diag'])
    accs.append(metrics['acc'])
    model.eval(all_data)

    time.sleep(3)

    print('===================== Final =====================')
    print(f'ARI: {np.mean(aris):.2f}±{np.std(aris):.2f}')
    print(f'R^2: {np.mean(r2s):.2f}±{np.std(r2s):.2f}')

    logger.info('')
    logger.info(datetime.datetime.now())
    logger.info(model_kwargs)
    logger.info('ARI: {:.1f} R^2: {:.1f} / ARI: {:.2f} R^2: {:.2f}'.format(np.mean(aris), np.mean(r2s), np.mean(aris), np.mean(r2s)))
    logger.info('{:.2f}+{:.2f}'.format(np.mean(aris), np.std(aris)))
    logger.info('{:.2f}+{:.2f}'.format(np.mean(r2s), np.std(r2s)))
    logger.info(f"ARI: {[f'{ari:.2f}' for ari in aris]}")
    logger.info(f"R^2: {[f'{r2:.2f}' for r2 in r2s]}")
    logger.info(f'=================================')