import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn import preprocessing
import sys
import argparse
import random
import datetime

sys.path.append('.')

from src.utils.utils import setup_logger
from src.network.interface import scMoE
from src.network.configs import *
import torch
import os
from tqdm import tqdm, trange

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
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4) # 1e-3
    parser.add_argument('--n_patches', type=int, default=8) # 1, 8, 16, 64
    parser.add_argument('--n_transformers', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=1) # 1,2,3,4
    parser.add_argument('--n_routers', type=int, default=1)
    parser.add_argument('--n_experts', type=int, default=16) # 4, 8, 16, 32
    parser.add_argument('--gate_loss_weight', type=float, default=1e-1) # 1e-1, 1e-2, 1e-3
    parser.add_argument('--decoder_loss_weight', type=float, default=1.) # 1e-1, 1e-2, 1e-3
    parser.add_argument('--top_k', type=int, default=2) 
    parser.add_argument('--patch', type=str2bool, default=True)
    parser.add_argument('--key', type=str, default='pPDm')

    return parser.parse_known_args()

def change_label(adata, batch):
    adata.obs['batch'] = batch
    return adata


def pre_ps(adata_list,sc_pre = None):
    adata_list_all = [ad_x.copy() for ad_x in adata_list]
    scalars = []
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

    return adata_list_all,scalars

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    technique = 'Dyngen'
    view = 'mean'
    data_path = f"UnitedNet_NatComm_data/{technique}"
    root_save_path = f"../saved_results/{technique}"
    args, _ = parse_args()
    device = torch.device(f'cuda:{args.device}')
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
    
    keys = args.key# 'pm', 'Pm', 'Dm', 'pPDm'

    aris = []
    r2s = []

    logger = setup_logger('./logs', '-', f'{technique}.txt')

    for cv_index in trange(1):
        print(f'============= cv: {cv_index} =============')

        adata_pPRm_train = sc.read_h5ad(f'{data_path}/adata_pPRm_train_cv_{cv_index}.h5ad')
        adata_pPRm_test = sc.read_h5ad(f'{data_path}/adata_pPRm_test_cv_{cv_index}.h5ad')
        n_labels = len(adata_pPRm_train.obs.label.unique())
        
        adata_pPRm_train = change_label(adata_pPRm_train,'train')
        adata_pPRm_test = change_label(adata_pPRm_test,'test')

        adatas_train = [None] * len(keys)
        seq_length = []

        for i in range(len(keys)):
            key = keys[i]
            adatas_train[i] = ad.AnnData(X=adata_pPRm_train.obsm[key], obs=pd.DataFrame(np.arange(adata_pPRm_train.obsm[key].shape[0])))
            adatas_train[i].obs['label'] = adata_pPRm_train.obs['label']
            adatas_train[i].obs['batch'] = adata_pPRm_train.obs['batch']
            seq_length.append(adata_pPRm_train.obsm[key].shape[1])

        adatas_test = [None] * len(keys)

        for i in range(len(keys)):
            key = keys[i]
            adatas_test[i] = ad.AnnData(X=adata_pPRm_test.obsm[key], obs=pd.DataFrame(np.arange(adata_pPRm_test.obsm[key].shape[0])))
            adatas_test[i].obs['label'] = adata_pPRm_test.obs['label']
            adatas_test[i].obs['batch'] = adata_pPRm_test.obs['batch']
    
        adatas_train,_ = pre_ps(adatas_train)   
        adatas_test,_ = pre_ps(adatas_test) 

        dyngen_config = configuer(sequence_length=seq_length, categories=n_labels, **model_kwargs)

        model = scMoE(root_save_path, device=device, technique=dyngen_config)
        metrics = model.train(adatas_train, adatas_test, verbose=True)

        aris.append(metrics['ari'])
        r2s.append(metrics['r2_off_diag'])

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
    print(keys)