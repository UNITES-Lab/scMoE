
import numpy as np
import scanpy as sc
from sklearn import preprocessing
import sys
import argparse
import random

sys.path.append('.')

from src.utils.utils import setup_logger
from src.network.interface import scMoE
from src.network.configs import *
import torch
import os
from sklearn.neighbors import NearestNeighbors

import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='dyngen')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--task', type=str, default='unsupervised') # unsupervised, supervised
    parser.add_argument('--batch_size', type=int, default= 256)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-2) # 1e-3
    parser.add_argument('--n_patches', type=int, default=4) # 1, 8, 16, 64
    parser.add_argument('--n_transformers', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=4)# 1,2,3,4
    parser.add_argument('--n_routers', type=int, default=1)
    parser.add_argument('--n_experts', type=int, default=32) # 4, 8, 16, 32
    parser.add_argument('--gate_loss_weight', type=float, default=1e-1) # 1e-1, 1e-2, 1e-3
    parser.add_argument('--decoder_loss_weight', type=float, default=1.0) # 1e-1, 1e-2, 1e-3
    parser.add_argument('--top_k', type=int, default=4) 
    parser.add_argument('--patch', type=str2bool, default=True)

    return parser.parse_known_args()


def change_label(adata, batch):
    adata.obs['batch'] = batch
    return adata


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

    return adata_list_all,scalars

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def split_data(test_batch, adata_rna_all, adata_morph_all, adata_mrna_niche_all):
    adata_rna_train = adata_rna_all[adata_rna_all.obs['batch'] != test_batch]
    adata_morph_train = adata_morph_all[adata_morph_all.obs['batch'] != test_batch]
    adata_mrna_niche_train = adata_mrna_niche_all[adata_mrna_niche_all.obs['batch'] != test_batch]

    adata_rna_test = adata_rna_all[adata_rna_all.obs['batch'] == test_batch]
    adata_morph_test = adata_morph_all[adata_morph_all.obs['batch'] == test_batch]
    adata_mrna_niche_test = adata_mrna_niche_all[adata_mrna_niche_all.obs['batch'] == test_batch]

    return [adata_rna_train, adata_morph_train, adata_mrna_niche_train], [adata_rna_test, adata_morph_test,
                                                                          adata_mrna_niche_test]

def change_label(adata,batch):
    adata.obs['batch'] = batch
    adata.obs['imagecol'] = adata.obs['array_col']
    adata.obs['imagerow'] = adata.obs['array_row']
    adata.obs['label'] = adata.obs['cell_type']
    return adata

if __name__ == '__main__':
    technique = 'DBiTSeq'
    data_path = f"UnitedNet_NatComm_data/DBiTSeq"
    root_save_path = f"saved_results/{technique}"
    device=torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    view = 'mean'
    args, _ = parse_args()
    seed_everything(seed=0)

    test_batches = ['s1d1', 's1d2', 's1d3', 's2d1', 's2d4', 's2d5', 's3d3', 's3d6', 's3d7',
       's3d10', 's4d1', 's4d8', 's4d9']

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

    adata_niche_rna_train = sc.read_h5ad(f'{data_path}/adata_niche_rna_train.h5ad')
    adata_niche_rna_test = sc.read_h5ad(f'{data_path}/adata_niche_rna_test.h5ad')

    adata_rna_train = sc.read_h5ad(f'{data_path}/adata_rna_train.h5ad')
    adata_rna_test = sc.read_h5ad(f'{data_path}/adata_rna_test.h5ad')

    adata_protein_train = sc.read_h5ad(f'{data_path}/adata_protein_train.h5ad')
    adata_protein_test = sc.read_h5ad(f'{data_path}/adata_protein_test.h5ad')
    
    adata_rna_train = change_label(adata_rna_train,'train')
    adata_protein_train=change_label(adata_protein_train,'train')
    adata_niche_rna_train=change_label(adata_niche_rna_train,'train')

    adata_rna_test = change_label(adata_rna_test,'test')
    adata_protein_test = change_label(adata_protein_test,'test')
    adata_niche_rna_test = change_label(adata_niche_rna_test,'test')

    adatas_train = [adata_rna_train, adata_protein_train, adata_niche_rna_train]
    adatas_test = [adata_rna_test, adata_protein_test, adata_niche_rna_test]

    adatas_all = []
    for ad_train, ad_test in zip(adatas_train,adatas_test):
        ad_all = ad_train.concatenate(ad_test,batch_key='sample')
        ad_all = change_label(ad_all,'test')
        adatas_all.append(ad_all)
    adatas_all,_ = pre_ps(adatas_all)  

    
    adatas_train, ccale = pre_ps(adatas_train)   
    adatas_test,_ = pre_ps(adatas_test)

    seq_length = [adatas_train[0].X.shape[1], adatas_train[1].X.shape[1], adatas_train[2].X.shape[1]]
    n_labels = len(adatas_train[0].obs.label.unique())

    dyngen_config = dbitseq_config(sequence_length=seq_length, categories=n_labels, **model_kwargs)

    aris = []
    r2s = []
    logger = setup_logger('./logs', '-', f'{technique}.txt')

    model = scMoE(root_save_path, device=device, technique=dyngen_config)
    metrics = model.train(adatas_train, adatas_test, verbose=True)
    logger.info(f'{metrics}')
    train_metric = model.eval(adatas_train)
    all_metric = model.eval(adatas_all)
    test_metric = model.eval(adatas_test)

    n_train = adatas_train[0].X.shape[0]

    united_clus=list(all_metric['pred_text'])
    coord=np.array((list(adatas_all[0].obs['array_row'].astype('int')),
                list(adatas_all[0].obs['array_col'].astype('int')))).T

    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(coord)
    distances,indices = nbrs.kneighbors(coord)

    united_clus_new=[]
    for indi,i in enumerate(united_clus):
        np.array(united_clus)[(indices[indi])]
        occurence_count=Counter(np.array(united_clus)[(indices[indi])])
        united_clus_new.append(occurence_count.most_common(1)[0][0])

    cluster_pl = sns.color_palette('tab20',20)
    color_list = [cluster_pl[5],
        cluster_pl[1],
        cluster_pl[2],
        cluster_pl[4],
        cluster_pl[11],
        cluster_pl[6],
        cluster_pl[3],
        cluster_pl[7],
        cluster_pl[8],
        cluster_pl[0],
        cluster_pl[12]
        ]
    plt.figure(figsize=(6,5))
    print(united_clus_new)
    for idx,clus_id in enumerate(set(united_clus_new)):
        
        plt.scatter(adatas_all[0].obs['array_row'][(united_clus_new==clus_id)],
                adatas_all[0].obs['array_col'][(united_clus_new==clus_id)],
                color=color_list[idx],cmap='tab20')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("full_data_dbit.svg")

    ###############################

    plt.figure(figsize=(6,5))
    for idx,clus_id in enumerate(set(united_clus_new)):
        
        plt.scatter(adatas_all[0].obs['array_row'][:n_train][(united_clus_new[:n_train]==clus_id)],
                adatas_all[0].obs['array_col'][:n_train][(united_clus_new[:n_train]==clus_id)],
                color=color_list[idx],cmap='tab20')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("train_data_dbit.svg")

    plt.figure(figsize=(6,5))
    for idx,clus_id in enumerate(set(united_clus_new)):
        if (united_clus_new[n_train:]==clus_id).sum() == 0:
            continue
        plt.scatter(adatas_all[0].obs['array_row'][n_train:][(united_clus_new[n_train:]==clus_id)],
                adatas_all[0].obs['array_col'][n_train:][(united_clus_new[n_train:]==clus_id)],
                color=color_list[idx],cmap='tab20')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_data_dbit.svg")