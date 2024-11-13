import numpy as np
import torch
import anndata as ad
import scanpy as sc
import torch.utils.data as D
import pickle

from cytomoe.network.constants import *

class MMDataset(D.Dataset):
    def __init__(self, modalities, labels):
        super().__init__()
        self.modalities = [
            torch.tensor(modality, dtype=torch.float) for modality in modalities
        ]

        if labels is None:
            labels = [-1 for _ in range(len(modalities[0]))]
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __getitem__(self, index):
        modalities = [modality[index] for modality in self.modalities]
        return modalities, self.labels[index]
    
    def __len__(self):
        return len(self.labels)
    

def create_dataloader_from_dataset(dataset, shuffle, batch_size):
    g = torch.Generator()

    return D.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
    )
    

def create_dataset(model, adatas, fit_label):
    modalities = [adata.X for adata in adatas]
    if str_label in adatas[0].obs.keys():
        if fit_label:
            labels = model.label_encoder.fit_transform(list(adatas[0].obs[str_label]))
        else:
            labels = model.label_encoder.transform(list(adatas[0].obs[str_label]))
    else:
        labels = None
    return MMDataset(modalities, labels)


def create_dataloader(model, adatas, shuffle=False, batch_size=512, fit_label=False):
    dataset = create_dataset(model, adatas, fit_label)
    return create_dataloader_from_dataset(dataset, shuffle, batch_size)


def create_joint_dataloader(
        model, adatas0, adatas1, shuffle=False, batch_size=512, fit_label=False
):
    dataset = D.ConcatDataset(
        [
            create_dataset(model, adatas0, fit_label),
            create_dataset(model, adatas1, fit_label),
        ]
    )
    return create_dataloader_from_dataset(dataset, shuffle, batch_size)


from sklearn import preprocessing
def patch_seq_pre_ps(adata_rna_raw, adata_ephys_raw, adata_morph_raw, cv, ind_dict,split=False):
    adata_rna, adata_ephys, adata_morph = adata_rna_raw.copy(), adata_ephys_raw.copy(), adata_morph_raw.copy()
    adatas_train, adatas_test = [], []
    assert (adata_rna.X >= 0).all(), "poluted input"
    for mod in [adata_rna, adata_ephys, adata_morph]:
        mod.obs['label'] = mod.obs['cell_type_TEM']
        if split:
            m_train = mod[ind_dict[cv]['train']]
            scaler = preprocessing.StandardScaler().fit(m_train.X)
            m_train.X = scaler.transform(m_train.X)

            m_test = mod[ind_dict[cv]['val']]
            scaler = preprocessing.StandardScaler().fit(m_test.X)
            m_test.X = scaler.transform(m_test.X)
        else:
            scaler = preprocessing.StandardScaler().fit(mod.X)
            mod.X = scaler.transform(mod.X)
            m_train = mod[ind_dict[cv]['train']]
            m_test = mod[ind_dict[cv]['val']]

        adatas_train.append(m_train)
        adatas_test.append(m_test)
    adatas_all = [ad.concat([m_train, m_test]) for m_train, m_test in zip(adatas_train, adatas_test)]
    return adatas_train, adatas_test, adatas_all