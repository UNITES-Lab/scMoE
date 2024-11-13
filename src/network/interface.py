import numpy as np
import torch
import torch.utils.data as D
import os
from tqdm import tqdm
# import tabulate
from sklearn.utils import class_weight
from sklearn.metrics import (
    normalized_mutual_info_score, 
    adjusted_rand_score, r2_score, 
    confusion_matrix,
    )
from sklearn.utils.multiclass import unique_labels

from scipy.optimize import linear_sum_assignment
from src.network.loss import *

from src.network.model import Model
from src.network.constants import *
from src.dataloader import create_dataloader
from src.network.schedule import Schedule
from src.utils.utils import (
    amplify_value_dictionary_by_sample_size, 
    sum_value_dictionaries,
    inplace_combine_tensor_lists,
    concat_tensor_lists,
    )
from tabulate import tabulate
import copy

class scMoE:
    def __init__(self, save_path=None, device="cpu", technique=None):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.device = device
        self._create_model_for_technique(technique)

    def _set_device(self):
        self.model = self.model.to(device=self.device)
        self.model.device_in_use = self.device

    def _create_model_for_technique(self, technique):
        # current suported config: dlpfc_config
        self._create_model_from_config(technique)

    def _create_model_from_config(self, config):
        self.model = Model(config)
        self.model.save_path = self.save_path
        self._set_device()
    
    def train(self, adatas_train, adatas_val=None, save_path=None, init_classify=False, verbose=False):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_path = save_path
        if str_label in adatas_train[0].obs.keys():
            labels = adatas_train[0].obs[str_label]
            self.model.class_weights = list(
                class_weight.compute_class_weight(
                    "balanced", classes=np.unique(labels), y=labels
                )
            )
            
        dataloader_train = create_dataloader(
            self.model,
            adatas_train,
            shuffle=True,
            batch_size=self.model.config[str_train_batch_size],
            fit_label=True,
        )
        
        
        if adatas_val is None:
            adatas_val = adatas_train
        dataloader_test = create_dataloader(
            self.model,
            adatas_val,
            shuffle=False,
            batch_size=self.model.config[str_train_batch_size],
            fit_label=False,
        )
        metrics = run_train(
            self.model, dataloader_train, dataloader_test, verbose=verbose
        )
        return metrics

    def collect_grad(self, adatas_train, adatas_val=None, save_path=None, init_classify=False, verbose=False):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_path = save_path
        if str_label in adatas_train[0].obs.keys():
            labels = adatas_train[0].obs[str_label]
            self.model.class_weights = list(
                class_weight.compute_class_weight(
                    "balanced", classes=np.unique(labels), y=labels
                )
            )
            
        dataloader_train = create_dataloader(
            self.model,
            adatas_train,
            shuffle=True,
            batch_size=self.model.config[str_train_batch_size],
            fit_label=True,
        )
        
        
        if adatas_val is None:
            adatas_val = adatas_train
        dataloader_test = create_dataloader(
            self.model,
            adatas_val,
            shuffle=False,
            batch_size=self.model.config[str_train_batch_size],
            fit_label=False,
        )
        metrics = run_train(
            self.model, dataloader_train, dataloader_test, verbose=verbose
        )
        return metrics
    


    def eval(self, dataset):
        dataloader_eval = create_dataloader(
            self.model,
            dataset,
            shuffle=False,
            batch_size=self.model.config[str_train_batch_size],
            fit_label=True,
        )
        with torch.no_grad():
            self.model.eval()
            metrics,pred_results = run_evaluate(self.model, dataloader_eval)
            headers = ["Metrics", "Value"]
            values = list(metrics.items())
            print(tabulate(values, headers=headers))
    
        pred, label = pred_results
        pred_text = self.model.label_encoder.inverse_transform(pred)
        label_text = self.model.label_encoder.inverse_transform(label)
        metrics['pred_text'] = pred_text
        metrics['label_text'] = label_text
        
        return metrics


def run_train(model, dataloader_train, dataloader_val, verbose=False, plot=False):
    ''''''
    parameters = model.parameters()
    criterion_ce = torch.nn.CrossEntropyLoss().to(device=model.device_in_use)
    criterion_recon = torch.nn.MSELoss().to(device=model.device_in_use)
    loss_weight = model.config[str_train_loss_weight]
    ddc_loss = DDCLoss(model, loss_weight)
    optimizer = torch.optim.Adam(parameters, lr=model.config[str_lr])

    best_ari = 0.
    for epoch in tqdm(range(model.config[str_train_epochs])):
        epoch += 1
        model.cur_epoch = epoch
        model.train()
        task_loss = []
        router_loss = []
        correct, total = 0, 0
        for modalities, labels in dataloader_train:
            if hasattr(labels, 'to'):
                labels = labels.to(device=model.device_in_use)
            probs, recons = model(modalities, labels)
            for i in range(len(recons)):
                for j in range(len(recons[i])):
                    if i+j == 0:
                        # continue
                        loss = criterion_recon(recons[i][j], modalities[i].to(model.device_in_use))
                    else:
                        loss += criterion_recon(recons[i][j], modalities[i].to(model.device_in_use))
            loss *= model.config[str_decoder_loss_weight]
            if 'supervised' == model.config[str_task]:
                loss += criterion_ce(probs, labels) * 10
            loss += ddc_loss(model) * 0.1
            task_loss.append(loss.item())
            loss += model.config[str_gate_loss_weight] * model.gate_loss()
            router_loss.append(loss.item()-task_loss[-1])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=1)
            optimizer.step()

            predicted, pred_idx = torch.max(probs, dim=1)
            total += labels.size(0)
            correct += (pred_idx == labels).sum().item()
        
        print(f" [Epoch {epoch}/{model.config[str_train_epochs]}] Task Loss: {np.mean(task_loss):.2f}, Router Loss: {np.mean(router_loss):.2f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for modalities, labels in dataloader_val:
                labels = labels.to(device=model.device_in_use)
                probs, recons = model(modalities, labels)
                predicted, pred_idx = torch.max(probs, dim=1)
                total += labels.size(0)
                correct += (pred_idx == labels).sum().item()

        if epoch % model.config[str_checkpoint] == 0:
            model.save_model(f"{str_train}_epoch_{epoch}")
            if verbose:
                print(f"model saved at {model.save_path}/{str_train}_epoch_{epoch}.pt", "\n")

        if verbose:
            if epoch % 1 == 0:
                metrics,_ = run_evaluate(model, dataloader_val)
                if best_ari < metrics["ari"]:
                    best_ari = metrics["ari"]
                    model.save_model(f"{str_train}_best")
                    if verbose:
                        print(f"best model saved at {model.save_path}/{str_train}_best.pt", "\n")

                headers = ["Metrics", "Value"]
                values = list(metrics.items())
                print(tabulate(values, headers=headers))
        model.train()
    model.load_model(f"{str_train}_best")
    return metrics


def inplace_combine_tensor_lists(lists, new_list):
    """\
    In place add a new (nested) tensor list to current collections.
    This operation will move all concerned tensors to CPU.
    """
    if len(lists) == 0:
        for new_l in new_list:
            if isinstance(new_l, list):
                l = []
                inplace_combine_tensor_lists(l, new_l)
                lists.append(l)
            else:
                lists.append([new_l if type(new_l) == np.ndarray else new_l.detach().cpu()])
    else:
        for l, new_l in zip(lists, new_list):
            if isinstance(new_l, list):
                inplace_combine_tensor_lists(l, new_l)
            else:
                l.append(new_l if type(new_l) == np.ndarray else new_l.detach().cpu())

from src.utils.utils import inplace_combine_tensor_lists, concat_tensor_lists
def run_evaluate(model, dataloader):
    num_modalities = len(model.encoders)
    all_predicted = []
    all_recons = [[[] for _ in range(num_modalities)] for _ in range(num_modalities)]
    with torch.no_grad():
        for modalities, labels in dataloader:
            labels = labels.to(device=model.device_in_use)
            probs, recons = model(modalities, labels, use_cluster = True)
            predicted, pred_idx = torch.max(probs, dim=1)
            all_predicted.extend(pred_idx.cpu().numpy())
            for i, list_of_lists in enumerate(recons):
                for j, tensor_list in enumerate(list_of_lists):
                    all_recons[i][j].append(tensor_list.cpu().numpy())

    all_predicted = np.array(all_predicted)
    for i in range(num_modalities):
        for j in range(num_modalities):
            all_recons[i][j] = np.concatenate(all_recons[i][j], axis=0)
    return evalaute_outputs(dataloader, all_predicted, all_recons), [all_predicted, dataloader.dataset.labels]


from src.utils.evaluate import ordered_cmat
def evalaute_outputs(dataloader, predictions, recons):
    if not isinstance(dataloader.sampler, D.SequentialSampler):
        raise Exception("Please only evaluate outputs with non-shuffling dataloader.")
    dataset = dataloader.dataset
    labels = dataset.labels
    labels = labels if type(labels)==np.ndarray else labels.numpy()
    predictions = predictions if type(predictions)==np.ndarray else predictions.numpy()
    recons = recons if type(predictions)==np.ndarray else predictions.numpy()
    r2s = [
        [
            r2_score(modality if type(modality)==np.ndarray else modality.cpu().numpy(), translation if type(translation)==np.ndarray else translation.cpu().numpy())
            for translation in translations
        ]
        for modality, translations in zip(dataset.modalities, recons)
    ]

    accuracy, conf_mat = ordered_cmat(labels,predictions)

    metrics = {
        "r2": np.array(r2s),
        "confusion": conf_mat,
        "r2_off_diag": np.mean(np.array(r2s)[np.triu_indices_from(np.array(r2s), k=1)]),
        "acc": accuracy,
        "ari": adjusted_rand_score(labels, predictions),
        "nmi": normalized_mutual_info_score(
            labels, predictions, average_method="geometric"
        ),
    }
    return metrics

def average_dictionary_values_by_sample_size(dictionary, sample_size):
    if sample_size < 1:
        raise Exception("Please use positive count to average dictionary values.")
    for key in dictionary:
        dictionary[key] /= sample_size
    return dictionary


def run_schedule(runner):
    def wrapper_run_schedule(
            model,
            dataloader,
            schedule=None,
            train_model=False,
            infer_model=False,
            best_model_path=None,
            give_losses=False,
            verbose=False
    ):
        if train_model and schedule is not None and schedule.name == str_classification:
            for _ in range(len(dataloader.dataset.modalities)*2):
                outputs = runner(
                    model,
                    dataloader,
                    schedule,
                    train_model,
                    infer_model,
                    best_model_path,
                    give_losses=give_losses,
                    verbose=verbose
                )
            return outputs
        else:
            return runner(
                model, dataloader, schedule, train_model, infer_model, best_model_path,
                give_losses=give_losses, verbose=verbose
            )

    return wrapper_run_schedule

@run_schedule
def run_through_dataloader(
        model,
        dataloader,
        schedule=None,
        train_model=False,
        infer_model=False,
        best_model_path=None,
        give_losses=False,
        verbose=False
):
    all_outputs = []
    all_losses = {}

    for modalities, labels in dataloader:
        outputs = model(modalities, labels)
        if schedule is not None:
            losses = schedule.step(model, train_model)
            losses = amplify_value_dictionary_by_sample_size(losses, len(labels))
            all_losses = sum_value_dictionaries(all_losses, losses)

        if infer_model:
            inplace_combine_tensor_lists(all_outputs, outputs)

    if all_losses:
        all_losses = average_dictionary_values_by_sample_size(
            all_losses, len(dataloader.dataset)
        )
        for ls_name in all_losses:
            if ('ddc' in ls_name) or ('cross_entropy' in ls_name):
                ls_name_hd = '_'.join(ls_name.split('_')[:-1])
                head_losses = {k: all_losses[k] for k in all_losses.keys() if ls_name_hd in k}
                current_best_head = torch.tensor(int(min(head_losses, key=head_losses.get).split('_')[-1]))
                if hasattr(model, 'potential_best_head'):
                    model.potential_best_head.append(current_best_head)
                model.best_head = current_best_head
                break

    if best_model_path is not None:
        if len(model.potential_best_head)>0:
            cur_bc_heads = torch.bincount(torch.tensor(model.potential_best_head))
            if any(cur_bc_heads >= model.config[str_train_epochs]//3):
                model.best_head = torch.argmax(cur_bc_heads)
        else:
            model.best_head = torch.tensor(0, dtype=torch.long)
        schedule.check_and_save_best_model(model, all_losses, best_model_path, verbose=verbose)
    if give_losses:
        assert len(all_losses.keys()) > 0, 'wrong losses, the losses are empty'
        return all_losses
    else:
        return concat_tensor_lists(all_outputs)