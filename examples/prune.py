import copy
import itertools
import os
import random
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
from collections import defaultdict
import networkx as nx
from glob import glob
from tqdm import tqdm

try:
    import wandb
except Exception as e:
    pass

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSPseudolabeledSubset
from torchvision.transforms import ToPILImage

from cifar10_c_transform import get_d
from utils import projection_simplex
from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool, get_model_prefix, move_to
from train import train, evaluate, infer_predictions
from algorithms.initializer import initialize_algorithm, infer_d_out
from transforms import initialize_transform
from models.initializer import initialize_model
from configs.utils import populate_defaults

import configs.supported as supported

import torch.multiprocessing

def main():
    
    ''' Arg defaults are filled in according to examples/configs/ '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, default='fmow')
    parser.add_argument('--algorithm', choices=supported.algorithms, default='ERM')
    parser.add_argument('--root_dir',
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).',
                        default='./data')

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to download the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

    # Unlabeled Dataset
    parser.add_argument('--unlabeled_split', default=None, type=str, choices=wilds.unlabeled_splits,  help='Unlabeled split to use. Some datasets only have some splits available.')
    parser.add_argument('--unlabeled_version', default=None, type=str, help='WILDS unlabeled dataset version number.')
    parser.add_argument('--use_unlabeled_y', default=False, type=parse_bool, const=True, nargs='?', 
                        help='If true, unlabeled loaders will also the true labels for the unlabeled data. This is only available for some datasets. Used for "fully-labeled ERM experiments" in the paper. Correct functionality relies on CrossEntropyLoss using ignore_index=-100.')

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--unlabeled_loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--unlabeled_n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--unlabeled_batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

    # Model
    parser.add_argument('--model', choices=supported.models, default='resnet50')
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument('--noisystudent_add_dropout', type=parse_bool, const=True, nargs='?', help='If true, adds a dropout layer to the student model of NoisyStudent.')
    parser.add_argument('--noisystudent_dropout_rate', type=float)
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')
    parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')

    # NoisyStudent-specific loading
    parser.add_argument('--teacher_model_path', type=str, help='Path to NoisyStudent teacher model weights. If this is defined, pseudolabels will first be computed for unlabeled data before anything else runs.')

    # Transforms
    parser.add_argument('--transform', choices=supported.transforms)
    parser.add_argument('--additional_train_transform', choices=supported.additional_transforms, help='Optional data augmentations to layer on top of the default transforms.')
    parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--resize_scale', type=float)
    parser.add_argument('--max_token_length', type=int)
    parser.add_argument('--randaugment_n', type=int, help='Number of RandAugment transformations to apply.')

    # Objective
    parser.add_argument('--loss_function', choices=supported.losses)
    parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--dann_penalty_weight', type=float)
    parser.add_argument('--dann_classifier_lr', type=float)
    parser.add_argument('--dann_featurizer_lr', type=float)
    parser.add_argument('--dann_discriminator_lr', type=float)
    parser.add_argument('--afn_penalty_weight', type=float)
    parser.add_argument('--safn_delta_r', type=float)
    parser.add_argument('--hafn_r', type=float)
    parser.add_argument('--use_hafn', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--self_training_lambda', type=float)
    parser.add_argument('--self_training_threshold', type=float)
    parser.add_argument('--pseudolabel_T2', type=float, help='Percentage of total iterations at which to end linear scheduling and hold lambda at the max value')
    parser.add_argument('--soft_pseudolabels', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--algo_log_metric')
    parser.add_argument('--process_pseudolabels_function', choices=supported.process_pseudolabels_functions)

    # Model selection
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Misc
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

    # Weights & Biases
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')
    
    
    parser.add_argument('--sweep_dir', type=str, default='./sweep/fmow')
    parser.add_argument('--features', action='store_true')
    parser.add_argument('--num_models', type=int, default=10)
    parser.add_argument('--algo_z', choices=['DRO', 'SRM', 'ERM', 'uniform_prior', 'greedy', 'laplacian'], default='SRM')
    parser.add_argument('--lambda_', type=float, default=1)
    parser.add_argument('--sparsity', type=int, default=None)
    parser.add_argument('--severity', type=int, default=None)
    parser.add_argument('--corrupt_type', choices=['noise', 'blur', 'weather', 'digital'], default='noise')

    config = parser.parse_args()
    config = populate_defaults(config)

    config.save_dir = os.path.join('./experiments', config.split_scheme, config.algo_z)
    os.makedirs(config.save_dir, exist_ok=True)

    split_scheme_to_src_domain = {
        'official': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'time_before_2004':[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'time_mid':[0, 1, 2, 3, 4, 5, 10, 11, 12 ,13, 14, 15],
    }

    split_scheme_to_target_domain = {
        'official': [14, 15],
        'time_before_2004':[0, 1],
        'time_mid':[7, 8],
    }

    torch.set_default_device('cuda')
    np.set_printoptions(precision=4)

    # Set device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(config.device) > device_count:
            raise ValueError(f"Specified {len(config.device)} devices, but only {device_count} devices found.")

        config.use_data_parallel = len(config.device) > 1
        device_str = ",".join(map(str, config.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        config.device = torch.device("cuda")
    else:
        config.use_data_parallel = False
        config.device = torch.device("cpu")

    # Set random seed
    set_seed(config.seed)

    if config.features:
        y = torch.load(os.path.join(config.sweep_dir, 'y.pkl'), map_location='cuda')
        pred = torch.load(os.path.join(config.sweep_dir, 'pred.pkl'), map_location='cuda')
        region = torch.load(os.path.join(config.sweep_dir, 'region.pkl'), map_location='cuda')

        if config.severity:
            corrupted_pred = torch.load(os.path.join(config.sweep_dir, f'{config.corrupt_type}_{config.severity}_pred.pkl'), map_location='cuda')
            pred['test'] = corrupted_pred['test']


        src_domain = split_scheme_to_src_domain[config.split_scheme]
        target_domain = split_scheme_to_target_domain[config.split_scheme]

        num_envs = len(src_domain)
        num_models = config.num_models

        n_steps = 50
        eta_q = 1e-1
        eta_z = 3e-2
        lambda_ = config.lambda_
        prior = np.ones(num_envs) / num_envs

        def evaluate(z, test_accs, val_accs):
            print(f'Step {step},', end='')
            accs_test, accs_val = [], []
            p_test, p_val = 0, 0
            for i in range(num_models):
                p_test += z[i] * torch.cat([pred['test'][i][target_domain[0]], pred['test'][i][target_domain[1]]], dim=0).softmax(dim=1)
                p_val += z[i] * torch.cat([pred['val'][i][target_domain[0]], pred['val'][i][target_domain[1]]], dim=0).softmax(dim=1)
            correct_test = (p.argmax(dim=1) == torch.cat([y['test'][i][target_domain[0]], y['test'][i][target_domain[1]]], dim=0))
            correct_val = (p.argmax(dim=1) == torch.cat([y['val'][i][target_domain[0]], y['val'][i][target_domain[1]]], dim=0))
            for r in region['test'][0][target_domain[0]].unique():
                idx_test = torch.cat([region['test'][0][target_domain[0]], region['test'][0][target_domain[1]]], dim=0) == r
                acc_test = correct_test[idx_test].float().mean()
                idx_val = torch.cat([region['val'][0][target_domain[0]], region['val'][0][target_domain[1]]], dim=0) == r
                acc_val = correct_val[idx_val].float().mean()
                accs_test.append(acc_test.item())
                accs_val.append(acc_val.item())
            print(f' test_acc: {correct_test.float().mean().item():.4f}', end='')
            print(f' val_acc: {correct_val.float().mean().item():.4f}', end='')
            print(f' worst_region_acc_test: {np.min(accs_test):.4f}', end='')
            print(f' worst_region_acc_val: {np.min(accs_val):.4f}')
            test_accs.append(np.min(accs_test))
            val_accs.append(np.min(accs_val))
            print('z', z.cpu().data.numpy(), 'q', q.cpu().data.numpy())
            return test_accs, val_accs


        if config.algo_z == 'DRO':
            lambda_ = 0
        elif config.algo_z == 'SRM':
            prior = np.load(os.path.join('./centrality', f'fmow_closeness.npy'))
        elif config.algo_z == 'ERM':
            eta_q = 0
        elif config.algo_z == 'uniform_prior':
            prior = np.ones(num_envs) / num_envs
        elif config.algo_z == 'laplacian':
            dis_matrix = np.load(os.path.join('./dis_matrices', f'fmow.npy'))
            G = nx.from_numpy_array(dis_matrix)
            l = torch.tensor(nx.normalized_laplacian_matrix(G).toarray(), dtype=torch.float32)

        prior = torch.tensor(prior)
        q = torch.ones(num_envs) / num_envs
        z = torch.ones(num_models) / num_models
        z.requires_grad = True
        optim_z = torch.optim.SGD([z], lr=eta_z)
        test_accs, val_accs = [], []
        for step in range(n_steps):
            evaluate(z, test_accs, val_accs)
            losses = torch.zeros(num_envs)
            for im, m in enumerate(src_domain):
                p = 0
                for i in range(num_models):
                    p += z[i] * torch.cat([pred['id_val'][i][m], pred['id_test'][i][m]], dim=0).softmax(dim=1)
                losses[im] = F.nll_loss(p.log(), torch.cat([y['id_val'][i][m], y['id_test'][i][m]], dim=0))
                if config.algo_z == 'laplacian':
                    q[im] += eta_q * losses[im].data
                else:
                    q[im] += eta_q * (losses[im].data - lambda_ * (q[im] - prior[im]))
            if config.algo_z == 'laplacian':
                q -= eta_q * lambda_ * (l @ q)
            projection_simplex(q)
            loss = torch.dot(losses, q)
            optim_z.zero_grad()
            loss.backward()
            optim_z.step()
            z.detach_()
            projection_simplex(z)
            z.requires_grad = True
        torch.save(test_accs, os.path.join(config.save_dir, f'test_accs.pkl'))
        torch.save(val_accs, os.path.join(config.save_dir, f'val_accs.pkl'))

    else:
        # Data
        full_dataset = wilds.get_dataset(
            dataset=config.dataset,
            version=config.version,
            root_dir=config.root_dir,
            download=config.download,
            split_scheme=config.split_scheme,
            **config.dataset_kwargs)
        
        transform = initialize_transform(
            transform_name=config.transform,
            config=config,
            dataset=full_dataset,
            is_training=False)
        
        if config.severity is not None:
            corrupt_names = {'noise': ['Gaussian Noise', 'Shot Noise', 'Impulse Noise'], 
                             'blur': ['Defocus Blur', 'Glass Blur', 'Motion Blur', 'Zoom Blur'],
                             'weather': ['Snow', 'Frost', 'Brightness'], 
                             'digital': ['Contrast', 'Elastic', 'Pixelate', 'JPEG']
                            }
            d = get_d()
            transform_ = lambda x: transform(ToPILImage()(np.uint8(d[random.choice(corrupt_names[config.corrupt_type])](x, config.severity))))
        else:
            transform_ = transform

        grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=config.groupby_fields
        )

        region_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['region']
        )

        datasets = defaultdict(dict)
        for split in ['train', 'id_val', 'id_test', 'val', 'test']:
            datasets[split]['dataset'] = full_dataset.get_subset(
                split,
                frac=config.frac,
                transform=transform_)
            
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=grouper,
                batch_size=512,
                **config.loader_kwargs)
            
        
        d_out = infer_d_out(datasets['test']['dataset'], config)

        y = defaultdict(list)
        pred = defaultdict(list)
        region = defaultdict(list)

        with torch.no_grad():
            for save_path in glob(os.path.join(config.sweep_dir, '*/*best_model.pth')):
                config.pretrained_model_path = save_path
                model = initialize_model(config, d_out)
                model.eval()
                model.cuda()
                for split in ['id_val', 'id_test', 'val', 'test']:
                    ps, ys, gs, rs = [], [], [], []
                    for batch in tqdm(datasets[split]['loader']):
                        x, y_true, metadata = batch
                        p = model(x.cuda())
                        g = grouper.metadata_to_group(metadata)
                        r = region_grouper.metadata_to_group(metadata)
                        ps.append(p.detach().cpu())
                        ys.append(y_true)
                        gs.append(g)
                        rs.append(r)
                    ps = torch.cat(ps, dim=0)
                    ys = torch.cat(ys, dim=0)
                    gs = torch.cat(gs, dim=0)
                    rs = torch.cat(rs, dim=0)
                    ys_env, ps_env, rs_env = [], [], []
                    for i in range(16):
                        idx = gs == i
                        ys_env.append(ys[idx])
                        ps_env.append(ps[idx])
                        rs_env.append(rs[idx])
                    y[split].append(ys_env)
                    pred[split].append(ps_env)
                    region[split].append(rs_env)
        # torch.save(pred, os.path.join(config.sweep_dir, f'{config.corrupt_type}_{config.severity}_pred.pkl'))
        torch.save(y, os.path.join(config.sweep_dir, 'y.pkl'))
        torch.save(pred, os.path.join(config.sweep_dir, 'pred.pkl'))
        torch.save(region, os.path.join(config.sweep_dir, 'region.pkl'))

if __name__=='__main__':
    main()
