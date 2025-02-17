# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import glob
import os
import pprint
import re
import shutil
from operator import itemgetter
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import yaml
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
import mlflow
import numpy as np

import _init_paths
from core.config import config
from core.config import get_model_name
from core.config import update_config
from core.function import train
from core.function import validate
from core.loss import JointsMSELoss
from utils.dataset import init_paths, create_subset
from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint

import dataset
import models


def remove_keys(dict_, keys):
    for key in keys:
        if key in dict_:
            dict_.pop(key)
    return dict_


def prepare_training_set(dataset_dir, train_set, val_set, temp_path):
    dataset_files = {}
    for mode in ['train', 'val']:
        for file in glob.glob(os.path.join(dataset_dir, mode, '*.zip')):
            dataset_files[int(re.search('task_dji_[0-9]{4}', file).group().replace('task_dji_', ''))] = file

    test_set = glob.glob(os.path.join(dataset_dir, 'test', '*.zip'))
    val_set = itemgetter(*val_set)(dataset_files)
    if type(val_set) is not tuple:
        val_set = (val_set,)
    train_set = itemgetter(*train_set)(dataset_files)
    if type(train_set) is not tuple:
        train_set = (train_set,)

    init_paths(os.path.join(temp_path, 'annotations'))
    create_subset(train_set, temp_path, 'train')
    create_subset(val_set, temp_path, 'val')
    create_subset(test_set, temp_path, 'test', output_base_name='image_info_')


def prepare_config(temp_path, experiment_output_path, default_config_path, dataset_params,
                   train_params, deterministic=False):

    with open(default_config_path) as file:
        config = yaml.safe_load(file.read())
    _, config_name = os.path.split(default_config_path)
    config_name, config_ext = os.path.splitext(config_name)
    exp_config_path = os.path.join(temp_path,
                                   f"{config_name}_{str(datetime.datetime.now()).replace(' ', '_')}{config_ext}")
    config['DATASET']['ROOT'] = os.path.abspath(temp_path)
    config['OUTPUT_DIR'] = os.path.abspath(experiment_output_path)

    if dataset_params and type(dataset_params) is dict:
        config['DATASET'].update(dataset_params)
    if train_params and type(train_params) is dict:
        config['TRAIN'].update(train_params)
    if deterministic:
        config['CUDNN']['DETERMINISTIC'] = True

    with open(exp_config_path, 'w+') as file:
        yaml.safe_dump(config, file)

    return exp_config_path


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    parser.add_argument('--mlflow-run',
                        default=None,
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def train_loop(cfg_path, print_frequence=config.PRINT_FREQ, gpus='0', num_workers=4, mlflow_run=None):
    print(mlflow_run)

    if mlflow_run:
        mlflow.start_run(run_id=mlflow_run)
    
    update_config(cfg_path)
    args = edict({'cfg': cfg_path, 'frequent': print_frequence, 'gpus': gpus, 'workers': num_workers})
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.use_deterministic_algorithms(config.CUDNN.DETERMINISTIC)
    if config.CUDNN.DETERMINISTIC:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(
        config, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
                 final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    writer_dict['writer'].add_graph(model, (dump_input,), verbose=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    optimizer = get_optimizer(config, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.' + config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_perf = 0.0
    best_model = False

    if mlflow_run:
        mlflow.log_params(remove_keys(dict(config['DATASET']), ['DATASET', 'ROOT', 'DATA_FORMAT', 'TEST_SET', 'TRAIN_SET']))
        mlflow.log_params(remove_keys(dict(config['TRAIN']), []))
        mlflow.log_artifact(cfg_path)
        mlflow.log_param('lr_scheduler', lr_scheduler.__class__.__name__)

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                  criterion, final_output_dir, tb_log_dir,
                                  writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if mlflow_run:
            mlflow.log_metric('learning_rate', lr_scheduler.get_last_lr()[0], step=epoch)
            metrics = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc']
            for metric in metrics:
                mlflow.log_metric(metric, writer_dict[metric], step=epoch)

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')    
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    if mlflow_run:
        if os.path.exists(os.path.join(final_output_dir, 'final_state.pth.tar')):
            mlflow.log_artifact(os.path.join(final_output_dir, 'final_state.pth.tar'))
        if os.path.exists(os.path.join(final_output_dir, 'model_best.pth.tar')):
            mlflow.log_artifact(os.path.join(final_output_dir, 'model_best.pth.tar'))
        mlflow.end_run

    writer_dict['writer'].close()


def main():
    args = parse_args()
    train_loop(args.cfg, args.frequent, args.gpus, args.workers, args.mlflow_run)


if __name__ == '__main__':
    main()
