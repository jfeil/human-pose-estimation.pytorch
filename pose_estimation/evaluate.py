import os
import numpy as np
import time
import logging
import shutil
import glob

import torch
import mlflow
from tqdm.notebook import tqdm

import _init_paths
import dataset
from models import pose_resnet
from core.config import update_config, config
from core.inference import get_max_preds, get_final_preds
from utils.visualization import draw_rectangle, draw_skeleton
from utils.dataset import init_paths
from core.loss import JointsMSELoss
from core.function import AverageMeter
from core.evaluate import accuracy
from utils.vis import save_debug_images


def eval_model(weights_path, test_loader):
    checkpoint_weights = torch.load(weights_path)
    
    state_dict = {}
    for i in checkpoint_weights:
        state_dict[i.replace('module.', '')] = checkpoint_weights[i]
    
    model = pose_resnet.get_pose_net(config, is_train=False)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.cuda()
    
    
    model.eval()
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    
    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    losses = AverageMeter()
    acc = AverageMeter()
    batch_time = AverageMeter()
    
    num_samples = len(test_loader.dataset)
    
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    
    
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(tqdm(test_loader, leave=False, desc="Epochs")):
            # compute output
            output = model(input.cuda())
            if config.TEST.FLIP_TEST:
                raise Exception("NYI")
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           test_loader.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
    
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0
    
                output = (output + output_flipped) * 0.5
    
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
    
            loss = criterion(output, target, target_weight)
    
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())
    
            acc.update(avg_acc, cnt)
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
    
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
    
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
    
            idx += num_images
    
            if True or i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(test_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)
    
                prefix = '{}_{}'.format(os.path.join('test', 'test'), i)
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)
        return test_loader.dataset.evaluate(config, all_preds, 'temp/test', all_boxes, image_path), all_preds, all_boxes, image_path


def evaluate_run(run_id, test_loader, temp_path='tmp'):
    init_paths(temp_path)
    mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=temp_path)
    config_file = glob.glob(f"{temp_path}/*.yaml")
    assert len(config_file) == 1
    config_file = config_file[0]
    
    weights = glob.glob(f"{temp_path}/*.pth.tar")
    assert len(weights) == 2
    results = []
    for weight in tqdm(weights, desc="Models"):
        upload_dict = {}
        results += [eval_model(weight, test_loader)]
    shutil.rmtree(temp_path)
    return results