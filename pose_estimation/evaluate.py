import os
import numpy as np
import time
import logging
import shutil
import glob

import torch
import mlflow
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt
import cv2

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
    
    weight_paths = glob.glob(f"{temp_path}/*.pth.tar")
    prefixes = [os.path.split(path)[1].replace('.pth.tar', '') for path in weight_paths]
    results = []

    for weight in tqdm(weight_paths, desc="Models"):
        upload_dict = {}
        results += [eval_model(weight, test_loader)]
    shutil.rmtree(temp_path)
    return prefixes, results


def upload_test_results(run_id, test_loader, temp_path='tmp', dry_run=False):
    prefixes, results = evaluate_run(run_id, test_loader, temp_path)
    upload_dict = {}
    for prefix, result in zip(prefixes, results):
        for key in result[0][0]:
            upload_dict[f"{prefix} {key.replace('(', '').replace(')', '')}"] = result[0][0][key]
    if not dry_run:
        with mlflow.start_run(run_id):
            mlflow.log_metrics(upload_dict)
    else:
        print(upload_dict)


class DisplayResults:

    def __init__(self, run_id, test_loader, temp_path='tmp', dry_run_results=None, frame_size=100):
        if not dry_run_results:
            _, self.results = evaluate_run(run_id, test_loader, temp_path)
        else:
            self.results = dry_run_results
        self.current_frame = 0
        self.total_frames = len(self.results[0][3])
        self.frame_size = frame_size

        # Text field widget
        self.text = widgets.Text(value=str(self.current_frame), description='Frame:')
        self.text.observe(self.on_frame_number_change, names='value')
        
        # Forward and backward buttons
        self.button_backward = widgets.Button(description='Backward')
        self.button_forward = widgets.Button(description='Forward')
        self.button_backward.on_click(self.on_backward_click)
        self.button_forward.on_click(self.on_forward_click)

    def display_fig(self):
        self.fig, self.ax = plt.subplots(1, 2)
        self.flat_axs = self.ax.flatten()
        self.update_display()

    def display_buttons(self):
        # Display widgets
        display(widgets.HBox([self.button_backward, self.text, self.button_forward]))

    
    # Function to handle frame number change
    def on_frame_number_change(self, change):
        new_frame = int(change.new)
        if new_frame >= 0 and new_frame < self.total_frames:
            self.current_frame = new_frame
            self.update_display()
    
    # Function to handle forward button click
    def on_forward_click(self, b):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.update_display()
    
    # Function to handle backward button click
    def on_backward_click(self, b):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_display()
    
    # Function to update display based on current_frame value
    def update_display(self):
        self.text.value = str(self.current_frame)
        # Add your code to display the frame based on the current_frame value
        # For demonstration, printing the current frame
        frame = cv2.imread(self.results[0][3][self.current_frame])
        for index, res in enumerate(self.results):
            all_preds = res[1]
            frame_new = draw_skeleton(frame, all_preds[self.current_frame])
            x,y = self.results[index][2][self.current_frame][0:2]
            x = int(x)
            y = int(y)
        
            frame_new = frame_new[:, :, ::-1]
            
            self.flat_axs[index].imshow(frame_new[max(0, y-self.frame_size):min(2160, y+self.frame_size),max(0,x-self.frame_size):min(3840,x+self.frame_size),:])
        self.fig.canvas.draw()