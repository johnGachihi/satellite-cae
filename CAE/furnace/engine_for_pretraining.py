import math
import sys
import time
from typing import Iterable

import torch
import torch.nn as nn

from models.modeling_finetune import PatchEmbed
import furnace.utils as utils
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def patchify(imgs, c, p=16):
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    return x

import torch

def equal_width_binning(tensor, num_bins):
    # Calculate min and max values
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Create bin edges
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1).to(tensor.device)
    
    # Digitize the tensor into bins
    # For each value, find the bin it belongs to using bucketization
    binned = torch.bucketize(tensor, bin_edges, right=True)
    
    # Adjust indices to be 0-based
    binned = binned - 1
    
    # Clamp values to ensure they're within valid range
    binned = torch.clamp(binned, 0, num_bins - 1)
    
    return binned, bin_edges


def loss_selector(loss_type, pred, target):
    if loss_type == 'mse':
        return F.mse_loss(pred, target, reduction="mean")
    elif loss_type == 'kld':
        return F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(target, dim=-1), reduction='mean')

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]


        samples, bool_masked_pos = batch

        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        # TODO arg-ize c            
        labels = patchify(samples, c=13)[bool_masked_pos]  # [B * num_masked_patches, p**2 * c]

        with torch.cuda.amp.autocast():
            outputs, latent, latent_target = model(samples, bool_masked_pos=bool_masked_pos)

            # outputs, _ = equal_width_binning(outputs, 10)
            # outputs = F.one_hot(outputs.flatten(), num_classes=10).float()
            # labels, _ = equal_width_binning(labels, 10)
            #labels = labels.flatten()
            
            # print(f"outputs {outputs.shape}")
            # print(f"labels {labels.shape}")
            # print(f"outputs {outputs[:20]}")
            # import pdb; pdb.set_trace()
            
            # loss_main = nn.CrossEntropyLoss()(input=outputs, target=labels)
            # loss_main = nn.KLDivLoss()(outputs.float(), labels.float())
            loss_main = F.mse_loss(outputs.float(), labels, reduction="mean") * 100
            loss_align = args.align_loss_weight * loss_selector('mse', latent.float(), latent_target.detach().float())
            loss = loss_main + loss_align

        # print(f"latent {latent.flatten(1)[0, ::1000]}")
        # print(f"latent_target {latent_target.flatten(1)[0, ::1000]}")
        # print(f"outputs {outputs[::100, :]}")
        # print(f"labels {labels[::100]}")
#        print(f"outputs {outputs.mean(1)[:30]} labels {labels.mean(1)[:30]}")
        # import pdb; pdb.set_trace()

        loss_value = loss.item()
        loss_main_value = loss_main.item()
        loss_align_value = loss_align.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # mlm_acc = (outputs.max(-1)[1] == labels).float().mean().item()
        # metric_logger.update(mlm_acc=mlm_acc)
        # if log_writer is not None:
        #     log_writer.update(mlm_acc=mlm_acc, head="loss")


        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_main=loss_main_value)
        metric_logger.update(loss_align=loss_align_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_main=loss_main_value, head="loss_main")
            log_writer.update(loss_align=loss_align_value, head="loss_align")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(now_time, "Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
