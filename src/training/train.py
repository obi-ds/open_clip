import json
import logging
import math
import os
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from .distributed import is_master
from .precision import get_autocast

import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            model_out = model(images, texts)
            losses = loss(**model_out, output_dict=True)
            total_loss = sum(losses.values()) / args.accum_freq
            losses["loss"] = total_loss
        backward(total_loss, scaler)

        if ((i + 1) % args.accum_freq) > 0:
            continue

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Text LR: {optimizer.param_groups[-1]['lr']:5f} "
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"],
                "text_lr": optimizer.param_groups[-1]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            # TODO only run when possible
            fig = unwrap_model(model).log_visualizations(model_out['images'], model_out['images_noisy'],
                                                         model_out['images_cleaned'], model_out['reconstructions'], step)
            log_folder = os.path.join(args.checkpoint_path, 'logs')
            os.makedirs(log_folder, exist_ok=True)
            fig_path = os.path.join(log_folder, f'visualization_step_{step}.png')
            fig.savefig(fig_path)
            plt.close(fig)
            

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

                # Log visualizations if available
                if hasattr(unwrap_model(model), 'log_visualizations') and 'reconstructions' in model_out:
                    #fig = unwrap_model(model).log_visualizations(model_out['images'], model_out['images_noisy'],
                    #                                       model_out['images_cleaned'], model_out['reconstructions'], step)
                    wandb.log({"ecg_visualization": wandb.Image(fig)}, step=step)
                    plt.close(fig)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def evaluate_instruct_basic(
        model,
        data,
        epoch,
        args,
        eos_token_id,
        positive_token_id,
        negative_token_id,
        tb_writer=None,
        prefix='',
        ignore_index=-100,
        step=None,
):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_gen_loss = 0.0
        all_predictions, all_labels = [], []
        (
            all_diagnosis_scores,
            all_diagnosis_labels,
            all_no_context_diagnosis_scores,
            all_no_context_diagnosis_labels,
            all_context_diagnosis_scores,
            all_context_diagnosis_labels,
        ) = [], [], [], [], [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    model_logits = model_out['logits']
                    model_labels = model_out['labels']

                    (
                        mask,
                        diagnosis_mask,
                        no_context_diagnosis_select,
                        no_context_diagnosis_indexes,
                        context_diagnosis_select,
                        context_diagnosis_indexes,
                    ) = get_masks(
                        model_labels=model_labels,
                        positive_token_id=positive_token_id,
                        negative_token_id=negative_token_id,
                        eos_token_id=eos_token_id,
                        ignore_index=ignore_index,
                        device=device
                    )

                    labels = model_labels[mask]
                    logits = model_logits[mask]

                    diagnosis_labels = model_labels[diagnosis_mask]
                    diagnosis_logits = model_logits[diagnosis_mask]
                    no_context_diagnosis_labels = model_labels[
                        no_context_diagnosis_select, no_context_diagnosis_indexes
                    ]
                    no_context_diagnosis_logits = model_logits[
                        no_context_diagnosis_select, no_context_diagnosis_indexes
                    ]
                    context_diagnosis_labels = model_labels[
                        context_diagnosis_select, context_diagnosis_indexes
                    ]
                    context_diagnosis_logits = model_logits[
                        context_diagnosis_select, context_diagnosis_indexes
                    ]


                    all_predictions.append(logits.argmax(dim=-1).cpu())
                    all_labels.append(labels.cpu())
                    all_diagnosis_scores.append(diagnosis_logits[:, positive_token_id].cpu())
                    all_diagnosis_labels.append(diagnosis_labels.cpu())
                    all_no_context_diagnosis_scores.append(no_context_diagnosis_logits[:, positive_token_id].cpu())
                    all_no_context_diagnosis_labels.append(no_context_diagnosis_labels.cpu())
                    all_context_diagnosis_scores.append(context_diagnosis_logits[:, positive_token_id].cpu())
                    all_context_diagnosis_labels.append(context_diagnosis_labels.cpu())
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic

                    batch_size = images.shape[0]
                    gen_loss = compute_generative_loss(token_logits=model_logits, token_labels=model_labels)

                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                    )

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"{prefix} Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            generative_metrics = compute_generative_metrics(
                predictions=torch.cat(all_predictions),
                labels=torch.cat(all_labels),
                diagnosis_scores=torch.cat(all_diagnosis_scores),
                diagnosis_labels=torch.cat(all_diagnosis_labels),
                no_context_diagnosis_scores=torch.cat(all_no_context_diagnosis_scores),
                no_context_diagnosis_labels=torch.cat(all_no_context_diagnosis_labels),
                context_diagnosis_scores=torch.cat(all_context_diagnosis_scores),
                context_diagnosis_labels=torch.cat(all_context_diagnosis_labels),
                prefix=prefix
            )
            metrics.update(
                {**generative_metrics, "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({f"{prefix}generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics

def get_masks(
        model_labels: torch.Tensor,
        positive_token_id,
        negative_token_id,
        ignore_index,
        eos_token_id,
        device

):
    indexes = torch.arange(len(model_labels), device=device)
    mask = (model_labels != ignore_index) & (model_labels != eos_token_id)
    diagnosis_mask = (model_labels == positive_token_id) | (model_labels == negative_token_id)
    diagnosis_mask_flipped = torch.fliplr(diagnosis_mask)

    no_context_diagnosis_indexes = diagnosis_mask.long().argmax(dim=1)
    context_diagnosis_indexes = -diagnosis_mask_flipped.long().argmax(dim=1) - 1 + model_labels.shape[-1]
    first_context_indexes = (~diagnosis_mask & mask).long().argmax(dim=1)

    no_context_diagnosis_mask = (no_context_diagnosis_indexes != 0)
    no_context_mask = (
            ((no_context_diagnosis_indexes < first_context_indexes) | (first_context_indexes == 0))
            & no_context_diagnosis_mask
    )
    context_mask = (
            ((context_diagnosis_indexes != no_context_diagnosis_indexes) | ~no_context_mask)
            & no_context_diagnosis_mask
    )

    # Positions where diagnosis occurs without anything before - with no information
    no_context_diagnosis_indexes = no_context_diagnosis_indexes[no_context_mask]
    no_context_diagnosis_select = indexes[no_context_mask]

    # Positions where diagnosis occurs after everything occurs - with all information
    context_diagnosis_indexes = context_diagnosis_indexes[context_mask]
    context_diagnosis_select = indexes[context_mask]

    return (
        mask,
        diagnosis_mask,
        no_context_diagnosis_select,
        no_context_diagnosis_indexes,
        context_diagnosis_select,
        context_diagnosis_indexes,
    )


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

def compute_generative_loss(token_logits, token_labels, ignore_index=-100, reduction='mean'):
    return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels, ignore_index=ignore_index, reduction=reduction)

def compute_generative_metrics(
        predictions,
        labels,
        diagnosis_scores,
        diagnosis_labels,
        no_context_diagnosis_scores,
        no_context_diagnosis_labels,
        context_diagnosis_scores,
        context_diagnosis_labels,
        prefix):

    metrics = dict()

    metrics[f'{prefix}accuracy'] = accuracy_score(labels, predictions)
    metrics[f'{prefix}auc'] = roc_auc_score(diagnosis_labels, diagnosis_scores)

    if len(no_context_diagnosis_labels) and len(set(no_context_diagnosis_labels)) > 1:
        metrics[f'{prefix}auc_no_context'] = roc_auc_score(no_context_diagnosis_labels, no_context_diagnosis_scores)

    if len(context_diagnosis_labels) and len(set(context_diagnosis_labels)) > 1:
        metrics[f'{prefix}auc_context'] = roc_auc_score(context_diagnosis_labels, context_diagnosis_scores)

    return metrics

def compute_probability(loss):
    mask = loss != 0
    loss_mean = (loss * mask).sum(dim=1) / mask.sum(dim=1)
    return torch.exp(-loss_mean)