import torch
import torchmetrics as tm
import numpy as np

import os
import argparse
from typing import Union

from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint
from sde_lib import VESDE


def get_tpdm_sde(config):
    sigmas = mutils.get_sigmas(config)

    if config.training.sde.lower() == 'vesde':
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    else:
        raise NotImplementedError("TPDM is only implemented for VESDE")
    
    return sde, sigmas


def get_tpdm_models(config, ckpt_pri_path, ckpt_aux_path):
    score_model_pri = mutils.create_model(config)
    score_model_aux = mutils.create_model(config)

    ema_pri = ExponentialMovingAverage(score_model_pri.parameters(), decay=config.model.ema_rate)
    ema_aux = ExponentialMovingAverage(score_model_aux.parameters(), decay=config.model.ema_rate)
    state_pri = dict(step=0, model=score_model_pri, ema=ema_pri)
    state_aux = dict(step=0, model=score_model_aux, ema=ema_aux)

    state_pri = restore_checkpoint(ckpt_pri_path, state_pri, config.device, skip_optimizer=True)
    state_aux = restore_checkpoint(ckpt_aux_path, state_aux, config.device, skip_optimizer=True)
    ema_pri.copy_to(score_model_pri.parameters())
    ema_aux.copy_to(score_model_aux.parameters())

    return score_model_pri, score_model_aux


def load_tpdm_label_data(path):
    fname_list = [str(fname.name) for fname in path.glob("*.npy")]
    fname_list = sorted(fname_list, key=lambda x: float(x.split(".")[0]))
    fname_list = [(f"{i:03d}.npy", True) if f"{i:03d}.npy" in fname_list else (f"{i:03d}.npy", False) 
                                    for i in range(256)]
    assert len(fname_list) == 256

    print("Loading all data ...")
    all_img = []
    for fname in fname_list:
        fname, isfile = fname
        if isfile:
            img = np.load(os.path.join(path, fname)).squeeze()
            if np.issubdtype(img.dtype.type, np.integer):
                img = img.astype(np.float32) / 255
            elif np.issubdtype(img.dtype.type, np.floating):
                img = img.astype(np.float32)
            else:
                raise NotImplementedError(f"Image type {img.dtype.type} is not supported")
            img = torch.from_numpy(img)
        else:
            img = torch.zeros((256, 256), dtype=torch.float32)
        h, w = img.shape
        assert (h, w) == (256, 256)
        img = img.view(1, 1, h, w)
        all_img.append(img)
    all_img = torch.cat(all_img, dim=0)
    print(f"Data loaded shape: {all_img.shape}, min: {all_img.min()}, max: {all_img.max()}")
    print("Note: please check the data range is in about [0, 1]")
    
    return all_img, fname_list


def eval_recon_result(volume_recon, volume_label, plane, clip=True):
    if clip:
        volume_recon = torch.clip(volume_recon, 0.0, 1.0)
        volume_label = torch.clip(volume_label, 0.0, 1.0)

    if plane == "coronal":
        pass
    elif plane == "sagittal":
        volume_recon = volume_recon.permute((2, 1, 0, 3))
        volume_label = volume_label.permute((2, 1, 0, 3))
        pass
    elif plane == "axial":
        volume_recon = volume_recon.permute((3, 1, 0, 2))
        volume_label = volume_label.permute((3, 1, 0, 2))
        pass
    else:
        raise ValueError(f"Unknown plane {plane}")

    psnr = tm.functional.peak_signal_noise_ratio(volume_recon, volume_label, data_range=1.0 if clip else None)
    psnr = psnr.item()
    ssim = tm.functional.structural_similarity_index_measure(volume_recon, volume_label, data_range=1.0 if clip else None)
    ssim = ssim.item()

    return psnr, ssim


def print_and_save_eval_result(volume_recon, volume_label, save_root, clip=True):

    # evaluate result
    psnr_c, ssim_c = eval_recon_result(volume_recon, volume_label, "coronal", clip=clip)
    psnr_s, ssim_s = eval_recon_result(volume_recon, volume_label, "sagittal", clip=clip)
    psnr_a, ssim_a = eval_recon_result(volume_recon, volume_label, "axial", clip=clip)

    # print result
    print("\n<Evaluation result>")
    print(f"PSNR (coronal) : {psnr_c:.4f}, SSIM (coronal) : {ssim_c:.4f}")
    print(f"PSNR (sagittal) : {psnr_s:.4f}, SSIM (sagittal) : {ssim_s:.4f}")
    print(f"PSNR (axial) : {psnr_a:.4f}, SSIM (axial) : {ssim_a:.4f}")

    # save result
    with open(save_root / 'result.txt', 'w') as f:
        f.write(f"PSNR (coronal) : {psnr_c:.4f}, SSIM (coronal) : {ssim_c:.4f}\n")
        f.write(f"PSNR (sagittal) : {psnr_s:.4f}, SSIM (sagittal) : {ssim_s:.4f}\n")
        f.write(f"PSNR (axial) : {psnr_a:.4f}, SSIM (axial) : {ssim_a:.4f}\n")


def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid number: {value}")
        

def check_K(K: Union[int, float]):
    if isinstance(K, int):
        if K < 2:
            raise ValueError(f"K should be greater than 1 when K is integer, but got {K}")
    elif isinstance(K, float):
        if K <= 0.0:
            raise ValueError(f"K should be greater than 0 when K is float, but got {K}")
    else:
        assert False, f"Unexpected type {type(K)} for K"


def is_primary_tern(i: int, K: Union[int, float]) -> bool:
    if isinstance(K, int):
        return i % K != K - 1
    elif isinstance(K, float):
        return (torch.rand(1) > 1/K).item()
    else:
        assert False, f"Unexpected type {type(K)} for K"