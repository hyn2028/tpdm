import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
import argparse
import datetime
import logging

from configs.ve import BMR_ZSR_256 as configs
from utils import clear, get_mask, fft2, ifft2
import models.ncsnpp
from sampling import get_predictor, get_corrector
from physics import zsr
import controllable_generation
import datasets
import tpdm_utils as tutils

BMR_CS_DPS_WEIGHT_MAP = {8: 0.002, 24: 0.007, 48: 0.01}

def main():
    # parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("label_path", type=Path, help="Path to the label data.")
    argparser.add_argument('--primary-ckpt', type=Path, default=Path(f"./checkpoints/BMR_ZSR_256_YZ/checkpoint.pth"), 
                            help="(optional) Path to primary model checkpoint.")
    argparser.add_argument('--auxiliary-ckpt', type=Path, default=Path(f"./checkpoints/BMR_ZSR_256_XY/checkpoint.pth"), 
                            help="(optional) Path to auxiliary model checkpoint.")
    argparser.add_argument('--problem-name', type=str, default="BMR_CS_256",
                            help="(optional) Name of the problem which will be used to name the result directory.")
    argparser.add_argument('--batch-size', type=int, default=6,
                            help="(optional) Batch size for sampling.")
    argparser.add_argument('--acc-factor', type=int, default=48,
                            help="(optional) Accerelation factor for CS-MRI (Poisson).")
    argparser.add_argument('--dps-weight', type=float, default=None,
                            help="(optional) Weight of DPS step.")
    argparser.add_argument('--K', type=tutils.int_or_float, default=2,
                            help="(optional) Sampling contribution ratio of primary and auxiliary models. " + 
                            "Int inputs use a deterministic scheduler, while float inputs use a stochastic scheduler." + 
                            "Int K means primary model and auxiliary model will be updated K-1 times and 1 time, respectively." + 
                            "Float K means 1-(1/K) probability of updating the primary model and 1/K probability of updating the auxiliary model.")
    args = argparser.parse_args()

    if args.dps_weight is None:
        if args.acc_factor in BMR_CS_DPS_WEIGHT_MAP:
            args.dps_weight = BMR_CS_DPS_WEIGHT_MAP[args.acc_factor]
            logging.warning("Using auto DPS weight for the given accerelation factor. For the best performance, please specify the DPS weight manually.")
        else:
            raise ValueError("There is no default DPS weight for the given accerelation factor. Please specify the DPS weight manually.")


    save_root = Path(f"./invp_results/{args.problem_name}/x{args.acc_factor}/K{args.K}/lamb{args.dps_weight}/{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}/")
    config = configs.get_config()
    config.eval.batch_size = args.batch_size


    # setup sampling environment
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    scaler = datasets.get_data_scaler(config)

    sde, sigmas = tutils.get_tpdm_sde(config)

    score_model_pri, score_model_aux = tutils.get_tpdm_models(config, args.primary_ckpt, args.auxiliary_ckpt)

    predictor = get_predictor(config.sampling.predictor)
    corrector = get_corrector(config.sampling.corrector)

    forward_mask = get_mask(torch.zeros(1, 1, 256, 256), 256, config.eval.batch_size,
                            type='poisson', acc_factor=args.acc_factor, center_fraction=None)
    forward_mask = forward_mask.to(config.device)
    forward_mask_full = forward_mask.repeat(256, 1, 1, 1)

    tpdm_cs_mri = controllable_generation.get_tpdm_cs_mri(sde,
                                                            predictor, corrector,
                                                            inverse_scaler,
                                                            config,
                                                            save_progress=True,
                                                            save_root=save_root,
                                                            dps_weight=args.dps_weight,
                                                            K=args.K,
                                                            denoise=True)


    # load data
    label, fname_list = tutils.load_tpdm_label_data(args.label_path)
    label = label.to(config.device)
    label = scaler(label)


    # prepair measurements
    label_kspace = fft2(label)
    measure_kspace = label_kspace * forward_mask_full
    measure_dagger = torch.real(ifft2(measure_kspace))


    # prepair result directory
    save_root.mkdir(parents=True, exist_ok=True)
    irl_types = ['measure', 'recon', 'label', 'measure_dagger']
    for t in irl_types:
        if t == 'recon':
            save_root_f = save_root / t / 'progress'
            save_root_f.mkdir(exist_ok=True, parents=True)
        else:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)


    # save input data
    plt.imsave(save_root / 'mask.png', clear(forward_mask), cmap='gray')
    for i, (fname, _) in enumerate(fname_list):
        just_name = fname.split('.')[0]
        plt.imsave(save_root / 'measure' / f'{just_name}.png', clear(inverse_scaler(torch.real(measure_kspace[i]))), cmap='gray')
        plt.imsave(save_root / 'measure_dagger' / f'{just_name}.png', clear(inverse_scaler(measure_dagger[i])), cmap='gray')
        plt.imsave(save_root / 'label' / f'{just_name}.png', clear(inverse_scaler(label[i])), cmap='gray')


    # start reconstruction
    recon = tpdm_cs_mri(score_model_pri, score_model_aux, measure_kspace, forward_mask_full)
    label = inverse_scaler(label)


    # save result
    for recon_img, label_one, (fname, _) in zip(recon, label, fname_list):
        just_name = fname.split('.')[0]
        plt.imsave(save_root / 'recon' / f'{just_name}.png', clear(recon_img), cmap='gray')
        np.save(str(save_root / 'label' / f'{just_name}.npy'), label_one.detach().cpu().numpy())
        np.save(str(save_root / 'recon' / f'{just_name}.npy'), recon_img.detach().cpu().numpy())


    # evaluate result
    tutils.print_and_save_eval_result(recon, label, save_root)


if __name__ == '__main__':
    main()