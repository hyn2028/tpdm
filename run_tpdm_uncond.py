import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import argparse
import datetime

from configs.ve import BMR_ZSR_256 as configs
from utils import clear
import models.ncsnpp
from sampling import get_predictor, get_corrector
import controllable_generation
import datasets
import tpdm_utils as tutils


def main():
    # parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--primary-ckpt', type=Path, default=Path(f"./checkpoints/BMR_ZSR_256_YZ/checkpoint.pth"), 
                           help="(optional) Path to primary model checkpoint.")
    argparser.add_argument('--auxiliary-ckpt', type=Path, default=Path(f"./checkpoints/BMR_ZSR_256_XY/checkpoint.pth"), 
                           help="(optional) Path to auxiliary model checkpoint.")
    argparser.add_argument('--problem-name', type=str, default="BMR_UNCOND_256",
                           help="(optional) Name of the problem which will be used to name the result directory.")
    argparser.add_argument('--batch-size', type=int, default=32,
                           help="(optional) Batch size for sampling.")
    argparser.add_argument('--K', type=tutils.int_or_float, default=2,
                            help="(optional) Sampling contribution ratio of primary and auxiliary models. " + 
                            "Int inputs use a deterministic scheduler, while float inputs use a stochastic scheduler." + 
                            "Int K means primary model and auxiliary model will be updated K-1 times and 1 time, respectively." + 
                            "Float K means 1-(1/K) probability of updating the primary model and 1/K probability of updating the auxiliary model.")
    args = argparser.parse_args()

    save_root = Path(f"./invp_results/{args.problem_name}/K{args.K}/{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}")
    config = configs.get_config()
    config.eval.batch_size = args.batch_size


    # setup sampling environment
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    scaler = datasets.get_data_scaler(config)

    sde, sigmas = tutils.get_tpdm_sde(config)

    score_model_pri, score_model_aux = tutils.get_tpdm_models(config, args.primary_ckpt, args.auxiliary_ckpt)

    predictor = get_predictor(config.sampling.predictor)
    corrector = get_corrector(config.sampling.corrector)

    tpdm_uncond= controllable_generation.get_tpdm_uncond(sde,
                                                            predictor, corrector,
                                                            inverse_scaler,
                                                            config,
                                                            save_progress=True,
                                                            save_root=save_root,
                                                            K=args.K,
                                                            denoise=True)


    # prepair result directory
    save_root.mkdir(parents=True, exist_ok=True)
    save_root_f = save_root / 'progress'
    save_root_f.mkdir(exist_ok=True, parents=True)


    # start sampling
    recon = tpdm_uncond(score_model_pri, score_model_aux)


    # save result
    for i, recon_img in enumerate(recon):
        just_name = f"{i:03d}"
        plt.imsave(save_root / f'{just_name}.png', clear(recon_img), cmap='gray')
        np.save(str(save_root / f'{just_name}.npy'), recon_img.detach().cpu().numpy())


if __name__ == '__main__':
    main()