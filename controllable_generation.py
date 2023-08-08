import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

import functools

from sampling import shared_corrector_update_fn, shared_predictor_update_fn
from utils import clear, fft2


def get_tpdm_zsr(sde, predictor, corrector, inverse_scaler, config, dps_weight,
                          factor, save_root, save_progress, denoise=True, eps=1e-5, z_mask_idxs=None):

    def get_M(factor: int):
        if factor == 2:
            return torch.tensor(([[0.70710678118, 0.70710678118],
                                  [0.70710678118, -0.70710678118]]))
        elif factor == 3:
            return torch.tensor([[5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                                 [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                                 [5.7735026e-01, 4.0824822e-01, -7.0710683e-01]])
        elif factor == 4:
            return torch.tensor([[0.5,  0.866025403784439,  0,                 0],
                                 [0.5, -0.288675134594813,  0.816496580927726, 0],
                                 [0.5, -0.288675134594813, -0.408248290463863, 0.707106781186548],
                                 [0.5, -0.288675134594813, -0.408248290463863, -0.707106781186548]])
        elif factor == 5:
            return torch.tensor([[0.447213595499958,  0.894427190999916,  0,                 0,                  0],
                                 [0.447213595499958, -0.223606797749979,  0.866025403784439, 0,                  0],
                                 [0.447213595499958, -0.223606797749979, -0.288675134594813,  0.816496580927726, 0],
                                 [0.447213595499958, -0.223606797749979, -0.288675134594813, -0.408248290463863, 0.707106781186548],
                                 [0.447213595499958, -0.223606797749979, -0.288675134594813, -0.408248290463863, -0.707106781186548]])
        else:
            raise ValueError(f"unsupported zsr-factor ({factor}) for kernel")

    M = get_M(factor)
    invM = torch.inverse(M)


    # Decouple a down-sampled image with `M`
    def decouple(inputs):
        B, C, H, W = inputs.shape
        w_pad_size = 256 % factor

        inputs_rs = inputs[..., 0:256 - w_pad_size].reshape(B, C, H, W // factor, factor)
        inputs_decp_rs = torch.einsum('bchwi,ij->bchwj', inputs_rs, M.to(inputs.device))
        inputs_decp = inputs_decp_rs.reshape(B, C, H, 256 - w_pad_size)
        inputs_decp =  F.pad(inputs_decp, (0, w_pad_size, 0, 0, 0, 0, 0, 0))

        return inputs_decp

    # The inverse function to `decouple`.
    def couple(inputs):
        B, C, H, W = inputs.shape
        w_pad_size = 256 % factor

        inputs_rs = inputs[..., 0:256 - w_pad_size].reshape(B, C, H, W // factor, factor)
        inputs_cp_rs = torch.einsum('bchwi,ij->bchwj', inputs_rs, invM.to(inputs.device))
        inputs_cp = inputs_cp_rs.reshape(B, C, H, 256 - w_pad_size)
        inputs_cp =  F.pad(inputs_cp, (0, w_pad_size, 0, 0, 0, 0, 0, 0))

        return inputs_cp
    
    def get_mask(image, channel):
        B, C, H, W = image.shape
        mask = torch.zeros((B, C, H, 256), device=image.device)
        mask[:, :, :, ::channel] = 1

        if z_mask_idxs is not None:
            mask[:, :, :, z_mask_idxs] = 0

        return mask


    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=config.sampling.probability_flow,
                                            continuous=config.training.continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=config.training.continuous,
                                            snr=config.sampling.snr,
                                            n_steps=config.sampling.n_steps_each)


    def get_tpdm_zsr_update_fn(update_fn):
        def tpdm_zsr_update_fn(model, measure_dagger, x, t):
            mask = get_mask(x, factor)
            vec_t = torch.ones(x.shape[0], device=x.device) * t

            # input to the score function
            x = x.requires_grad_()
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            masked_data_mean, std = sde.marginal_prob(decouple(measure_dagger), vec_t)
            masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]

            # x0 hat prediction
            _, bt = sde.marginal_prob(x, vec_t)
            hatx0 = x + (bt[..., None, None, None] ** 2) * score

            # DPS step for the data consistency
            norm = torch.norm(couple(decouple(hatx0) * mask - masked_data * mask))
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

            x_next = x_next - norm_grad * dps_weight
            x_next_mean = x_next_mean - norm_grad * dps_weight

            x_next = x_next.detach()
            x_next_mean = x_next_mean.detach()

            return x_next, x_next_mean

        return tpdm_zsr_update_fn

    def get_tpdm_uncond_update_fn(update_fn):
        def tpdm_uncond_update_fn(model, x, t):
            vec_t = torch.ones(x.shape[0], device=x.device) * t

            with torch.no_grad():
                x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            return x_next, x_next_mean

        return tpdm_uncond_update_fn

    predictor_tpdm_zsr_update_fn = get_tpdm_zsr_update_fn(predictor_update_fn)
    corrector_tpdm_zsr_update_fn = get_tpdm_zsr_update_fn(corrector_update_fn)
    predictor_tpdm_uncond_update_fn = get_tpdm_uncond_update_fn(predictor_update_fn)
    corrector_tpdm_uncond_update_fn = get_tpdm_uncond_update_fn(corrector_update_fn)


    def tpdm_zsr(model_pri, model_aux, measure_dagger):
        batch_len = len(measure_dagger) // config.eval.batch_size if len(measure_dagger) % config.eval.batch_size == 0 else len(measure_dagger) // config.eval.batch_size + 1
        B, C, H, W = measure_dagger.shape
        shape = (B, C, H, 256)
        mask = get_mask(measure_dagger, factor)

        # Initial sample
        x = couple(decouple(measure_dagger) * mask + \
                   decouple(sde.prior_sampling(shape).to(measure_dagger.device)
                            * (1. - mask)))
        timesteps = torch.linspace(sde.T, eps, sde.N)
        dataloader_md = torch.tensor_split(measure_dagger, batch_len)
        
        for i in tqdm(range(sde.N), colour="blue", unit="step", smoothing=0):
            if i % 2 == 1:
                # [X, 1, Y, Z] -> [Z, 1, X, Y]
                x = x.permute(3, 1, 0, 2)

            dataloader = torch.tensor_split(x, batch_len)
            
            x_batch_s = []
            x_mean_batch_s = []

            for x_batch, gray_scale_img_batch in tqdm(zip(dataloader, dataloader_md), total=batch_len, colour="blue", unit="mb", leave=False):
                if i % 2 == 0: # reverse diffusion with primary model + DPS
                    t = timesteps[i]
                    x_batch, x_mean_batch = corrector_tpdm_zsr_update_fn(model_pri, gray_scale_img_batch, x_batch, t)
                    x_batch, x_mean_batch = predictor_tpdm_zsr_update_fn(model_pri, gray_scale_img_batch, x_batch, t)

                else: # reverse diffusion with auxiliary model
                    t = timesteps[i]
                    x_batch, x_mean_batch = corrector_tpdm_uncond_update_fn(model_aux, x_batch, t)
                    x_batch, x_mean_batch = predictor_tpdm_uncond_update_fn(model_aux, x_batch, t)

                x_batch_s.append(x_mean_batch)
                x_mean_batch_s.append(x_mean_batch)

            x = torch.cat(x_batch_s, dim=0)
            x_mean = torch.cat(x_mean_batch_s, dim=0)

            if i % 2 == 1:
                # [Z, 1, X, Y] -> [X, 1, Y, Z]
                x = x.permute(2, 1, 3, 0)
                x_mean = x_mean.permute(2, 1, 3, 0)

            if save_progress:
                if (i % 50) == 0:
                    plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x_mean[len(x_mean)//2:len(x_mean)//2+1]), cmap='gray')
        
        return inverse_scaler(x_mean if denoise else x)

    return tpdm_zsr


def get_tpdm_cs_mri(sde, predictor, corrector, inverse_scaler, config,
                    save_root, save_progress, dps_weight, denoise=True, eps=1e-5):

    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=config.sampling.probability_flow,
                                            continuous=config.training.continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=config.training.continuous,
                                            snr=config.sampling.snr,
                                            n_steps=config.sampling.n_steps_each)

    def get_tpdm_cs_mri_update_fn(update_fn):
        def tpdm_cs_mri_update_fn(model, measure_kspace, mask, x, t):
            vec_t = torch.ones(x.shape[0], device=x.device) * t

            # input to the score function
            x = x.requires_grad_()
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            # x0 hat prediction
            _, bt = sde.marginal_prob(x, vec_t)
            hatx0 = x + (bt[..., None, None, None] ** 2) * score

            # DPS step for the data consistency
            norm = torch.norm(fft2(hatx0) * mask - measure_kspace)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

            x_next = x_next - norm_grad * dps_weight
            x_next_mean = x_next_mean - norm_grad * dps_weight

            x_next = x_next.detach()
            x_next_mean = x_next_mean.detach()

            return x_next, x_next_mean

        return tpdm_cs_mri_update_fn

    def get_tpdm_uncond_update_fn(update_fn):
        def tpdm_uncod_update_fn(model, x, t):
            vec_t = torch.ones(x.shape[0], device=x.device) * t

            with torch.no_grad():
                x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            return x_next, x_next_mean

        return tpdm_uncod_update_fn

    predictor_tpdm_cs_mri_update_fn = get_tpdm_cs_mri_update_fn(predictor_update_fn)
    corrector_tpdm_cs_mri_update_fn = get_tpdm_cs_mri_update_fn(corrector_update_fn)
    predictor_tpdm_uncond_update_fn = get_tpdm_uncond_update_fn(predictor_update_fn)
    corrector_tpdm_uncond_update_fn = get_tpdm_uncond_update_fn(corrector_update_fn)

    def tpdm_cs_mri(model_pri, model_aux, measure_kspace, mask):
        batch_len = len(measure_kspace) // config.eval.batch_size if len(measure_kspace) % config.eval.batch_size == 0 else len(measure_kspace) // config.eval.batch_size + 1
        B, C, H, W = measure_kspace.shape
        shape = (B, C, H, 256)

        # Initial sample
        x = sde.prior_sampling(shape).to(measure_kspace.device)
        timesteps = torch.linspace(sde.T, eps, sde.N)
        dataloader_measure_kspace = torch.tensor_split(measure_kspace, batch_len)
        dataloader_mask = torch.tensor_split(mask, batch_len)
        
        for i in tqdm(range(sde.N), colour="blue", unit="step", smoothing=0):
            if i % 2 == 1:
                # [Z, 1, X, Y] -> [X, 1, Y, Z]
                x = x.permute(2, 1, 3, 0)

            dataloader = torch.tensor_split(x, batch_len)
            
            x_batch_s = []
            x_mean_batch_s = []

            for x_batch, measure_kspace_batch, mask_batch in tqdm(zip(dataloader, dataloader_measure_kspace, dataloader_mask), total=batch_len, colour="blue", unit="mb", leave=False):
                if i % 2 == 0: # reverse diffusion with primary model + DPS
                    t = timesteps[i]
                    x_batch, x_mean_batch = corrector_tpdm_cs_mri_update_fn(model_pri, measure_kspace_batch, mask_batch, x_batch, t)
                    x_batch, x_mean_batch = predictor_tpdm_cs_mri_update_fn(model_pri, measure_kspace_batch, mask_batch, x_batch, t)

                else: # reverse diffusion with auxiliary model
                    t = timesteps[i]
                    x_batch, x_mean_batch = corrector_tpdm_uncond_update_fn(model_aux, x_batch, t)
                    x_batch, x_mean_batch = predictor_tpdm_uncond_update_fn(model_aux, x_batch, t)

                x_batch_s.append(x_mean_batch)
                x_mean_batch_s.append(x_mean_batch)

            x = torch.cat(x_batch_s, dim=0)
            x_mean = torch.cat(x_mean_batch_s, dim=0)

            if i % 2 == 1:
                # [X, 1, Y, Z] -> [Z, 1, X, Y]
                x = x.permute(3, 1, 0, 2)
                x_mean = x_mean.permute(3, 1, 0, 2)

            if save_progress:
                if (i % 50) == 0:
                    plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x_mean[len(x_mean)//2:len(x_mean)//2+1]), cmap='gray')
        
        return inverse_scaler(x_mean if denoise else x)

    return tpdm_cs_mri


def get_tpdm_sv_ct(sde, predictor, corrector, inverse_scaler, config, radon,
                   save_root, save_progress, dps_weight, denoise=True, eps=1e-5):

    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=config.sampling.probability_flow,
                                            continuous=config.training.continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=config.training.continuous,
                                            snr=config.sampling.snr,
                                            n_steps=config.sampling.n_steps_each)

    def get_tpdm_sv_ct_update_fn(update_fn):
        def sv_ct_update_fn(model, measure_sino, x, t):
            vec_t = torch.ones(x.shape[0], device=x.device) * t

            # input to the score function
            x = x.requires_grad_()
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            # x0 hat prediction
            _, bt = sde.marginal_prob(x, vec_t)
            hatx0 = x + (bt[..., None, None, None] ** 2) * score

            # DPS step for the data consistency
            norm = torch.norm(radon.A(hatx0) - measure_sino)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

            x_next = x_next - norm_grad * dps_weight
            x_next_mean = x_next_mean - norm_grad * dps_weight

            x_next = x_next.detach()
            x_next_mean = x_next_mean.detach()

            return x_next, x_next_mean

        return sv_ct_update_fn

    def get_tpdm_uncond_update_fn(update_fn):
        def tpdm_uncod_update_fn(model, x, t):
            vec_t = torch.ones(x.shape[0], device=x.device) * t

            with torch.no_grad():
                x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            return x_next, x_next_mean

        return tpdm_uncod_update_fn

    predictor_tpdm_sv_ct_update_fn = get_tpdm_sv_ct_update_fn(predictor_update_fn)
    corrector_tpdm_sv_ct_update_fn = get_tpdm_sv_ct_update_fn(corrector_update_fn)
    predictor_tpdm_uncond_update_fn = get_tpdm_uncond_update_fn(predictor_update_fn)
    corrector_tpdm_uncond_update_fn = get_tpdm_uncond_update_fn(corrector_update_fn)


    def tpdm_sv_ct(model_pri, model_aux, measure_sino):
        batch_len = len(measure_sino) // config.eval.batch_size if len(measure_sino) % config.eval.batch_size == 0 else len(measure_sino) // config.eval.batch_size + 1
        B, C, _, _ = measure_sino.shape
        shape = (B, C, 256, 256)

        # Initial sample
        x = sde.prior_sampling(shape).to(measure_sino.device)
        timesteps = torch.linspace(sde.T, eps, sde.N)
        dataloader_measure_sino = torch.tensor_split(measure_sino, batch_len)
       
        for i in tqdm(range(sde.N), colour="blue", unit="step", smoothing=0):
            if i % 2 == 1:
                # [Z, 1, X, Y] -> [X, 1, Y, Z]
                x = x.permute(2, 1, 3, 0)

            dataloader = torch.tensor_split(x, batch_len)
            
            x_batch_s = []
            x_mean_batch_s = []

            for x_batch, measure_sino_batch in tqdm(zip(dataloader, dataloader_measure_sino), total=batch_len, colour="blue", unit="mb", leave=False):
                if i % 2 == 0: # reverse diffusion with primary model + DPS
                    t = timesteps[i]
                    x_batch, x_mean_batch = corrector_tpdm_sv_ct_update_fn(model_pri, measure_sino_batch, x_batch, t)
                    x_batch, x_mean_batch = predictor_tpdm_sv_ct_update_fn(model_pri, measure_sino_batch, x_batch, t)

                else: # reverse diffusion with auxiliary model
                    t = timesteps[i]
                    x_batch, x_mean_batch = corrector_tpdm_uncond_update_fn(model_aux, x_batch, t)
                    x_batch, x_mean_batch = predictor_tpdm_uncond_update_fn(model_aux, x_batch, t)

                x_batch_s.append(x_mean_batch)
                x_mean_batch_s.append(x_mean_batch)

            x = torch.cat(x_batch_s, dim=0)
            x_mean = torch.cat(x_mean_batch_s, dim=0)

            if i % 2 == 1:
                # [X, 1, Y, Z] -> [Z, 1, X, Y]
                x = x.permute(3, 1, 0, 2)
                x_mean = x_mean.permute(3, 1, 0, 2)

            if save_progress:
                if (i % 50) == 0:
                    plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x_mean[len(x_mean)//2:len(x_mean)//2+1]), cmap='gray')
        
        return inverse_scaler(x_mean if denoise else x)

    return tpdm_sv_ct


def get_tpdm_uncond(sde, predictor, corrector, inverse_scaler, config, 
                    save_root, save_progress, denoise=True, eps=1e-5):
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=config.sampling.probability_flow,
                                            continuous=config.training.continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=config.training.continuous,
                                            snr=config.sampling.snr,
                                            n_steps=config.sampling.n_steps_each)

    def get_tpdm_uncond_update_fn(update_fn):
        def tpdm_uncond_update_fn(model, x, t):
            vec_t = torch.ones(x.shape[0], device=x.device) * t

            with torch.no_grad():
                x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            return x_next, x_next_mean

        return tpdm_uncond_update_fn

    predictor_tpdm_uncond_update_fn = get_tpdm_uncond_update_fn(predictor_update_fn)
    corrector_tpdm_uncond_update_fn = get_tpdm_uncond_update_fn(corrector_update_fn)

    def tpdm_uncond(model, model_aux):
        B = 256
        shape = (B, 1, 256, 256)
        batch_len = B // config.eval.batch_size if B % config.eval.batch_size == 0 else B // config.eval.batch_size + 1

        # Initial sample
        x = sde.prior_sampling(shape).to(config.device)
        timesteps = torch.linspace(sde.T, eps, sde.N)
        
        for i in tqdm(range(sde.N), colour="blue", unit="step", smoothing=0):
            if i % 2 == 1:
                # [X, 1, Y, Z] -> [Z, 1, X, Y]
                x = x.permute(3, 1, 0, 2)

            dataloader = torch.tensor_split(x, batch_len)
            
            x_batch_s = []
            x_mean_batch_s = []

            for x_batch in tqdm(dataloader, total=batch_len, colour="blue", unit="mb", leave=False):
                if i % 2 == 0: # reverse diffusion with primary model
                    t = timesteps[i]
                    x_batch, x_mean_batch = corrector_tpdm_uncond_update_fn(model, x_batch, t)
                    x_batch, x_mean_batch = predictor_tpdm_uncond_update_fn(model, x_batch, t)

                else: # reverse diffusion with auxiliary model
                    t = timesteps[i]
                    x_batch, x_mean_batch = corrector_tpdm_uncond_update_fn(model_aux, x_batch, t)
                    x_batch, x_mean_batch = predictor_tpdm_uncond_update_fn(model_aux, x_batch, t)

                x_batch_s.append(x_mean_batch)
                x_mean_batch_s.append(x_mean_batch)

            x = torch.cat(x_batch_s, dim=0)
            x_mean = torch.cat(x_mean_batch_s, dim=0)

            if i % 2 == 1:
                # [Z, 1, X, Y] -> [X, 1, Y, Z]
                x = x.permute(2, 1, 3, 0)
                x_mean = x_mean.permute(2, 1, 3, 0)

            if save_progress:
                if (i % 50) == 0:
                    plt.imsave(save_root / 'progress' / f'progress{i}.png', clear(x_mean[len(x_mean)//2:len(x_mean)//2+1]), cmap='gray')
            
        return inverse_scaler(x_mean if denoise else x)

    return tpdm_uncond