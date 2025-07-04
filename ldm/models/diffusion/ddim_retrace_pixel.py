"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from scripts.utils import *
import matplotlib.pyplot as plt

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr
from scripts.utils import clear_color

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.loss = torch.nn.MSELoss() # MSE loss

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        if ddim_num_steps < 1000:
          ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                    ddim_timesteps=self.ddim_timesteps,
                                                                                    eta=ddim_eta,verbose=verbose)
          self.register_buffer('ddim_sigmas', ddim_sigmas)
          self.register_buffer('ddim_alphas', ddim_alphas)
          self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
          self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
              (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                          1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        """
        Sampling wrapper function for UNCONDITIONAL sampling.
        """

        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates


    def posterior_sampler(self, measurement, measurement_cond_fn, operator_fn,
               S,
               batch_size,
               shape,
               cond_method=None,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        """
        Sampling wrapper function for inverse problem solving.
        """
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        if cond_method is None or cond_method == 'resample':
            samples = self.resample_sampling(measurement, measurement_cond_fn,
                                                    conditioning, size,
                                                        operator_fn=operator_fn,
                                                        callback=callback,
                                                        img_callback=img_callback,
                                                        quantize_denoised=quantize_x0,
                                                        mask=mask, x0=x0,
                                                        ddim_use_original_steps=False,
                                                        noise_dropout=noise_dropout,
                                                        temperature=temperature,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs,
                                                        x_T=x_T,
                                                        log_every_t=log_every_t,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,
                                                        )
            
        else:
            raise ValueError(f"Condition method string '{cond_method}' not recognized.")
        
        return samples, None


    def resample_sampling(self, measurement, measurement_cond_fn, cond, shape, operator_fn=None,
                     inter_timesteps=10, x_T=None, ddim_use_original_steps=False,
                     callback=None, timesteps=None, quantize_denoised=False,
                     mask=None, x0=None, img_callback=None, log_every_t=100,
                     temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                     unconditional_guidance_scale=1., unconditional_conditioning=None,):
        """
        DDIM-based sampling function for ReSample.

        Arguments:
            measurement:            Measurement vector y in y=Ax+n.
            measurement_cond_fn:    Function to perform DPS. 
            operator_fn:            Operator to perform forward operation A(.)
            inter_timesteps:        Number of timesteps to perform time travelling.

        """

        device = self.model.betas.device
        b = shape[0]


        """ if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end] """

        #intermediates = {'x_inter': [img], 'pred_x0': [img]}

        num_timesteps = 3
        timesteps = np.array([i*(500//num_timesteps)*2 +1 for i in range(1, num_timesteps)] + [999])
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        # Need for measurement consistency
        alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas 
        alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev

        img = torch.randn(shape, requires_grad=True, device=device)

        outer_steps = 50

        outer_iterator = tqdm(np.arange(outer_steps), desc='Seed Optimization', disable=False)

        for i_pass in outer_iterator:

            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True)

            img_trajectory = [img.detach().clone()]

            # plot histogram of elements of img
            plt.hist(img.detach().cpu().numpy().flatten(), bins=100)
            plt.savefig(f'{i_pass}_histogram.png')
            plt.close()

            for i, step in enumerate(iterator):    
                reconstructed = clear_color(img.detach())
                plt.imsave(f'{i_pass}_{i}_x0.png', reconstructed) 
            
                
                # Instantiating parameters
                index = (step+1)//2 - 1
                if i+1 < len(time_range):
                    prev_index = (time_range[i+1]+1)//2 - 1
                else:
                    prev_index = -1

                ts = torch.full((b,), step, device=device, dtype=torch.long)
                self.a_t = torch.full((b, 1, 1, 1), alphas[index], device=device, requires_grad=False) # Needed for ReSampling
                if prev_index >= 0:
                    self.a_prev = torch.full((b, 1, 1, 1), alphas[prev_index], device=device, requires_grad=False)
                else:
                    self.a_prev = torch.full((b, 1, 1, 1), alphas_prev[0], device=device, requires_grad=False)

                if mask is not None:
                    assert x0 is not None
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                    img = img_orig * mask + (1. - mask) * img

                # Unconditional DDIM sampling step
                # pred_x0 is computing \hat{x}_0 using Tweedie's formula
                out, pred_x0 = self.p_sample_ddim(img, cond, ts, 
                                        quantize_denoised=quantize_denoised, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning)
                
                original_out = out.detach().clone()
            
                # Latent DPS
                img, _ = measurement_cond_fn(x_t=out, # x_t is x_{t-1}
                                                measurement=measurement,
                                                noisy_measurement=measurement,
                                                x_prev=img, # x_prev is x_t
                                                x_0_hat=pred_x0,
                                                scale = 0.05 # * self.a_t, # For DPS learning rate / scale
                                                )
                # norm of img
                print(f"{i_pass}_{i}", (img-original_out).norm().item())
                
                img_trajectory.append(img.detach().clone())

        
            reconstructed = clear_color(img.detach())
            plt.imsave(f'{i_pass}_{i+1}_x0.png', reconstructed)

            img_trajectory[-1] = measurement

            for i, step in enumerate(reversed(time_range)):        
                # Instantiating parameters
                index = (step+1)//2 - 1
                if i+1 < len(time_range):
                    prev_index = (time_range[i+1]+1)//2 - 1
                else:
                    prev_index = -1

                ts = torch.full((b,), step, device=device, dtype=torch.long)
                self.a_t = torch.full((b, 1, 1, 1), alphas[index], device=device, requires_grad=False) # Needed for ReSampling
                if prev_index >= 0:
                    self.a_prev = torch.full((b, 1, 1, 1), alphas[prev_index], device=device, requires_grad=False)
                else:
                    self.a_prev = torch.full((b, 1, 1, 1), alphas_prev[0], device=device, requires_grad=False)

                prev_img = img_trajectory[-i-1]
                noisy_img = img_trajectory[-i-2]
                
                noisy_img = self.forward_optimization(noisy_img, prev_img, cond, ts, index, quantize_denoised, 
                                                      score_corrector, corrector_kwargs,
                                                      unconditional_guidance_scale, unconditional_conditioning)
                
                
                img_trajectory[-i-2] = noisy_img

            img = img_trajectory[0].detach().clone().requires_grad_()
                

            """ # Callback functions if needed
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0) """       
                

        img = pred_x0.detach().clone()

        return img #intermediates
    
    def forward_optimization(self, noisy_img, prev_img, cond, ts, index, quantize_denoised, score_corrector,
                             corrector_kwargs, unconditional_guidance_scale, unconditional_conditioning, lambda1=2.0, lambda2=100.0, lambda3=400, eps=1e-3, max_iters=50):

        opt_var = noisy_img.detach().clone().requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=1e-1)
        
        for _ in range(max_iters):
            optimizer.zero_grad()
            
            # Compute the loss
            img_ddim, _ = self.p_sample_ddim(opt_var, cond, ts,
                                        quantize_denoised=quantize_denoised, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning)
            
            consistency_loss = self.loss(img_ddim, prev_img)
            mean = torch.mean(opt_var)
            std = torch.std(opt_var, unbiased=False)
            regularization_term = 0 #lambda1 * (mean**2 + (std - 1)**2)
            #print(f"{index} Consistency loss: {consistency_loss.item()}, Regularization term: {regularization_term.item()}")
            total_loss = consistency_loss #+ regularization_term + l2_term
            total_loss.backward()
            #torch.nn.utils.clip_grad_norm_([opt_var], max_norm=0.1)
            optimizer.step()
        

        return opt_var.detach().clone()


    def p_sample_ddim(self, x, c, t, quantize_denoised=False, score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)


        # current prediction for x_0
        pred_x0 = (x - (1-self.a_t).sqrt() * e_t) / self.a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - self.a_prev).sqrt() * e_t
        #noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        #if noise_dropout > 0.:
        #    noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = self.a_prev.sqrt() * pred_x0 + dir_xt

        return x_prev, pred_x0