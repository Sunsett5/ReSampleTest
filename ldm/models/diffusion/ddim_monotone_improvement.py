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
        self.model = model.to()
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.loss = torch.nn.MSELoss()

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
            samples, intermediates = self.resample_sampling(measurement, measurement_cond_fn,
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
        
        return samples, intermediates


    def resample_sampling(self, measurement, measurement_cond_fn, cond, shape, operator_fn=None,
                     inter_timesteps=10, x_T=None, ddim_use_original_steps=False,
                     callback=None, timesteps=None, quantize_denoised=False,
                     mask=None, x0=None, img_callback=None, log_every_t=100,
                     temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                     unconditional_guidance_scale=1., unconditional_conditioning=None, block=500, rewind=0):
        """
        DDIM-based sampling function for ReSample.

        Arguments:
            measurement:            Measurement vector y in y=Ax+n.
            measurement_cond_fn:    Function to perform DPS. 
            operator_fn:            Operator to perform forward operation A(.)
            inter_timesteps:        Number of timesteps to perform time travelling.

        """

        self.opt_change_history = [None for _ in range(4)]

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        
        img = img.requires_grad_() # Require grad for data consistency

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        # Need for measurement consistency
        self.alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas.to(device)
        self.sqrt_one_minus_alphas  = self.model.sqrt_one_minus_alphas_cumprod if ddim_use_original_steps else self.ddim_sqrt_one_minus_alphas.to(device)
        self.alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
        self.betas = self.model.betas
        self.sigmas = self.ddim_sigmas_for_original_num_steps if ddim_use_original_steps else self.ddim_sigmas.to(device)

        rewind_time_range = self.staggered_indices(time_range, block_size=block, step_back=rewind)

        iterator = tqdm(rewind_time_range, desc='DDIM Sampler', disable=False)

        for i, step in enumerate(iterator):   
            # Instantiating parameters
            index = (step.item()+1)//2 - 1
            if i+1 < len(rewind_time_range):
                prev_index = (rewind_time_range[i+1].item()+1)//2 - 1
            else:
                prev_index = -1    

            ts = torch.full((b,), step, device=device, dtype=torch.long) 
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            # Unconditional sampling step
            # pred_x0 is from DDIM

            if index % 20 == 0 and index < 200:
                img_decoded = self.model.decode_first_stage(img.detach()) # Decode the latent space into pixel space
                reconstructed = clear_color(img_decoded)
                plt.imsave(f'{index}_before_opt.png', reconstructed)
                img = self.monotone_optimization(img, cond, measurement, ts, index, prev_index, ddim_use_original_steps, quantize_denoised, temperature, noise_dropout, 
                            score_corrector, corrector_kwargs, unconditional_guidance_scale, unconditional_conditioning, n_shots=4, operator_fn=operator_fn, eps=1e-3, max_iters=20)
                img_decoded = self.model.decode_first_stage(img.detach()) # Decode the latent space into pixel space
                reconstructed = clear_color(img_decoded)
                plt.imsave(f'{index}_after_opt.png', reconstructed)
        
            out, pred_x0, _ = self.p_sample_ddim(img, cond, ts, index=index, prev_index=prev_index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            
            a_t = torch.full((b, 1, 1, 1), self.alphas[index], device=device, requires_grad=False)
            if prev_index >= 0:
                a_prev = torch.full((b, 1, 1, 1), self.alphas[prev_index], device=device, requires_grad=False)
            else:
                a_prev = torch.full((b, 1, 1, 1), self.alphas_prev[0], device=device, requires_grad=False)

            #img = out.detach().clone() # Detaching to avoid gradient flow from here

            img, _ = measurement_cond_fn(x_t=out, # x_t is x_{t-1}
                                            measurement=measurement,
                                            noisy_measurement=measurement,
                                            x_prev=img, # x_prev is x_t
                                            x_0_hat=pred_x0,
                                            scale=a_t*.5, # For DPS learning rate / scale
                                            )
            
            # Instantiating time-travel parameters
            splits = 3 # TODO: make this not hard-coded
            index_split = total_steps // splits

            # Performing time-travel if in selected indices
            if index <= (total_steps - index_split) and index > 0:   

                if self.opt_change_history[-1] is not None:
                    with torch.no_grad():
                        
                        img = img.detach() + 0.3 * self.opt_change_history[-1]
                        #pred_x0 = pred_x0.detach() + 0.35 * self.opt_change_history[-1]

                    img = img.requires_grad_() # Requiring grad again after adding change

                x_t = img.detach().clone()

                # Performing only every 5 steps (or so)
                # TODO: also make this not hard-coded
                #if index < prev_index:
                """
                if index % 5 == 0:
                         
                    # Some arbitrary scheduling for sigma

                    # Pixel-based optimization for second stage
                    if index >= index_split: 
                        
                        # Enforcing consistency via pixel-based optimization
                        pred_x0 = pred_x0.detach() 
                        pred_x0_pixel = self.model.decode_first_stage(pred_x0) # Get \hat{x}_0 into pixel space

                        opt_var = self.pixel_optimization(measurement=measurement, 
                                                          x_prime=pred_x0_pixel,
                                                          operator_fn=operator_fn)
                        
                        opt_var = self.model.encode_first_stage(opt_var) # Going back into latent space

                        if index >= 0:
                            sigma = 40*(1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)  
                        else:
                            sigma = 0.5

                        img = self.stochastic_resample(pred_x0=opt_var, x_t=x_t, a_t=a_prev, sigma=sigma)
                        img = img.requires_grad_() # Seems to need to require grad here

                    # Latent-based optimization for third stage
                    elif index < index_split: # Needs to (possibly) be tuned

                        # Enforcing consistency via latent space optimization
                        pred_x0, _ = self.latent_optimization(measurement=measurement,
                                                             z_init=pred_x0.detach(),
                                                             operator_fn=operator_fn)


                        sigma = 40 * (1-a_prev)/(1 - a_t) * (1 - a_t / a_prev) # Change the 40 value for each task

                        img = self.stochastic_resample(pred_x0=pred_x0, x_t=x_t, a_t=a_prev, sigma=sigma) 
                """

            # Callback functions if needed
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)       
                
        psuedo_x0, _ = self.latent_optimization(measurement=measurement,
                                                             z_init=img.detach(),
                                                             operator_fn=operator_fn)
        img = psuedo_x0.detach().clone()
            
        return img, intermediates
    
    def monotone_optimization(self, x_t, cond, measurement, t, original_index, original_prev_index, use_original_steps, quantize_denoised, temperature, noise_dropout, score_corrector, 
                            corrector_kwargs, unconditional_guidance_scale, unconditional_conditioning, n_shots=4, operator_fn=None, eps=1e-3, max_iters=50):
        
        opt_var = x_t.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=1e-1) # Initializing optimizer
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop

        for i in range(max_iters):
            optimizer.zero_grad()

            out, pred_x0, measurement_losses = self.precise_tweedie(opt_var, cond, measurement, operator_fn, t, original_index, original_prev_index, use_original_steps, quantize_denoised, temperature, noise_dropout, 
                            score_corrector, corrector_kwargs, unconditional_guidance_scale, unconditional_conditioning, n_shots)
            
            average_loss = torch.dot(measurement_losses, torch.ones_like(measurement_losses)) / measurement_losses.shape[0] # Average loss
            if len(measurement_losses) > 1:
                loss_ratio = measurement_losses[1:] / measurement_losses[:-1]
            else:
                loss_ratio = torch.tensor([1.0], device=measurement_losses.device)

            print(measurement_losses)

            print("Ratio", torch.mean(loss_ratio))

            adjusted_loss = average_loss * torch.exp(3*torch.mean(loss_ratio)-1) # Adjusting loss based on the ratio of losses
            
            adjusted_loss.backward() # Take GD step
            optimizer.step()

            #print(f"Loss at step {i}: {loss.item()}")

            # Convergence criteria
            if adjusted_loss < eps**2: # needs tuning according to noise level for early stopping
                break
        
        return opt_var
                              
        

    def staggered_indices(self, original, block_size=10, step_back=5):
        idx = 0
        result = []
        length = len(original)
        while True:
            end = min(idx + block_size, length)
            result.extend(original[idx:end])
            if end - idx == step_back:
                break
            idx = end - step_back
        return np.array(result)


    def pixel_optimization(self, measurement, x_prime, operator_fn, eps=1e-3, max_iters=200):
        """
        Function to compute argmin_x ||y - A(x)||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            x_prime:               Estimation of \hat{x}_0 using Tweedie's formula
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        """

        loss = torch.nn.MSELoss() # MSE loss

        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=1e-2) # Initializing optimizer
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop

        for _ in range(max_iters):
            optimizer.zero_grad()
            
            measurement_loss = loss(measurement, operator_fn( opt_var ) ) 
            
            measurement_loss.backward() # Take GD step
            optimizer.step()

            # Convergence criteria
            if measurement_loss < eps**2: # needs tuning according to noise level for early stopping
                break

        return opt_var


    def latent_optimization(self, measurement, z_init, operator_fn, eps=1e-3, max_iters=50, lr=None):

        """
        Function to compute argmin_z ||y - A( D(z) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        
        Optimal parameters seem to be at around 500 steps, 200 steps for inpainting.

        """

        # Base case
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()

        if lr is None:
            lr_val = 5e-3
        else:
            lr_val = lr.item()

        loss = torch.nn.MSELoss() # MSE loss
        optimizer = torch.optim.AdamW([z_init], lr=lr_val) # Initializing optimizer ###change the learning rate
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop
        init_loss = 0
        losses = []

        z_init_orig = z_init.detach().clone()
        
        for itr in range(max_iters):
            optimizer.zero_grad()
            output = loss(measurement, operator_fn( self.model.differentiable_decode_first_stage( z_init ) ))          

            if itr == 0:
                init_loss = output.detach().clone()
                
            output.backward() # Take GD step
            optimizer.step()
            cur_loss = output.detach().cpu().numpy() 
            
            # Convergence criteria

            if itr < 200: # may need tuning for early stopping
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    break
                else:
                    losses.pop(0)
                    
            if cur_loss < eps**2:  # needs tuning according to noise level for early stopping
                break

        opt_change = z_init.detach() - z_init_orig

        """ if self.x_0_history[-1] is not None:
            current_delta_x0 = z_init - self.x_0_history[-1]
        for lag in range(len(self.x_0_history) - 1, 0, -1):
            if self.x_0_history[lag-1] is not None:
                prev_delta_x0 = self.x_0_history[lag] -  self.x_0_history[lag-1]
                cosine_similarity = torch.nn.functional.cosine_similarity(current_delta_x0.view(-1), prev_delta_x0.view(-1), dim=0)
                print("LAG", len(self.x_0_history)-lag, "Cosine", cosine_similarity.item())
            if self.opt_change_history[lag-1] is not None:
                opt_cosine_similarity = torch.nn.functional.cosine_similarity(opt_change.view(-1), self.opt_change_history[lag-1].view(-1), dim=0)
                print("LAG", len(self.opt_change_history)-lag, "Opt Cosine", opt_cosine_similarity.item()) """

        self.opt_change_history.append(opt_change.detach().clone())
        #self.opt_change_plot.append(torch.linalg.norm(opt_change.detach()).item())
        del self.opt_change_history[0]

        #self.x_0_history.append(z_init.detach().clone())
        #l self.x_0_history[0]

        return z_init, init_loss       


    def stochastic_resample(self, pred_x0, x_t, a_t, sigma):
        """
        Function to resample x_t based on ReSample paper.
        """
        device = self.model.betas.device
        noise = torch.randn_like(pred_x0, device=device)
        return (sigma * a_t.sqrt() * pred_x0 + (1 - a_t) * x_t)/(sigma + 1 - a_t) + noise * torch.sqrt(1/(1/sigma + 1/(1-a_t)))


    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        """
        Function for unconditional sampling using DDIM.
        """

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=False)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates
    
    def precise_tweedie(self, x, c, measurement, operator_fn, t, original_index, original_prev_index, use_original_steps, quantize_denoised, temperature, noise_dropout, 
                        score_corrector, corrector_kwargs, unconditional_guidance_scale, unconditional_conditioning, repeat_noise=False, n_shots=4):
        b, *_, device = *x.shape, x.device
        step_size = t//n_shots
        time_steps = [((t - i*step_size)//2)*2 +1 for i in range(0, n_shots)] + [torch.tensor(-1)]
        time_steps = [time_steps[i] for i in range(len(time_steps)) if i==len(time_steps)-1 or (time_steps[i+1] != time_steps[i])]

        x_inter = x.clone()

        losses = torch.zeros((len(time_steps)-1,)).to(device)

        #print([step.item() for step in time_steps])

        for i, step in enumerate(time_steps[:-1]):
            index = (step.item()+1)//2 - 1
            prev_index = (time_steps[i+1].item()+1)//2 - 1
            x_inter, pred_x0, _ = self.p_sample_ddim(x_inter, c, step, index=index, prev_index=prev_index, use_original_steps=use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            losses[i] = self.loss(measurement, operator_fn(self.model.differentiable_decode_first_stage(x_inter)))
            
        a_t = torch.full((b, 1, 1, 1), self.alphas[original_index], device=device, requires_grad=False)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.sqrt_one_minus_alphas[original_index], device=device, requires_grad=False)
        if original_prev_index >= 0:
            a_prev = torch.full((b, 1, 1, 1), self.alphas[original_prev_index], device=device, requires_grad=False)
        else:
            a_prev = torch.full((b, 1, 1, 1), self.alphas_prev[0], device=device, requires_grad=False) 
        sigma_t = torch.full((b, 1, 1, 1), self.sigmas[original_index], device=device, requires_grad=False) 
            
        e_t = (x - pred_x0 * a_t.sqrt())/sqrt_one_minus_at
        
        sigma_t = torch.full((b, 1, 1, 1), self.sigmas[original_index], device=device, requires_grad=False) 
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        x_prev = a_prev.sqrt() * + dir_xt + noise
            
        return x_prev, pred_x0, losses


    def p_sample_ddim(self, x, c, t, index, prev_index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, n_shots=4):
        b, *_, device = *x.shape, x.device


        a_t = torch.full((b, 1, 1, 1), self.alphas[index], device=device, requires_grad=False) # Needed for ReSampling
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.sqrt_one_minus_alphas[index], device=device, requires_grad=False)
        if prev_index >= 0:
            a_prev = torch.full((b, 1, 1, 1), self.alphas[prev_index], device=device, requires_grad=False)
        else:
            a_prev = torch.full((b, 1, 1, 1), self.alphas_prev[0], device=device, requires_grad=False) 
        sigma_t = torch.full((b, 1, 1, 1), self.sigmas[index], device=device, requires_grad=False) 

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
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0, e_t


    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)


    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec



    def ddecode(self, x_latent, cond=None, t_start=50, temp = 1, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps, temperature = temp, 
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec


               
