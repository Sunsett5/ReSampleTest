from ldm_inverse.condition_methods import get_conditioning_method
#from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddim_retrace import DDIMSampler
from data.dataloader import get_dataset, get_dataloader
from scripts.utils import clear_color, mask_generator
import matplotlib.pyplot as plt
from ldm_inverse.measurements import get_noise, get_operator
from functools import partial
import numpy as np
from model_loader import load_model_from_config, load_yaml
import os
import torch
import torchvision.transforms as transforms
import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import pandas as pd

def compute_nmse(y_true_temp, y_pred_temp, eps=1e-6):
    if isinstance(y_true_temp, np.ndarray):
        y_true = torch.from_numpy(y_true_temp)
    else:
        y_true = y_true_temp.clone()
    
    if isinstance(y_pred_temp, np.ndarray):
        y_pred = torch.from_numpy(y_pred_temp)
    else:
        y_pred = y_pred_temp.clone()

    device = y_true.device
    y_pred = y_pred.to(device)

    # Reshape from (256, 256, 3) to (3, 256, 256)
    if y_true.shape[-1] == 3:
        y_true = y_true.permute(2, 0, 1)
        y_pred = y_pred.permute(2, 0, 1)

    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    loss = torch.mean(torch.pow(y_true_flat - y_pred_flat, 2), dim=-1) / (
        torch.mean(torch.pow(y_true_flat, 2), dim=-1) + eps
    )
    return torch.mean(loss)

def compute_psnr(y_true_temp, y_pred_temp, max_val=1.0, eps=1e-6):
    if isinstance(y_true_temp, np.ndarray):
        y_true = torch.from_numpy(y_true_temp)
    else:
        y_true = y_true_temp.clone()
    
    if isinstance(y_pred_temp, np.ndarray):
        y_pred = torch.from_numpy(y_pred_temp)
    else:
        y_pred = y_pred_temp.clone()

    device = y_true.device
    y_pred = y_pred.to(device)

    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    # Clamp the values to the range [0, max_val]
    y_true = torch.clamp(y_true, 0, max_val)
    y_pred = torch.clamp(y_pred, 0, max_val)

    msef = torch.mean(torch.pow(y_true_flat - y_pred_flat, 2), dim=-1)
    maxf = torch.amax(y_true_flat, dim=-1)
    psnr = 20 * torch.log10(maxf) - 10 * torch.log10(msef)
    return torch.mean(psnr)

def compute_ssim(y_true_temp, y_pred_temp):
    y_true_np = y_true_temp
    y_pred_np = y_pred_temp
    meas_ssim, diff = ssim(y_true_np, y_pred_np, full=True, multichannel=True, channel_axis=2, data_range=1.0)
    return meas_ssim

def compute_lpips(y_true_temp, y_pred_temp):
    if isinstance(y_true_temp, np.ndarray):
        y_true = torch.from_numpy(y_true_temp)
    else:
        y_true = y_true_temp.clone()
    
    if isinstance(y_pred_temp, np.ndarray):
        y_pred = torch.from_numpy(y_pred_temp)
    else:
        y_pred = y_pred_temp.clone()

    # Reshape from (256, 256, 3) to (3, 256, 256)
    if y_true.shape[-1] == 3:
        y_true = y_true.permute(2, 0, 1)
        y_pred = y_pred.permute(2, 0, 1)

    device = y_true.device
    y_pred = y_pred.to(device)
    loss_fn = lpips.LPIPS(net='vgg').to(device)

    lpips_score = loss_fn(y_true, y_pred)
    return lpips_score.item()


def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_config, args.gpu)

    return model


parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str)
parser.add_argument('--ldm_config', default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml", type=str)
parser.add_argument('--diffusion_config', default="models/ldm/model.ckpt", type=str)
parser.add_argument('--task_config', default="configs/tasks/gaussian_deblur_config.yaml", type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--ddim_steps', default=500, type=int)
parser.add_argument('--ddim_eta', default=0.0, type=float)
parser.add_argument('--n_samples_per_class', default=1, type=int)
parser.add_argument('--ddim_scale', default=1.0, type=float)

args = parser.parse_args()

# Load configurations
task_config = load_yaml(args.task_config)

# Device setting
device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
print(f"Device set to {device_str}.")
device = torch.device(device_str) 

# Loading model
model = get_model(args)
sampler = DDIMSampler(model) # Sampling using DDIM

# Prepare Operator and noise
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])
print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

# Prepare conditioning method
cond_config = task_config['conditioning']
cond_method = get_conditioning_method(cond_config['method'], model, operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning
print("Condition", cond_method)
print(f"Conditioning method", cond_method.conditioning)
print(f"Conditioning sampler : {task_config['conditioning']['main_sampler']}")

# Instantiating sampler
sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward,
                                        S=args.ddim_steps,
                                        cond_method=task_config['conditioning']['main_sampler'],
                                        conditioning=None,
                                        ddim_use_original_steps=True,
                                        batch_size=args.n_samples_per_class,
                                        shape=[3, 64, 64], # Dimension of latent space
                                        verbose=False,
                                        unconditional_guidance_scale=args.ddim_scale,
                                        unconditional_conditioning=None, 
                                        eta=args.ddim_eta)

# Working directory
out_path = os.path.join(args.save_dir)
os.makedirs(out_path, exist_ok=True)
for img_dir in ['input', 'recon', 'progress', 'label']:
    os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

# Prepare dataloader
data_config = task_config['data']
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )
dataset = get_dataset(**data_config, transforms=transform)
loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
for i, ref_img in enumerate(loader):
    print(i)

# Exception) In case of inpainting, we need to generate a mask 
if measure_config['operator']['name'] == 'inpainting':
  mask_gen = mask_generator(**measure_config['mask_opt'])

metric_results = {
    'psnr': [],
    'nmse': [],
    'ssim': [],
    'lpips': []
}

# Do inference
for i, ref_img in enumerate(loader):

    print(f"Inference for image {i}")
    fname = str(i).zfill(3)
    ref_img = ref_img.to(device)

    # Exception) In case of inpainting
    if measure_config['operator'] ['name'] == 'inpainting':
      mask = mask_gen(ref_img)
      mask = mask[:, 0, :, :].unsqueeze(dim=0)
      operator_fn = partial(operator.forward, mask=mask)
      measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
      sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator_fn=operator_fn)

      # Forward measurement model
      y = operator_fn(ref_img)
      y_n = noiser(y)

    else:
      y = operator.forward(ref_img)
      y_n = noiser(y).to(device)

    # Sampling
    samples_ddim, _ = sample_fn(measurement=y_n)
    
    x_samples_ddim = model.decode_first_stage(samples_ddim.detach())

    # Post-processing samples
    label = clear_color(y_n)
    reconstructed = clear_color(x_samples_ddim)
    true = clear_color(ref_img)

    # Saving images
    plt.imsave(os.path.join(out_path, 'input', fname+'_true.png'), true)
    plt.imsave(os.path.join(out_path, 'label', fname+'_label.png'), label)
    plt.imsave(os.path.join(out_path, 'recon', fname+'_recon.png'), reconstructed)

    meas_psnr_all = psnr(true, reconstructed).item()
    meas_nmse_all = compute_nmse(true, reconstructed).item()
    meas_ssim_all = compute_ssim(true, reconstructed)
    meas_lpips_all = compute_lpips(true, reconstructed)

    metric_results['psnr'].append(meas_psnr_all)
    metric_results['nmse'].append(meas_nmse_all)
    metric_results['ssim'].append(meas_ssim_all)
    metric_results['lpips'].append(meas_lpips_all)

    print("measurement psnr: ", meas_psnr_all)
    print("measurement nmse: ", meas_nmse_all)
    print("measurement ssim: ", meas_ssim_all)
    print("measurement lpips: ", meas_lpips_all)
""" 
    #plot norm and hhi (twinax)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax1 = ax.twinx()
    line1 = ax.plot(sampler.norm_history, label='Grad Norm')
    line2 = ax1.plot(sampler.hhi_history, label='HHI', color='orange')
    ax.set_xlabel('Iteration')
    ax.set_title('Grad Norm and HHI')
    ax.set_ylabel('Grad Norm')
    ax1.set_ylabel('HHI')
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='best')
    fig.savefig("norm_HHI_plot.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sampler.ratio_history, label='HHI Ratio')
    ax.set_xlabel('Iteration')
    ax.set_title('HHI Ratio')
    ax.set_ylabel('HHI Ratio')
    ax.legend(loc='best')
    fig.savefig("HHI_ratio_plot.png", dpi=300)
    plt.close(fig) """

# Average metrics
average_psnr = np.mean(metric_results['psnr'])
average_nmse = np.mean(metric_results['nmse'])
average_ssim = np.mean(metric_results['ssim'])
average_lpips = np.mean(metric_results['lpips'])

print("COMPLETE")
print("average psnr: ", average_psnr)
print("average nmse: ", average_nmse)
print("average ssim: ", average_ssim)
print("average lpips: ", average_lpips)

df = pd.DataFrame({
    'psnr': metric_results['psnr'],
    'nmse': metric_results['nmse'],
    'ssim': metric_results['ssim'],
    'lpips': metric_results['lpips'],
})

# Save to Excel
df.to_csv("./excel_results/output_consistent_opti_30_start_100_every_5_NoInject_NoDPS.csv", index=False)

