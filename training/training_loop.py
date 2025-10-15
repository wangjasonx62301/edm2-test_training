# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc
from torch.utils.tensorboard import SummaryWriter
import torchvision

from generate_images import generate_images, edm_sampler    

#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

torch.autograd.set_detect_anomaly(True)

def guidance_score_scheduler(t, initial=0.5, final=2.5, T=torch.tensor(24000000)):
    # exponential schedule
    ratio = t / T
    return initial * ((final / initial) ** ratio)
    # return initial + (final - initial) * ratio

def normal_and_guidance_interpolation_scheduler(t, initial=0.3, final=0.5, T=torch.tensor(24000000)):
    # linear schedule
    ratio = t / T
    return initial + (final - initial) * ratio

# def continuous_to_discrete(sigma, T=torch.tensor(24000000), rho=7):
#     return ((sigma**(1/rho) - 0.002**(1/rho)) / (80**(1/rho) - 0.002**(1/rho)) * T).long()

def continuous_to_discrete(sigma, T=24000000, sigma_min=0.002, sigma_max=80):
    sigma = sigma.to(torch.float32)
    return (sigma * T).long()

# def discrete_to_continuous(t, T=torch.tensor(24000000), rho=7):
#     a = 0.002**(1/rho)
#     b = 80**(1/rho)
#     sigma = (a + (b - a) * (t / T))**rho
#     sigma_norm = (sigma - 0.002) / (80 - 0.002)  
#     return sigma_norm

def discrete_to_continuous(t, T=24000000):
    t = t.to(torch.float32)
    return t / T

def discrete_interval_scheduler(t, min_interval=10, max_interval=10000, T=24000000):
    ratio = t / T
    return (min_interval + (max_interval - min_interval) * ratio).long() if type(min_interval) is int else (min_interval + (max_interval - min_interval) * ratio).float()
    

def forward_process_to_noise(x0, sigma, sigma_data=0.5):
    """Convert clean images to noisy images."""
    noise = torch.randn_like(x0)
    return x0 * (sigma_data / (sigma**2 + sigma_data**2).sqrt()) + noise * (sigma**2 / (sigma**2 + sigma_data**2)).sqrt()

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.guidance_scale = 2.0
        self.sigma_gap = -0.03
        self.g_score_schedule = 0.0001

    def __call__(self, net, images, labels=None, global_step=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        # first sigma
        T = 24000000
        P_mean_s = 0.85
        start_guidance = False
        guidance_min_gamma = 1.5
        guidance_max_gamma = 2.8
        guidance_score_scale_min = 0.001
        guidance_score_scale_max = 0.05
        # print(self.P_std, self.P_mean)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # sigma_s
        # sigma_s = torch.clamp(sigma.clone() - 0.001, min=0.0)
        # discrete step
        
        # mode
        mode = f'dis_2_cont_with_linear_and_scheduler_GuidanceGamma_{guidance_min_gamma}-{guidance_max_gamma}_GuidanceScale_{guidance_score_scale_min}-{guidance_score_scale_max}_Pmean_{self.P_mean}_Pstd_{self.P_std}'

        sigma_discrete = continuous_to_discrete(sigma)
        
        interval = discrete_interval_scheduler(sigma_discrete, T=torch.tensor(T).to(images.device))

        self_g_interval = discrete_interval_scheduler(global_step, T=torch.tensor(T).to(images.device), min_interval=guidance_min_gamma, max_interval=guidance_max_gamma)

        # sigma_s_discrete = torch.clamp(sigma_discrete - interval, min=sigma_discrete // 2)
        
        # sigma_s = discrete_to_continuous(sigma_s_discrete)
        # rnd_normal_s = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        
        sigma_s = (rnd_normal * self.P_std - P_mean_s).exp()

        g_score_scale = guidance_score_scheduler(global_step, T=torch.tensor(T).to(images.device), initial=guidance_score_scale_min, final=guidance_score_scale_max)
        # dist.print0(f'global_step: {global_step}, g_score_scale: {g_score_scale}')
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # primary noise
        noise = torch.randn_like(images) * sigma
        # secondary noise
        # noise_s = torch.randn_like(images) * sigma_s
        
        denoised, logvar = net(images + noise, sigma, labels, return_logvar=True)
        # noise_s = forward_process_to_noise(denoised, sigma_s, self.sigma_data)
        noise_s = torch.randn_like(images) * sigma_s
        denoised_s, _ = net((images + noise_s), sigma_s, labels, return_logvar=True)
        guidance = denoised + self_g_interval * (denoised - denoised_s)

        # guidance = (guidance - images).square().mean(dim=[1,2,3])
        # dist.print0(f'guidance distance mean: {torch.mean(torch.abs(denoised - denoised_s))}')
        
        g_score = g_score_scale * guidance
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        loss_guidance = g_score.mean()
        interp_alpha = normal_and_guidance_interpolation_scheduler(global_step, T=torch.tensor(T).to(images.device))


        # loss_guidance = torch.clamp(loss_guidance, min=0.0)
        if torch.isnan(loss_guidance) or torch.isinf(loss_guidance):
            loss_guidance = torch.tensor(0.0, device=images.device)
            dist.print0('warning: nan guidance loss!')
        
        # scheduled interpolation between normal loss and guidance loss
        # if 5000000 > global_step > 600000:
        if global_step < 600000:
            start_guidance = True
            loss = (1 - interp_alpha) * loss + interp_alpha * loss_guidance
        
        else: start_guidance = False
        
        dist.print0(f'start_guidance: {start_guidance} global_step {global_step}, interp_alpha {interp_alpha:.12f}, g_score_scale {g_score_scale:.12f}, g_score {g_score.mean().item():.12f}, loss {loss.mean().item():.4f}, self_g_interval {self_g_interval.mean().item():.4f}, sigma {sigma.mean().item():.4f}, sigma_s {sigma_s.mean().item():.4f}')

        
        # loss = loss.mean() + loss_guidance
        training_mode = f"{T}-{mode}--{P_mean_s}"
        return loss, g_score_scale, training_mode
        
        # rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        # sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # # sigma_s
        # sigma_s = torch.clamp(sigma.clone() + 0.03, max=1.0)
        # noise_s = torch.randn_like(images) * sigma_s

        # weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # noise = torch.randn_like(images) * sigma
        # denoised, logvar = net(images + noise, sigma, labels, return_logvar=True)
        # denoised_s, _ = net((images + noise_s), sigma_s, labels, return_logvar=True)

        # loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        # return loss

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    eval_interval       = None,     # Interval for evaluating FID and saving sample images. None = disable.
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ], max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    
    # tensorboard summary writer
    writer = None
    if dist.get_rank() in [0, 1, 2, 3]:
        tb_log_dir = os.path.join(run_dir, 'tensorboard')
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        dist.print0(f'create tensorboard summary writer at {tb_log_dir}')
    
    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    loss_history = []
    while True and state.cur_nimg < 24000000:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() in [0, 1, 2, 3]:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()
                
                # also write to tensorboard
                try:
                    if writer is not None:
                        stats_dict = training_stats.default_collector.as_dict()
                        # write scalar metrics
                        for name, val in stats_dict.items():
                            try:
                                writer.add_scalar(name, float(val.mean), state.cur_nimg)
                            except Exception:
                                # ignore any weird metric formatting
                                pass
                        # some additional useful scalars
                        writer.add_scalar('Resources/cpu_mem_gb', cpu_memory_usage / 2**30, state.cur_nimg)
                        try:
                            writer.add_scalar('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30, state.cur_nimg)
                            writer.add_scalar('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30, state.cur_nimg)
                        except Exception:
                            pass
                        writer.flush()
                except Exception as e:
                    dist.print0(f'Warning: TensorBoard writer failed to write stats: {e}')

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True
        
        # Save network snapshot.
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data # conserve memory

        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(net)

        # Done?
        if done:
            break
        
        g_score_scale = None
        scalar_loss = None
        training_mode = None
        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                loss, g_score_scale, training_mode = loss_fn(net=ddp, images=images, labels=labels.to(device), global_step=state.cur_nimg)
                training_stats.report('Loss/loss', loss)
                scalar_loss = float(loss.mean().detach().cpu().item())
                # dist.print0(f'mean loss: {np.mean(loss_history[:-100])}, cur loss: {scalar_loss}')
                if len(loss_history) > 100 and not \
                scalar_loss < 2.0 and not \
                scalar_loss - 0.15 < np.mean(loss_history[-10:]) < scalar_loss + 0.15:
                    dist.print0(f'mean loss: {np.mean(loss_history[:-10])}, cur loss: {scalar_loss}')
                    dist.print0('warning: loss seems to be stuck, breaking out of training loop')
                    loss_history.append(scalar_loss)
                    del scalar_loss, loss, g_score_scale
                    continue
                loss_history.append(scalar_loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                
        if dist.get_rank() in [0, 1, 2, 3] and writer is not None:
            try:
                # dist.print0(f'cur_nimg: {state.cur_nimg}, loss: {scalar_loss}')
                writer.add_scalar(f'Train-{training_mode}-{dist.get_rank()}/loss_step', scalar_loss, state.cur_nimg)
                writer.add_scalar(f'Train-{training_mode}-{dist.get_rank()}/guidance_scale', g_score_scale, state.cur_nimg)
            except Exception:
                pass
                # loss_guidance.backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr

        if dist.get_rank() in [0, 1, 2, 3] and writer is not None:
            try:
                writer.add_scalar(f'Train-{training_mode}-{dist.get_rank()}/learning_rate', lr, state.cur_nimg)
            except Exception:
                pass
            
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time
        
        if state.cur_nimg != 0 and state.cur_nimg % status_nimg == 0 and dist.get_rank() == 0 and writer is not None:
        
            dist.print0(f'=== eval at {state.cur_nimg // 1000} kimg ===')
            # generate sample images
            fix_seeds = [0, 1, 2, 3]
            net_ema = ema.get()[0][0] if ema is not None else net
            image_iter = generate_images(
                net=net_ema,
                gnet=None,
                encoder=encoder,
                outdir=None,
                seeds=fix_seeds,
                device=device,
                max_batch_size=16,
                sampler_fn=edm_sampler,
                num_steps=32,
                sigma_min=0.002,
                sigma_max=80,
                rho=7,
            )
            
            for r in image_iter:
                # print(r.images.dtype, r.images.min().item(), r.images.max().item())
                grid = torchvision.utils.make_grid(r.images, nrow=len(fix_seeds),)
                writer.add_image(f'Train-{training_mode}-{dist.get_rank()}/fixed_samples_{state.cur_nimg}', grid, state.cur_nimg)
                break
        
    if dist.get_rank() in [0, 1, 2, 3] and writer is not None:
        try:
            writer.close()
        except Exception:
            pass

#----------------------------------------------------------------------------
