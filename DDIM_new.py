"""
DDIM    Script Version Apr 17th 2023
"""

import os
import gc
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import torch
from torch.cuda import amp
from torchmetrics import MeanMetric
from IPython.display import display
from tools import get, make_a_grid_based_cv2_npy, cv2_to_pil, make_a_grid_based_PIL_npy, frames2vid_for_cv2frames
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
class Diffusion_setting:
    def __init__(self, num_diffusion_timesteps=1000, img_shape=(3, 64, 64), device="cpu"):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device

        self.initialize()

    def initialize(self):
        # calculate all the settings for every timesteps, store in a tensor matrix
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.betas = self.get_betas()  # a set of all beta(i)s
        self.alphas = 1 - self.betas  # a set of all alpha(i)s
        self.one_by_sqrt_alpha_s = 1. / torch.sqrt(self.alphas)
        self.sqrt_beta_s = torch.sqrt(self.betas)  # the var for denoising

        # a single calculated cumulative values
        self.alpha_cumulative = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def get_betas(self):
        # linear schedule, following original ddpm paper
        scale = 1000 / self.num_diffusion_timesteps  # scale to 1 (under 1000 timesteps)
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
                              dtype=torch.float32, device=self.device)

# 从x0一步到xt
def forward_diffusion(DS: Diffusion_setting, X_0: torch.Tensor, timestep: torch.Tensor):
    """
    Diffuse X(0) directly to the X(t) of the timestep t

    Mathematically we proved the X(t) can be calculated based on Betas (Alphas) and X(0) and a ep ~ N(0,1)

    :param DS: Diffusion_settings
    :param X_0: input image batch
    :param timestep: a batch of timesteps required to be diffused [B]

    :return: Diffused X(t)
    """
    # sample a batch of Noise ~ N(0,1)
    eps = torch.randn_like(X_0)  # [B,C,H,W]
    # Images scaled to X_t_batch/sqrt(alpha_t)
    mean = get(DS.sqrt_alpha_cumulative, idxs=timestep) * X_0  # [B,1,1,1] * [B,C,H,W] = [B,C,H,W]
    # Noises scaled to X_t_batch/[1-sqrt(alpha_all)]
    std_dev = get(DS.sqrt_one_minus_alpha_cumulative, idxs=timestep)  # [B,1,1,1]
    # X_t_batch of scaled inputs + scaled noise
    sample = mean + std_dev * eps  # [B,C,H,W] + [B,1,1,1] * [B,C,H,W] = [B,C,H,W]

    return sample, eps  # return ... , gt noise --> model predicts this

# 对于每一个图片给一个t，并且记录真实的noise
# Algorithm 1: Training with forward
def train_one_epoch(model, DS, dataloader, optimizer, loss_scaler, loss_fn, epoch, total_epochs, timesteps, device):
    """
    Training (estimating the noise used in diffusing)
    for each sample the timestep t is randomly assigned and then diffused to that step

    :param model:
    :param DS: Diffusion_settings
    :param dataloader:
    :param optimizer:
    :param loss_scaler:
    :param loss_fn:
    :param epoch:
    :param total_epochs:
    :param timesteps: total timesteps in ddpm setting
    :param device:
    :return:
    """
    # use MeanMetric to log loss
    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{total_epochs}")

        for X_0_batch, _ in dataloader:
            tq.update(1)

            # Assign a batch of timesteps to each X0 sample
            batch_timesteps = torch.randint(low=1, high=timesteps, size=(X_0_batch.shape[0],), device=device)

            # Diffuse the batch of X0 to their required step of t
            X_t_batch, Real_noise_batch = forward_diffusion(DS, X_0_batch, batch_timesteps)

            with amp.autocast():
                # the model are asked to predict the noise added in diffusing
                Pred_noise = model(X_t_batch, batch_timesteps)
                loss = loss_fn(Real_noise_batch, Pred_noise)

            # optimizer and scaler do the loss bp and update
            optimizer.zero_grad(set_to_none=True)
            loss_scaler.scale(loss).backward()

            loss_scaler.step(optimizer)
            loss_scaler.update()

            # log the noise predication loss
            loss_value = loss.detach().item()
            loss_record.update(loss_value)
            # tqdm print loss val
            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        # MeanMetric calculate loss mean
        mean_loss = loss_record.compute().item()
        # tqdm print mean_loss val
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss

def Denoising_onestep(model, DS: Diffusion_setting, X_t: torch.Tensor, timestep: torch.Tensor, eta, start_at_T=False):
    """
    Denoise X(t) to the X(t-1) using estimated noise at the timestep t

    Mathematically we proved the X(t) can be calculated based on Betas (Alphas) and X(0) and a ep ~ N(0,1)
    :param model:
    :param DS: Diffusion_settings
    :param X_t: input diffused-image batch : [B,C,H,W]
    :param timestep: a batch of timesteps required to be denoised: [B]
    :param eta: control the amount of noise added in the sampling process
    :param tau: control the speed of the denoising process
    :param start_at_T: the perturbation at step 0 (T) should be 0

    :return: X_t-1
    """
    # sample a batch of Noise ~ N(0,1)
    eps = torch.randn_like(X_t) if not start_at_T else torch.zeros_like(X_t)
    # based on XT to guess what are 'added noise' (denoise)
    predicted_noise = model(X_t, timestep)

    beta_t = get(DS.betas, timestep)  # [B] a batch of beta_t
    one_by_sqrt_alpha_t = get(DS.one_by_sqrt_alpha_s, timestep)  # [B] a batch of 1/sqrt(alpha_t)
    sqrt_one_minus_alpha_cumulative_t = get(DS.sqrt_one_minus_alpha_cumulative, timestep)  # [B] a batch of sqrt(1-alpha_all)

    mean = one_by_sqrt_alpha_t * (X_t - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
    var = eta * get(DS.sqrt_beta_s, timestep)  # Use eta to control the noise addition

    X_t_minus_1 = mean + var * eps

    return X_t_minus_1

# Algorithm 2: Inference with Denoise Sampling
@torch.inference_mode()
def reverse_diffusion(model, DS, img_shape=(3, 64, 64), num_images=5, nrow=8, device="cpu",
                      save_path=None, generate_video=True, eta=1.0, tau=1, scheduling='uniform'):
    # Assign a batch of X(T) (Noise ~ N(0,1)) to x_t (t = T)
    x_T = torch.randn((num_images, *img_shape), device=device)  # [num_images, C, H, W]
    x_t = x_T  # the first X_t

    model.eval()

    if generate_video:  # build the results into frames of a video
        frames_list = []  # all frames

    def _get_process_scheduling(DS, reverse=True):
        if scheduling == 'uniform':
            diffusion_process = list(range(0, DS.num_diffusion_timesteps, tau)) + [DS.num_diffusion_timesteps - 1]
        elif scheduling == 'exp':
            diffusion_process = (np.linspace(0, np.sqrt(DS.num_diffusion_timesteps * 0.8), tau) ** 2)
            diffusion_process = [int(s) for s in list(diffusion_process)] + [DS.num_diffusion_timesteps - 1]
        else:
            assert 'Not Implementation'

        if reverse:
            # 如果是反向过程，反转时间步序列
            diffusion_process = reversed(diffusion_process)

        # 将时间步序列转换为列表
        diffusion_process = list(diffusion_process)

        # 如果是反向过程，需要配对相邻的时间步
        if reverse:
            diffusion_process = list(zip(diffusion_process[:-1], diffusion_process[1:]))

        return diffusion_process

    # 获取调度过程并转换为列表
    ddim_process = _get_process_scheduling(DS, reverse=True)

    # 使用 tqdm 进行迭代
    for time_step_pair in tqdm(iterable=ddim_process, total=len(ddim_process), dynamic_ncols=False, desc="Sampling :: ",
                               position=0):
        # 解包元组，获取当前时间步
        _, time_step = time_step_pair  # 假设你想要的是第二个值

        # 生成 timesteps_batch
        timesteps_batch = torch.ones(num_images, dtype=torch.long, device=device) * time_step

        # 其他代码保持不变
        x_t = Denoising_onestep(model, DS, x_t, timesteps_batch, eta, start_at_T=True if time_step == 1 else False)

        # 如果需要保存帧，继续执行相关代码
        if generate_video:
            # the generated image is C,H,W and C is RGB format (PIL), values in 0-1 range
            grid_cv2_npy = make_a_grid_based_cv2_npy(x_t, nrow=nrow)
            # added to all the frames
            frames_list.append(grid_cv2_npy)

    if generate_video:  # Generate and save video of the entire reverse process
        frames2vid_for_cv2frames(frames_list, save_path)
        # Display the image at the final timestep of the reverse process.
        pil_image = cv2_to_pil(frames_list[-1])  # PIL format
        display(pil_image)
        return None

    else:  # Display and save the image at the final timestep of the reverse process.
        pil_image = make_a_grid_based_PIL_npy(x_t, nrow=nrow)
        pil_image.save(save_path, format=save_path[-3:].upper())  # save PIL image
        display(pil_image)  # show PIL image
        return None


def train(model, sd, dataloader, optimizer, scaler, loss_fn, img_shape, total_epochs, timesteps,
          log_dir, checkpoint_dir, generate_video=False, device='cpu', checkpoint_name="ddim.tar", eta=1.0, tau=1):

    ext = ".mp4" if generate_video else ".png"

    for epoch in range(1, total_epochs + 1):
        torch.cuda.empty_cache()
        gc.collect()

        # Algorithm 1: Training
        train_one_epoch(model, sd, dataloader, optimizer, scaler, loss_fn, epoch,
                        total_epochs=total_epochs, timesteps=timesteps,
                        device=device)

        if epoch % 1 == 0:
            save_path = os.path.join(log_dir, f"{epoch}{ext}")

            # Algorithm 2: Sampling
            reverse_diffusion(model, sd, img_shape=img_shape,
                              num_images=32, generate_video=generate_video,
                              save_path=save_path, device=device, eta=eta, tau=tau, scheduling='uniform')

            # clear_output()
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model": model.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, checkpoint_name))
            del checkpoint_dict


def inference(model, sd, img_shape, num_images=64, nrow=8,
              log_dir="inference_results", generate_video=False, device='cpu', eta=1.0, tau=1):

    os.makedirs(log_dir, exist_ok=True)
    ext = ".mp4" if generate_video else ".png"

    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)

    reverse_diffusion(
        model,
        sd,
        img_shape=img_shape,
        num_images=num_images,
        generate_video=generate_video,
        save_path=save_path,
        device=device,
        nrow=nrow,
        eta=eta,
        tau=tau,
        scheduling='uniform'
    )
    print('save_path:', save_path)
@torch.inference_mode()
def denoising_reverse_diffusion(model, DS, x_T , img_shape=(3, 64, 64), num_images=5, nrow=8, device="cpu",
                      save_path=None, generate_video=True, eta=1.0, tau=1, scheduling='uniform'):
    # Assign a batch of X(T) (Noise ~ N(0,1)) to x_t (t = T)
    x_t = x_T  # the first X_t

    model.eval()

    if generate_video:  # build the results into frames of a video
        frames_list = []  # all frames

    def _get_process_scheduling(DS, reverse=True):
        if scheduling == 'uniform':
            diffusion_process = list(range(0, DS.num_diffusion_timesteps, tau)) + [DS.num_diffusion_timesteps - 1]
        elif scheduling == 'exp':
            diffusion_process = (np.linspace(0, np.sqrt(DS.num_diffusion_timesteps * 0.8), tau) ** 2)
            diffusion_process = [int(s) for s in list(diffusion_process)] + [DS.num_diffusion_timesteps - 1]
        else:
            assert 'Not Implementation'

        if reverse:
            # 如果是反向过程，反转时间步序列
            diffusion_process = reversed(diffusion_process)

        # 将时间步序列转换为列表
        diffusion_process = list(diffusion_process)

        # 如果是反向过程，需要配对相邻的时间步
        if reverse:
            diffusion_process = list(zip(diffusion_process[:-1], diffusion_process[1:]))

        return diffusion_process

    # 获取调度过程并转换为列表
    ddim_process = _get_process_scheduling(DS, reverse=True)

    # 使用 tqdm 进行迭代
    for time_step_pair in tqdm(iterable=ddim_process, total=len(ddim_process), dynamic_ncols=False, desc="Sampling :: ",
                               position=0):
        # 解包元组，获取当前时间步
        _, time_step = time_step_pair  # 假设你想要的是第二个值

        # 生成 timesteps_batch
        timesteps_batch = torch.ones(num_images, dtype=torch.long, device=device) * time_step

        # 其他代码保持不变
        x_t = Denoising_onestep(model, DS, x_t, timesteps_batch, eta, start_at_T=True if time_step == 1 else False)

        # 如果需要保存帧，继续执行相关代码

        # Assign a batch of timesteps (value all at t) to each X(t): [B]
        timesteps_batch = torch.ones(num_images, dtype=torch.long, device=device) * time_step

        x_t = Denoising_onestep(model, DS, x_t, timesteps_batch, eta, start_at_T=True if time_step == 1 else False)

        # put the intermediate results into a frame
        if generate_video:
            # the generated image is C,H,W and C is RGB format (PIL), values in 0-1 range
            grid_cv2_npy = make_a_grid_based_cv2_npy(x_t, nrow=nrow)
            # added to all the frames
            frames_list.append(grid_cv2_npy)

    if generate_video:  # Generate and save video of the entire reverse process
        frames2vid_for_cv2frames(frames_list, save_path)
        # Display the image at the final timestep of the reverse process.
        pil_image = cv2_to_pil(frames_list[-1])  # PIL format
        display(pil_image)
        return None

    else:  # Display and save the image at the final timestep of the reverse process.
        pil_image = make_a_grid_based_PIL_npy(x_t, nrow=nrow)
        pil_image.save(save_path, format=save_path[-3:].upper())  # save PIL image
        display(pil_image)  # show PIL image
        return None


@torch.inference_mode()
def probability_flow(model, DS, img_shape=(3, 64, 64), num_images=5, nrow=8, device="cpu",
                     save_path=None, generate_video=True, eta=1.0, tau=1, scheduling='uniform', reverse=False, x_t=None):
    # 如果 x_t 没有传入，则随机生成初始噪声 x_T
    if x_t is None:
        x_T = torch.randn((num_images, *img_shape), device=device)
        x_t = x_T
    else:
        # 确保 x_t 的形状和设备正确
        assert x_t.shape == (num_images, *img_shape), f"Expected shape {(num_images, *img_shape)}, but got {x_t.shape}"
        x_t = x_t.to(device)

    def reparameterize_sigma(idx):
        alpha_cumulative = DS.alpha_cumulative[idx]
        return torch.sqrt((1 - alpha_cumulative) / alpha_cumulative)

    def reparameterize_x(x, idx):
        alpha_cumulative = DS.alpha_cumulative[idx]
        return x / torch.sqrt(alpha_cumulative)

    model.eval()

    def _get_process_scheduling(DS, reverse=True):
        if scheduling == 'uniform':
            diffusion_process = list(range(0, DS.num_diffusion_timesteps, tau)) + [DS.num_diffusion_timesteps - 1]
        elif scheduling == 'exp':
            diffusion_process = (np.linspace(0, np.sqrt(DS.num_diffusion_timesteps * 0.8), tau) ** 2)
            diffusion_process = [int(s) for s in list(diffusion_process)] + [DS.num_diffusion_timesteps - 1]
        else:
            assert 'Not Implemented'

        if reverse:
            # 如果是反向过程，先将 diffusion_process 转换为列表，再反转
            diffusion_process = list(diffusion_process)
            diffusion_process = diffusion_process[::-1]  # 使用切片反转列表
            # 配对相邻的时间步
            diffusion_process = list(zip(diffusion_process[:-1], diffusion_process[1:]))
        else:
            # 如果是正向过程，直接使用时间步序列
            # 配对相邻的时间步
            diffusion_process = list(zip(diffusion_process[:-1], diffusion_process[1:]))

        return diffusion_process

    dif_process = _get_process_scheduling(DS, reverse=reverse)
    # 使用 tqdm 进行迭代
    if generate_video:
        frames_list = []

    for time_step_pair in tqdm(iterable=dif_process, total=len(dif_process), dynamic_ncols=False, desc="Sampling :: ",
                               position=0):
        idx_delta_t, idx = time_step_pair  # 假设你想要的是第二个值

        # 确保 idx 是一个张量
        idx_tensor = torch.tensor([idx] * num_images, dtype=torch.long, device=device)

        x_bar_delta_t = reparameterize_x(x_t, idx) + 0.5 * (
            reparameterize_sigma(idx_delta_t) ** 2 - reparameterize_sigma(idx) ** 2) / reparameterize_sigma(idx) * model(x_t, idx_tensor)
        x_t = x_bar_delta_t * torch.sqrt(DS.alpha_cumulative[idx_delta_t])

        # 保存中间结果
        if generate_video:
            grid_cv2_npy = make_a_grid_based_cv2_npy(x_t, nrow=nrow)
            frames_list.append(grid_cv2_npy)

    # 生成视频或保存图像
    if generate_video:
        # 生成并保存视频
        frames2vid_for_cv2frames(frames_list, save_path)
        # 显示最终结果
        pil_image = cv2_to_pil(frames_list[-1])
        display(pil_image)
    else:
        # 保存最终结果
        pil_image = make_a_grid_based_PIL_npy(x_t, nrow=nrow)
        pil_image.save(save_path, format=save_path[-3:].upper())
        display(pil_image)

    return x_t