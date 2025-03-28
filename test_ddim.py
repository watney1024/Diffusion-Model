
"""
Inference Script Version Apr 17th 2023


"""

from dataclasses import dataclass

from tools import *
from unet import *
#from DDIM import *
from DDIM_new import *
from IPython.display import display, HTML
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "MNIST"  # "Cifar-10", "Cifar-100", "Flowers"

    # Path to log inference images and save checkpoints
    root = "./Logs_Checkpoints"
    os.makedirs(root, exist_ok=True)

    # Current log and checkpoint directory.
    log_folder = 'version_4'  # specific a folder name to load
    checkpoint_name = "ddim_1.tar"


@dataclass
class TrainingConfig:
    TIMESTEPS = 1000  # Define number of diffusion timesteps
    IMG_SHAPE = (1, 32, 32) if BaseConfig.DATASET == "MNIST" else (3, 32, 32)
    NUM_EPOCHS = 25
    BATCH_SIZE = 16
    LR = 2e-4

    NUM_WORKERS = 4 if str(BaseConfig.DEVICE) != "cpu" else 0


@dataclass
class ModelConfig:  # setting up attention unet
    BASE_CH = 64
    BASE_CH_MULT = (1, 2, 4, 8)
    APPLY_ATTENTION = (False, False, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 2


sd = Diffusion_setting(num_diffusion_timesteps=TrainingConfig.TIMESTEPS,
                       img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE)

generate_video = False

# test
log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig(), inference=True)

model = UNet(
    input_channels=TrainingConfig.IMG_SHAPE[0],
    output_channels=TrainingConfig.IMG_SHAPE[0],
    base_channels=ModelConfig.BASE_CH,
    base_channels_multiples=ModelConfig.BASE_CH_MULT,
    apply_attention=ModelConfig.APPLY_ATTENTION,
    dropout_rate=ModelConfig.DROPOUT_RATE,
    time_multiple=ModelConfig.TIME_EMB_MULT,
)
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, BaseConfig.checkpoint_name), map_location='cpu')["model"], False)
model.to(BaseConfig.DEVICE)

# 获取数据集
dataset = get_dataset(dataset_name=BaseConfig.DATASET)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
device_dataloader = DeviceDataLoader(dataloader, BaseConfig.DEVICE)

# 获取一批数据
X_0_batch, _ = next(iter(device_dataloader))

# 展示原图
original_images = make_a_grid_based_PIL_npy(X_0_batch, nrow=4)
original_images.save(os.path.join(log_dir, "original_images.png"))
display(original_images)

# 扩散过程
timesteps = torch.full((X_0_batch.shape[0],), TrainingConfig.TIMESTEPS-1, device=BaseConfig.DEVICE, dtype=torch.long)
X_t_batch, _ = forward_diffusion(sd, X_0_batch, timesteps)

# 展示扩散后的图
noisy_images = make_a_grid_based_PIL_npy(X_t_batch, nrow=4)
noisy_images.save(os.path.join(log_dir, "noisy_images.png"))
display(noisy_images)

os.makedirs(log_dir, exist_ok=True)
ext = ".mp4" if generate_video else ".png"

filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

save_path = os.path.join(log_dir, filename)
# 还原过程
denoising_reverse_diffusion(model, sd, x_T= X_t_batch, img_shape=TrainingConfig.IMG_SHAPE, num_images=X_0_batch.shape[0], nrow=4,
                  save_path=save_path, generate_video=generate_video, device=BaseConfig.DEVICE, eta=0, tau=5)
