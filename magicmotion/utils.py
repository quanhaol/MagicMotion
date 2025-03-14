import gc
import inspect
from typing import Optional, Tuple, Union

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.utils.torch_utils import is_compiled_module
import cv2
import numpy as np
import pandas as pd
import os
from torchvision.transforms.functional import resize
from diffusers.utils import export_to_video
import os
import tempfile
from typing import Any, Callable, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

import PIL.Image
import PIL.ImageOps
import requests
from diffusers.utils.import_utils import BACKENDS_MAPPING, is_imageio_available

logger = get_logger(__name__)


def get_optimizer(
    params_to_optimize,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.95,
    beta3: float = 0.98,
    epsilon: float = 1e-8,
    weight_decay: float = 1e-4,
    prodigy_decouple: bool = False,
    prodigy_use_bias_correction: bool = False,
    prodigy_safeguard_warmup: bool = False,
    use_8bit: bool = False,
    use_4bit: bool = False,
    use_torchao: bool = False,
    use_deepspeed: bool = False,
    use_cpu_offload_optimizer: bool = False,
    offload_gradients: bool = False,
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()

    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )

    if use_8bit and use_4bit:
        raise ValueError("Cannot set both `use_8bit` and `use_4bit` to True.")

    if (use_torchao and (use_8bit or use_4bit)) or use_cpu_offload_optimizer:
        try:
            import torchao

            torchao.__version__
        except ImportError:
            raise ImportError(
                "To use optimizers from torchao, please install the torchao library: `USE_CPP=0 pip install torchao`."
            )

    if not use_torchao and use_4bit:
        raise ValueError("4-bit Optimizers are only supported with torchao.")

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy", "came"]
    if optimizer_name not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {optimizer_name}. Supported optimizers include {supported_optimizers}. Defaulting to `AdamW`."
        )
        optimizer_name = "adamw"

    if (use_8bit or use_4bit) and optimizer_name not in ["adam", "adamw"]:
        raise ValueError("`use_8bit` and `use_4bit` can only be used with the Adam and AdamW optimizers.")

    if use_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if optimizer_name == "adamw":
        if use_torchao:
            from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit

            optimizer_class = AdamW8bit if use_8bit else AdamW4bit if use_4bit else torch.optim.AdamW
        else:
            optimizer_class = bnb.optim.AdamW8bit if use_8bit else torch.optim.AdamW

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "adam":
        if use_torchao:
            from torchao.prototype.low_bit_optim import Adam4bit, Adam8bit

            optimizer_class = Adam8bit if use_8bit else Adam4bit if use_4bit else torch.optim.Adam
        else:
            optimizer_class = bnb.optim.Adam8bit if use_8bit else torch.optim.Adam

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        init_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "beta3": beta3,
            "eps": epsilon,
            "weight_decay": weight_decay,
            "decouple": prodigy_decouple,
            "use_bias_correction": prodigy_use_bias_correction,
            "safeguard_warmup": prodigy_safeguard_warmup,
        }

    elif optimizer_name == "came":
        try:
            import came_pytorch
        except ImportError:
            raise ImportError("To use CAME, please install the came-pytorch library: `pip install came-pytorch`")

        optimizer_class = came_pytorch.CAME

        init_kwargs = {
            "lr": learning_rate,
            "eps": (1e-30, 1e-16),
            "betas": (beta1, beta2, beta3),
            "weight_decay": weight_decay,
        }

    if use_cpu_offload_optimizer:
        from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

        if "fused" in inspect.signature(optimizer_class.__init__).parameters:
            init_kwargs.update({"fused": True})

        optimizer = CPUOffloadOptimizer(
            params_to_optimize, optimizer_class=optimizer_class, offload_gradients=offload_gradients, **init_kwargs
        )
    else:
        optimizer = optimizer_class(params_to_optimize, **init_kwargs)

    return optimizer


def get_gradient_norm(parameters):
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    norm = norm**0.5
    return norm


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    if patch_size_t is None:
        # CogVideoX 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        # CogVideoX 1.5
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def reset_memory(device: Union[str, torch.device]) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device: Union[str, torch.device]) -> None:
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory_allocated=:.3f} GB")
    print(f"{max_memory_allocated=:.3f} GB")
    print(f"{max_memory_reserved=:.3f} GB")


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def load_frames_as_tensor(trajectory_maps_path, num_frames, height, width):
    # 获取所有的帧文件并排序（假设是png或jpg格式）
    frame_names = sorted([f for f in os.listdir(trajectory_maps_path) if f.endswith(('.png', '.jpg'))])[:num_frames]
    # indices = np.linspace(0, len(frame_names) - 1, num_frames, dtype=int)
    # frame_names = [frame_names[i] for i in indices]
    assert len(frame_names) == num_frames
    
    # 读取图像并转换为Tensor，同时提取bounding box坐标
    frames = []
    for frame_file in frame_names:
        frame_path = os.path.join(trajectory_maps_path, frame_file)
        image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # [C, H, W]
        frames.append(image_tensor)
    
    # 将帧列表堆叠成 (T, C, H, W) 的Tensor
    frames_resized = torch.stack([resize(frame, (height, width)) for frame in frames], dim=0)

    return frames_resized

def save_hidden_states_as_images(hidden_states_text, save_dir, T, H, W):
    from sklearn.decomposition import PCA
    # 确保保存路径存在
    os.makedirs(save_dir, exist_ok=True)

    # 1. 确保输入是 [B, L, C]，其中 L = T * H * W
    B, L, C = hidden_states_text.shape
    assert L == T * H * W, "L 必须等于 T * H * W"
    
    # 2. 使用 PCA 将 C 降维到 3
    for b in range(B):
        # 将当前 batch 转换为 numpy 数组
        data = hidden_states_text[b].detach().float().cpu().numpy()  # [L, C]
        
        # 应用 PCA 降维到 3
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(data)  # [L, 3]
        
        # 3. 重塑为 [T, H, W, 3]
        frames = reduced_data.reshape(T, H, W, 3)

        # 4. 归一化到 [0, 255]
        frames = (frames - frames.min()) / (frames.max() - frames.min() + 1e-8) * 255
        frames = frames.astype(np.uint8)

        # 5. 保存每一帧图片
        for t in range(T):
            frame = frames[t]
            save_path = os.path.join(save_dir, f"batch_{b}_frame_{t}.png")
            cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"Saved: {save_path}")

def save_tensor_as_images_with_pca(tensor, save_dir):
    from sklearn.decomposition import PCA
    from PIL import Image

    # 确保输出目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 检查输入形状
    assert len(tensor.shape) == 5, "输入张量的形状必须为 [B, T, C, H, W]"
    B, T, C, H, W = tensor.shape


    # 展平成 [B * T * H * W, C] 用于 PCA
    tensor_reshaped = tensor.permute(0, 1, 3, 4, 2).reshape(-1, C)  # [B * T * H * W, C]
    tensor_reshaped = tensor_reshaped.detach().float().cpu().numpy()

    # 应用 PCA 将 C 降维到 3
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(tensor_reshaped)  # [B * T * H * W, 3]

    # 重塑回原始尺寸 [B, T, H, W, 3]
    reduced = reduced.reshape(B, T, H, W, 3)

    # 遍历批次和时间帧保存图片
    for b in range(B):
        for t in range(T):
            # 提取第 b 批次，第 t 帧
            img = reduced[b, t]  # [H, W, 3]

            # 将 PCA 值缩放到 0-255 范围并转换为 uint8
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
            img = img.astype('uint8')

            # 保存图片
            img = Image.fromarray(img)
            img.save(os.path.join(save_dir, f"batch{b}_frame{t}.png"))

def save_tensor_as_video(tensor, output_path, fps=24):
    """
    Save a tensor as a video file.

    Args:
        tensor (torch.Tensor): Input tensor of shape [T, C, H, W] with values normalized between [-1, 1].
        output_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """
    # Step 1: 反归一化，将 [-1, 1] 转换为 [0, 1]
    tensor = (tensor * 0.5 + 0.5)  # 反归一化

    # Step 2: 将 [T, C, H, W] 转换为 [T, H, W, C]，并转换为 uint8 格式
    tensor = (tensor.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    # Step 3: 获取视频的高度、宽度和通道
    T, H, W, C = tensor.shape

    # Step 4: 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Step 5: 将每一帧写入视频
    for frame in tensor:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 将 RGB 转换为 BGR 格式
        video_writer.write(frame_bgr)

    # Step 6: 释放 VideoWriter
    video_writer.release()
    print(f"Video saved at {output_path}")

def save_images(pil_images, save_dir, prefix='image'):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 遍历每张图片并保存
    for idx, img in enumerate(pil_images):
        # 构建文件名
        file_path = os.path.join(save_dir, f"{prefix}_{idx:04d}.png")
        # 保存图片
        img.save(file_path)

    print(f"Images saved to {save_dir}")

def load_video(
    video: str,
    convert_method: Optional[Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]]] = None,
) -> List[PIL.Image.Image]:
    """
    Loads `video` to a list of PIL Image.

    Args:
        video (`str`):
            A URL or Path to a video to convert to a list of PIL Image format.
        convert_method (Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]], *optional*):
            A conversion method to apply to the video after loading it. When set to `None` the images will be converted
            to "RGB".

    Returns:
        `List[PIL.Image.Image]`:
            The video as a list of PIL images.
    """
    is_url = video.startswith("http://") or video.startswith("https://")
    is_file = os.path.isfile(video)
    was_tempfile_created = False

    if not (is_url or is_file):
        raise ValueError(
            f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {video} is not a valid path."
        )

    if is_url:
        response = requests.get(video, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Failed to download video. Status code: {response.status_code}")

        parsed_url = urlparse(video)
        file_name = os.path.basename(unquote(parsed_url.path))

        suffix = os.path.splitext(file_name)[1] or ".mp4"
        video_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name

        was_tempfile_created = True

        video_data = response.iter_content(chunk_size=8192)
        with open(video_path, "wb") as f:
            for chunk in video_data:
                f.write(chunk)

        video = video_path

    pil_images = []
    if video.endswith(".gif"):
        gif = PIL.Image.open(video)
        try:
            while True:
                pil_images.append(gif.copy())
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

    else:
        if is_imageio_available():
            import imageio
        else:
            raise ImportError(BACKENDS_MAPPING["imageio"][1].format("load_video"))

        try:
            imageio.plugins.ffmpeg.get_exe()
        except AttributeError:
            raise AttributeError(
                "`Unable to find an ffmpeg installation on your machine. Please install via `pip install imageio-ffmpeg"
            )

        with imageio.get_reader(video) as reader:
            # Read all frames
            for frame in reader:
                pil_images.append(PIL.Image.fromarray(frame))

    if was_tempfile_created:
        os.remove(video_path)

    if convert_method is not None:
        pil_images = convert_method(pil_images)

    return pil_images

def generate_gaussian_noise(height, width, mean=0, std_dev=1):
    """
    生成一个 H x W 的从高斯分布采样得到的噪声矩阵，并输出其均值和方差。
    
    Args:
        height (int): 矩阵的高度 H。
        width (int): 矩阵的宽度 W。
        mean (float): 高斯分布的均值，默认为 0。
        std_dev (float): 高斯分布的标准差，默认为 1。
    
    Returns:
        np.ndarray: 生成的 H x W 噪声矩阵。
        float: 噪声矩阵的均值。
        float: 噪声矩阵的方差。
    """
    # 从高斯分布采样生成噪声矩阵
    noise = np.random.normal(loc=mean, scale=std_dev, size=(height, width))
    
    # 计算均值和方差
    noise_mean = np.mean(noise)
    noise_variance = np.var(noise)
    
    return noise, noise_mean, noise_variance

def load_frames_as_tensor(trajectory_maps_path, num_frames):
    # 获取所有的帧文件并排序（假设是png或jpg格式）
    frame_names = sorted([f for f in os.listdir(trajectory_maps_path) if f.endswith(('.png', '.jpg'))])#[:num_frames]
    indices = np.linspace(0, len(frame_names) - 1, num_frames, dtype=int)
    frame_names = [frame_names[i] for i in indices]
    assert len(frame_names) == num_frames
    
    # 读取第一帧以提取唯一颜色
    first_frame_path = os.path.join(trajectory_maps_path, frame_names[0])
    first_image = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    df = pd.DataFrame(first_image.reshape(-1, 3), columns=['R', 'G', 'B'])
    unique_colors_df = df.drop_duplicates()
    unique_colors = unique_colors_df.to_numpy()
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]  # 排除黑色
    

    bounding_boxes = []
    for frame_file in frame_names:
        frame_path = os.path.join(trajectory_maps_path, frame_file)
        image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 提取bounding box坐标
        height, width, _ = image.shape
        frame_bounding_boxes = []
        for color in unique_colors:
            mask = cv2.inRange(image, np.array(color), np.array(color))
            coords = np.column_stack(np.where(mask))
            if coords.size > 0:
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                # 归一化坐标
                min_y, min_x = min_y / height, min_x / width
                max_y, max_x = max_y / height, max_x / width
                frame_bounding_boxes.append((min_x, min_y, max_x, max_y))
            else:
                frame_bounding_boxes.append(None)  # 如果没有找到该颜色的bounding box，添加None
                # assert 0
        bounding_boxes.append(frame_bounding_boxes)

    processed_bounding_boxes = []

    # 前 9 帧选 3 帧
    step_front = len(bounding_boxes[:9]) // 3  # 均匀间隔选帧
    for i in range(0, len(bounding_boxes[:9]), step_front):
        processed_bounding_boxes.append(bounding_boxes[i])

    # 后面的帧每 8 帧选 2 帧
    for start in range(9, len(bounding_boxes), 8):  # 从第10帧开始每8帧处理一次
        for i in range(2):  # 选 2 帧
            if start + i < len(bounding_boxes):  # 防止越界
                processed_bounding_boxes.append(bounding_boxes[start + i])

    return processed_bounding_boxes, bounding_boxes[0]

if __name__ == "__main__":
    H, W = 100, 100
    noise, mean, variance = generate_gaussian_noise(H, W, mean=0, std_dev=1)

    print(f"Generated noise shape: {noise.shape}")
    print(f"Mean: {mean}, Variance: {variance}")