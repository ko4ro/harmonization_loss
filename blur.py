import torch
import torch.nn.functional as F
import numpy as np

BLUR_KERNEL_SIZE = 10
BLUR_SIGMA = 10

def gaussian_kernel(size=BLUR_KERNEL_SIZE, sigma=BLUR_SIGMA):
    """
    Generates a 2D Gaussian kernel.

    Parameters
    ----------
    size : int, optional
        Kernel size, by default BLUR_KERNEL_SIZE
    sigma : int, optional
        Kernel sigma, by default BLUR_SIGMA

    Returns
    -------
    kernel : torch.Tensor
        A Gaussian kernel of shape (1, 1, size, size).
    """
    x_range = torch.arange(-(size-1)//2, (size-1)//2 + 1, dtype=torch.float32)
    y_range = torch.arange((size-1)//2, -(size-1)//2 - 1, -1, dtype=torch.float32)

    xs, ys = torch.meshgrid(x_range, y_range, indexing="ij")
    kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    kernel /= kernel.sum()  # 正規化
    kernel = kernel.view(1, 1, size, size)  # (1, 1, size, size) に reshape

    return kernel

def gaussian_blur(heatmap, kernel):
    """
    Blurs a batch of heatmaps with a Gaussian kernel.

    Parameters
    ----------
    heatmap : torch.Tensor
        The heatmap(s) to blur. Shape: (B, C, H, W), (C, H, W), (H, W), or (1, H, W)
    kernel : torch.Tensor
        The Gaussian kernel. Shape: (1, 1, K, K)

    Returns
    -------
    blurred_heatmap : torch.Tensor
        The blurred heatmap(s) with the same shape as input.
    """
    # heatmap の次元を統一
    if heatmap.dim() == 2:  # (H, W) の場合
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) に変換
    elif heatmap.dim() == 3:  # (C, H, W) or (1, H, W) の場合
        heatmap = heatmap.unsqueeze(0)  # (1, C, H, W) に変換
    elif heatmap.dim() == 4:  # (B, C, H, W) の場合はそのまま
        pass
    else:
        raise ValueError(f"Unsupported heatmap shape: {heatmap.shape}")

    B, C, H, W = heatmap.shape

    # カーネルを heatmap のデバイスに移動
    kernel = kernel.to(heatmap.device)

    # カーネルをブロードキャスト (B, C, 1, 1) に適用
    kernel = kernel.expand(C, 1, kernel.shape[2], kernel.shape[3])  # (C, 1, K, K)

    # グループ化畳み込みを使用してバッチ処理
    blurred_heatmap = F.conv2d(heatmap, kernel, padding="same", groups=C)

    return blurred_heatmap

# 使用例
# device = "cuda" if torch.cuda.is_available() else "cpu"

# kernel = gaussian_kernel().to(device)
# heatmaps = torch.rand(8, 1, 256, 256).to(device)  # 8枚のランダムなヒートマップ (B=8, C=1, H=256, W=256)

# blurred_heatmaps = gaussian_blur(heatmaps, kernel)

# print(blurred_heatmaps.shape)  # (8, 1, 256, 256)
