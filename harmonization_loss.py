import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _binomial_kernel(num_channels):
    """
    Create a binomial kernel for downsampling.

    Parameters
    ----------
    num_channels : int
        Number of channels in the image.

    Returns
    -------
    torch.Tensor
        Binomial kernel for downsampling.
    """
    kernel = np.array([1., 4., 6., 4., 1.], dtype=np.float32)
    kernel = np.outer(kernel, kernel)  # Create a 2D kernel
    kernel /= np.sum(kernel)  # Normalize
    kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 5, 5)
    kernel = kernel.repeat(num_channels, 1, 1, 1)  # Shape: (num_channels, 1, 5, 5)

    return kernel


def _downsample(image, kernel):
    """
    Downsample an image using a convolution with a binomial kernel.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (batch_size, num_channels, height, width).
    kernel : torch.Tensor
        Binomial kernel for downsampling.

    Returns
    -------
    torch.Tensor
        Downsampled image.
    """
    num_channels = image.shape[1]
    kernel = kernel.to(image.device)  # Ensure the kernel is on the same device as the image
    return F.conv2d(image, kernel, stride=2, padding=2, groups=num_channels)


def pyramidal_representation(image, num_levels):
    """
    Compute the pyramidal representation of an image.

    Parameters
    ----------
    image : torch.Tensor
        The image to compute the pyramidal representation.
        Shape: (batch_size, num_channels, height, width)
    num_levels : int
        The number of levels to use in the pyramid.

    Returns
    -------
    list of torch.Tensor
        The pyramidal representation.
    """
    num_channels = image.shape[1]
    kernel = _binomial_kernel(num_channels).to(image.device)  # Ensure the kernel is on the same device

    levels = [image]
    for _ in range(num_levels):
        image = _downsample(image, kernel)
        levels.append(image)

    return levels

def standardize_cut(heatmaps, axes=(1, 2), epsilon=1e-5):
    """
    Standardize the heatmaps (zero mean, unit variance) and apply ReLU.
    """
    means = torch.mean(heatmaps, dim=axes, keepdim=True)
    stds = torch.std(heatmaps, dim=axes, keepdim=True)
    heatmaps = (heatmaps - means) / (stds + epsilon)
    return F.relu(heatmaps)

def _mse_with_tokens(heatmaps_a, heatmaps_b, tokens):
    """
    Compute the MSE between two sets of heatmaps, weighted by the tokens.
    """
    return torch.mean((heatmaps_a - heatmaps_b) ** 2 * tokens[:, None, None, None])

def pyramidal_mse_with_tokens(true_heatmaps, predicted_heatmaps, tokens, nb_levels=5):
    """
    Compute mean squared error between two sets of heatmaps on a pyramidal representation.
    """
    pyramid_y = pyramidal_representation(true_heatmaps.unsqueeze(-1), nb_levels)
    pyramid_y_pred = pyramidal_representation(predicted_heatmaps.unsqueeze(-1), nb_levels)
    loss = torch.mean(torch.stack([
        _mse_with_tokens(pyramid_y[i], pyramid_y_pred[i], tokens)
        for i in range(nb_levels)
    ]))
    return loss

def harmonizer_loss(model, images, tokens, labels, true_heatmaps,
                    cross_entropy=nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none'),
                    lambda_weights=1e-5, lambda_harmonization=1.0):
    """
    Compute the harmonization loss: cross entropy + pyramidal mse of standardized-cut heatmaps.
    """
    images.requires_grad = True

    y_pred = model(images)
    loss_metapred = torch.sum(y_pred * labels, dim=-1)
    grads = torch.autograd.grad(loss_metapred, images, grad_outputs=torch.ones_like(loss_metapred), retain_graph=True)[0]
    sa_maps = torch.mean(grads, dim=1)  # Average over channels

    sa_maps_preprocess = standardize_cut(sa_maps)
    heatmaps_preprocess = standardize_cut(true_heatmaps)

    _hm_max = torch.max(heatmaps_preprocess, dim=(1, 2), keepdim=True)[0] + 1e-6
    _sa_max = torch.max(sa_maps_preprocess.detach(), dim=(1, 2), keepdim=True)[0] + 1e-6
    heatmaps_preprocess = heatmaps_preprocess / _hm_max * _sa_max

    harmonization_loss = pyramidal_mse_with_tokens(sa_maps_preprocess, heatmaps_preprocess, tokens)
    cce_loss = torch.mean(cross_entropy(y_pred, labels))

    weight_loss = sum(torch.norm(param, p=2) for name, param in model.named_parameters()
                      if not any(exclude in name for exclude in ['bn', 'normalization', 'embed', 'Norm', 'norm', 'class_token']))

    loss = cce_loss + lambda_weights * weight_loss + lambda_harmonization * harmonization_loss
    gradients = torch.autograd.grad(loss, model.parameters())
    assert len(gradients) == len(list(model.parameters())), "Mismatch between gradients and model parameters"
    return gradients

def loss_norm(loss: torch.Tensor) -> torch.Tensor:
    return loss / loss.item()
