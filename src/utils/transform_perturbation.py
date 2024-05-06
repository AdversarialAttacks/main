import torch
from torch import Tensor


class AddImagePerturbation(torch.nn.Module):
    """Add perturbation to an image tensor.
    This module adds a perturbation to an image tensor. The perturbation is added
    using the `add_image_perturbation` function from `torch.nn.functional`.
    Args:
        perturbation (torch.Tensor): Perturbation tensor to be added to the image.
    """

    def __init__(self, perturbation):
        super().__init__()
        self.perturbation = perturbation

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (torch.Tensor): Image tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Image tensor with perturbation added.
        """
        return img + self.perturbation

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(perturbation={self.perturbation})"
