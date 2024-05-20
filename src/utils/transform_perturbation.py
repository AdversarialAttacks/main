import torch
from torch import Tensor


class AddImagePerturbation(torch.nn.Module):
    """Add perturbation to an image tensor.
    This module adds a perturbation to an image tensor. The perturbation is added
    using the `add_image_perturbation` function from `torch.nn.functional`.
    Args:
        perturbation (torch.Tensor): Tensor containing perturbations to be added.
        p (float): Probability of adding the perturbation. Default: 0.5.
        idx (int): Index of the perturbation to be added. If None, a random
            perturbation is selected. Default: None.
    """

    def __init__(self, perturbation: torch.Tensor, p: float = 0.5, idx: int = None):
        super().__init__()
        self.perturbation = perturbation
        self.p = p
        self.idx = idx

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (torch.Tensor): Image tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Image tensor with perturbation added.
        """
        if torch.rand(1) >= self.p:
            return img

        idx = torch.randint(0, self.perturbation.size(0), (1,)) if self.idx is None else self.idx
        return img + self.perturbation[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(perturbation={self.perturbation})"
