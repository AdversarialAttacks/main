import torch
import numpy as np


class DeviceSetup:
    def __init__(self, seed=42):
        """
        Initialize the DeviceSetup class with an optional seed value.

        Args:
            seed (int, optional): Seed value for reproducibility. Defaults to 42.
        """
        self.seed = seed
        self.device = None

    def setup_device(self):
        """
        Set up the device (CPU or GPU) based on availability.

        Additionally, provide information about GPU usage if available.
        """
        # Setting device on GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        print()

        # Additional Info when using cuda
        if self.device.type == "cuda":
            print(torch.cuda.get_device_name(0))
            print("Memory Usage:")
            print(
                "Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB"
            )
            print(
                "Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB"
            )

    def set_seed(self):
        """
        Set the seed value for random number generation for reproducibility.
        """
        # Set the seed value for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def setup(self):
        """
        Perform the complete setup by calling setup_device() and set_seed() methods.
        """
        self.setup_device()
        self.set_seed()
