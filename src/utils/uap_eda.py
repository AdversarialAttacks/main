import os
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
plt.rcParams["figure.figsize"] = (16, 8)


class UAP_EDA:
    def __init__(self, model, dataset, n_image, max_robustification_level=5):
        self.model = model
        self.dataset = dataset
        self.n_image = n_image
        self.max_robustification_level = max_robustification_level
        self.uaps_tensor = self.return_stacked_uap_tensor(max_robustification=5)

    def _read_perturbations(self, n_robustification):
        uap_path = f"robustified_models/{self.model}-{self.dataset}-n_{self.n_image}-robustification_{n_robustification}/01_UAPs_pre_robustification/UAPs_tensor.pt"
        if os.path.isfile(uap_path):
            # Tensor shape: torch.Size([5, 3, 224, 224]) -> 5 UAPs, 3 channels, 224x224 pixels
            return torch.load(uap_path, map_location=torch.device("cpu")).detach()

        else:
            print(f"File not found at {uap_path}")

    def _read_all_perturbations(self, max_robustification):
        uaps = []
        for n_robustification in range(0, max_robustification):
            uap = self._read_perturbations(n_robustification)
            if uap is not None:
                uaps.append(uap)
            else:
                print(
                    f"Skipping tensor for robustification level {n_robustification} due to missing file."
                )
        return uaps

    def return_stacked_uap_tensor(self, max_robustification):
        uaps = self._read_all_perturbations(max_robustification)
        self.uaps_tensor = torch.stack(uaps)
        # Tensor shape: torch.Size([5, 5, 3, 224, 224]) -> 5 Robustification Levels, 5 UAPs, 3 channels, 224x224 pixels
        return self.uaps_tensor

    def _visualize_uaps_tensor(self):
        for i in range(self.uaps_tensor.shape[0]):
            uap = self.uaps_tensor[i]
            fig, ax = plt.subplots(
                1, self.uaps_tensor.shape[0], figsize=(3 * self.uaps_tensor.shape[0], 3)
            )
            for j in range(self.uaps_tensor.shape[1]):
                perturbations = uap[j].mean(dim=0).cpu().squeeze().numpy().astype(int)
                vmax = np.abs(perturbations).max()
                ax[j].imshow(perturbations, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                ax[j].axis("off")
                ax[j].set_title(f"UAP {j+1}")

            fig.suptitle(
                f"Heatmaps of UAPs for {self.model} on {self.dataset} (n={self.n_image}, Robustification Level {i})"
            )
            plt.show()

    def _visualize_uaps_tensor2(self):
        n_robust_levels = self.uaps_tensor.shape[0]  # Number of robustification levels
        n_uaps = self.uaps_tensor.shape[1]  # Number of UAPs

        for j in range(n_uaps):  # Iterate through each UAP index
            # Create a figure for each UAP with subplots for each robustification level
            fig, axs = plt.subplots(
                1, n_robust_levels, figsize=(3 * n_robust_levels, 3)
            )

            for i in range(
                n_robust_levels
            ):  # Iterate through each robustification level
                uap = self.uaps_tensor[i][j]
                perturbations = uap.mean(dim=0).cpu().squeeze().numpy().astype(int)
                vmax = np.abs(perturbations).max()

                # Access subplot for current robustification level
                ax = axs[i] if n_robust_levels > 1 else axs
                ax.imshow(perturbations, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                ax.axis("off")
                ax.set_title(f"Robustification Level {i+1}")

            # Set the title for the current UAP's figure
            fig.suptitle(
                f"Heatmap of UAP {j+1} for {self.model} on {self.dataset} (n={self.n_image})"
            )
            plt.tight_layout()
            plt.show()

    def visualize_uaps_tensor(self, progress=False):
        if progress:
            self._visualize_uaps_tensor()
        else:
            self._visualize_uaps_tensor2()

    def visualize_uap_violinplot(self, robustification_level=None):
        data = []
        labels = []
        if robustification_level is None:
            # Visualize all robustification levels
            for i in range(self.uaps_tensor.shape[0]):
                uap = self.uaps_tensor[i]
                for j in range(self.uaps_tensor.shape[1]):
                    perturbations = (
                        uap[j].mean(dim=0).cpu().squeeze().numpy().astype(int).flatten()
                    )
                    data.append(perturbations)
                    labels.append(f"UAP {j+1} - Level {i}")
        else:
            # Visualize specific robustification level
            uap = self.uaps_tensor[robustification_level]
            for j in range(self.uaps_tensor.shape[1]):
                perturbations = (
                    uap[j].mean(dim=0).cpu().squeeze().numpy().astype(int).flatten()
                )
                data.append(perturbations)
                labels.append(f"UAP {j+1} - Level {robustification_level}")

        plt.figure(figsize=(12, 6))
        plt.violinplot(data, showmedians=True, showmeans=True)
        plt.xticks(range(1, len(labels) + 1), labels, rotation=90, ha="right")
        plt.ylabel("Perturbation Value")
        if robustification_level is None:
            plt.title(
                f"Violinplot Distribution of UAP Pixel Value Across All Robustification Levels\n"
                f"Model: {self.model} on {self.dataset} - (n={self.n_image})"
            )
        else:
            plt.title(
                f"Violinplot Distribution of UAP Pixel Value for Robustification Level {robustification_level}\n"
                f"Model: {self.model} on {self.dataset} - (n={self.n_image})"
            )
        plt.grid(True)
        plt.show()

    def visualize_multiple_uaps(self, uap_indices, robustification_level):
        uaps = self._read_perturbations(robustification_level)
        num_uaps = len(uap_indices)
        fig, axes = plt.subplots(
            1, num_uaps, figsize=(5 * num_uaps, 5)
        )  # Adjust figure size based on number of UAPs

        for i, idx in enumerate(uap_indices):
            uap = uaps[idx]
            perturbations = uap.mean(dim=0).cpu().squeeze().numpy().astype(int)
            vmax = np.abs(perturbations).max()

            ax = axes[i] if num_uaps > 1 else axes
            ax.imshow(perturbations, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.axis("off")
            ax.set_title(
                f"Heatmap of UAP {idx} for {self.model} on {self.dataset} \n (n={self.n_image}, Robustification Level {robustification_level})"
            )

        plt.tight_layout()
        plt.show()
