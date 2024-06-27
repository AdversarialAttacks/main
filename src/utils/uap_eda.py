import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from matplotlib.colors import SymLogNorm


from src.utils.uap_helper import get_datamodule

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

    def _visualize_uaps_tensor(self, transform=None):
        for i in range(self.uaps_tensor.shape[0]):
            uap = self.uaps_tensor[i]
            fig, ax = plt.subplots(
                1, self.uaps_tensor.shape[0], figsize=(3 * self.uaps_tensor.shape[0], 3)
            )

            for j in range(self.uaps_tensor.shape[1]):
                perturbations = uap[j].mean(dim=0).cpu().squeeze().numpy().astype(int)
                vmax = np.abs(perturbations).max()
                symlognorm = SymLogNorm(
                    linthresh=0.03, linscale=0.03, vmin=-vmax, vmax=vmax
                )

                if transform == "symlog":
                    ax[j].imshow(
                        perturbations,
                        cmap="coolwarm",
                        norm=symlognorm,
                    )
                    ax[j].axis("off")
                    ax[j].set_title(f"UAP {j+1}")
                else:
                    ax[j].imshow(
                        perturbations,
                        vmin=-vmax,
                        vmax=vmax,
                        cmap="coolwarm",
                    )
                    ax[j].axis("off")
                    ax[j].set_title(f"UAP {j+1}")

            transform_name = transform if transform else "None"
            fig.suptitle(
                f"Heatmaps of UAPs for {self.model} on {self.dataset} (n={self.n_image}, Robustification Level {i}, Transformation: {transform_name})"
            )
            plt.show()

    def _visualize_uaps_tensor2(self, transform=None):
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

                symlognorm = SymLogNorm(
                    linthresh=0.03, linscale=0.03, vmin=-vmax, vmax=vmax
                )

                # Access subplot for current robustification level
                ax = axs[i] if n_robust_levels > 1 else axs

                if transform == "symlog":
                    ax.imshow(
                        perturbations,
                        cmap="coolwarm",
                        norm=symlognorm,
                    )
                    ax.axis("off")
                    ax.set_title(f"Robustification Level {i+1}")
                else:
                    ax.imshow(
                        perturbations,
                        vmin=-vmax,
                        vmax=vmax,
                        cmap="coolwarm",
                    )
                    ax.axis("off")
                    ax.set_title(f"Robustification Level {i+1}")

            transform_name = transform if transform else "None"
            # Set the title for the current UAP's figure
            fig.suptitle(
                f"Heatmap of UAP {j+1} for {self.model} on {self.dataset} (n={self.n_image}), Transformation: {transform_name}"
            )
            plt.tight_layout()
            plt.show()

    def visualize_uaps_tensor(self, progress=False, transform=None):
        if progress:
            self._visualize_uaps_tensor(transform=transform)
        else:
            self._visualize_uaps_tensor2(transform=transform)

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

    def visualize_multiple_uaps(
        self, uap_indices, robustification_level, transform=None
    ):
        uaps = self._read_perturbations(robustification_level)
        num_uaps = len(uap_indices)
        fig, axes = plt.subplots(
            1, num_uaps, figsize=(5 * num_uaps, 5)
        )  # Adjust figure size based on number of UAPs

        for i, idx in enumerate(uap_indices):
            uap = uaps[idx]
            perturbations = uap.mean(dim=0).cpu().squeeze().numpy().astype(int)
            vmax = np.abs(perturbations).max()

            symlognorm = SymLogNorm(
                linthresh=0.03, linscale=0.03, vmin=-vmax, vmax=vmax
            )

            ax = axes[i] if num_uaps > 1 else axes

            if transform == "symlog":
                im = ax.imshow(
                    perturbations,
                    cmap="coolwarm",
                    norm=symlognorm,
                )
            else:
                im = ax.imshow(
                    perturbations,
                    vmin=-vmax,
                    vmax=vmax,
                    cmap="coolwarm",
                )
            ax.axis("off")
            ax.set_title(
                f"Heatmap of UAP {idx} for {self.model} on {self.dataset} \n (n={self.n_image}, Robustification Level {robustification_level}), Transformation: {transform}"
            )

        plt.tight_layout()
        plt.show()

    def get_image(self, datapartition="train", index=0, seed=42, plot=False):
        datamodule = get_datamodule(self.dataset, seed=seed)
        datamodule.setup()

        if datapartition == "train":
            dataloader = datamodule.train_dataloader()
        elif datapartition == "val":
            dataloader = datamodule.val_dataloader()
        elif datapartition == "test":
            dataloader = datamodule.test_dataloader()
        else:
            raise ValueError(
                "Invalid data partition. Choose from 'train', 'val', or 'test'."
            )

        for i, batch in enumerate(dataloader):
            if i == index:
                images, _ = batch
                if plot:
                    plt.imshow(images[0].permute(1, 2, 0).numpy().astype(int))
                    plt.axis("off")
                    plt.title(f"Image {index} from {datapartition} set")
                    plt.show()

                return images  # Return the image in the format [batch, channel, height, width]

        raise IndexError("Index out of range in the dataset")

    def get_perturbation(self, uap_index, robustification_level):
        return self.uaps_tensor[robustification_level][uap_index].cpu()

    def visualize_image_uap_3D(
        self,
        uap_index=0,
        robustification_level=0,
        datapartition="train",
        image_index=0,
        seed=42,
    ):
        image = self.get_image(
            datapartition=datapartition, index=image_index, seed=seed
        )[0]
        uap = self.get_perturbation(uap_index, robustification_level)

        uap = uap.mean(dim=0).numpy()  # Average over the channels
        image = image.mean(dim=0).numpy()  # Average over the channels

        perturbed_image = image + uap
        perturbed_image = perturbed_image.clip(0, 255)

        fig = go.Figure(data=[go.Surface(z=perturbed_image)])

        fig.update_layout(
            title=f"3D Plot of Image with UAP {uap_index} for {self.model} on {self.dataset} (n={self.n_image}, Robustification Level {robustification_level})",
            scene=dict(
                xaxis_title="Width",
                yaxis_title="Height",
                zaxis_title="Pixel Value",
                zaxis=dict(
                    range=[-50, 255 * 2]
                ),  # Assuming you want to clamp between 0 and 255
            ),
            autosize=True,
            width=800,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90),
        )

        fig.show()

    def visualize_uap_with_data(
        self,
        uap_indices=[0],
        robustification_level=0,
        image=None,
    ):

        # Setup data module
        image = self.get_image(datapartition="train", index=5, seed=42)

        for uap_index in uap_indices:
            v = self.uaps_tensor[robustification_level][uap_index].cpu()

            # Prepare subplots for the current UAP index
            fig, axs = plt.subplots(1, 5, figsize=(25, 5))

            # Original Image
            original_image = image.cpu().squeeze().permute(1, 2, 0).numpy().astype(int)
            axs[0].imshow(original_image)
            axs[0].axis("off")
            axs[0].set_title("Ursprungs Bild")

            # Original Image + UAP
            perturbed_image = image + v
            perturbed_image = perturbed_image.clamp(0, 255)
            perturbed_image = (
                perturbed_image.cpu().squeeze().permute(1, 2, 0).numpy().astype(int)
            )
            axs[1].imshow(perturbed_image)
            axs[1].axis("off")
            axs[1].set_title("Perturbiertes Bild")

            # UAP Visualization
            perturbations = v.mean(dim=0).cpu().squeeze().numpy().astype(int)
            vmax = np.abs(perturbations).max()
            im = axs[2].imshow(perturbations, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            axs[2].axis("off")
            axs[2].set_title("Universal Adversarial Perturbation")
            cbar = plt.colorbar(
                im, ax=axs[2], fraction=0.046, pad=0.01, location="bottom"
            )
            cbar.set_label("Anpassende Helligkeit", fontsize=12)

            # UAP Violinplot
            axs[3].violinplot(v.flatten(), showmedians=True, showmeans=True)
            axs[3].set_title("UAP Violinplot", fontsize=16)
            axs[3].set_ylabel("Perturbations Wert", fontsize=12)

            # UAP Stats with hist plot
            axs[4].hist(v.flatten(), bins=100, color="grey", alpha=0.7)
            axs[4].set_title("UAP Histogramm", fontsize=16)
            axs[4].set_yticks([])
            axs[4].axis("on")
            axs[4].set_xlabel("Perturbations Wert", fontsize=12)
            axs[4].set_ylabel("Häufigkeit", fontsize=12)

            # Add overall title with statistics for each UAP
            plt.suptitle(
                f"- Pixel Perturbation stats: min: {v.min().item():.2f}, max: {v.max().item():.2f}, σ: {v.std().item():.2f}, μ: {v.mean().item():.2f}",
                fontsize=10,
                ha="left",
                y=0.075,
                x=0.145,
            )

        plt.tight_layout()
        plt.show()

    def visualize_uap_3d(self, uap_index=0, robustification_level=0):
        uap = self.uaps_tensor[robustification_level][uap_index].cpu()
        perturbations = uap.mean(dim=0).numpy()

        fig = go.Figure(data=[go.Surface(z=perturbations)])

        fig.update_layout(
            title=f"3D Plot of UAP {uap_index} for {self.model} on {self.dataset} (n={self.n_image}, Robustification Level {robustification_level})",
            scene=dict(
                xaxis_title="Width",
                yaxis_title="Height",
                zaxis_title="Perturbation Value",
            ),
            autosize=True,
            width=800,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90),
        )

        fig.show()
