import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ExplorativeDataAnalysis:
    def __init__(
        self,
        datamodule,
        image_size,
        batchsize,
        seed=42,
        dataset="covidx_data",
        train_sample_size=0.05,
        shuffle=True,
    ):

        self.datamodule = datamodule
        self.train_sample_size = train_sample_size
        self.dataset = dataset
        self.image_size = image_size
        self.batchsize = batchsize
        self.seed = seed
        self.shuffle = shuffle

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size, antialias=True),
            ]
        )

        if dataset == "covidx_data":
            self.data = self.datamodule(
                transform=self.transform,
                batch_size=self.batchsize,
                train_shuffle=self.shuffle,
                seed=self.seed,
                train_sample_size=self.train_sample_size,
            ).setup()

        elif dataset == "mri_data":
            self.data = self.datamodule(
                transform=self.transform,
                batch_size=self.batchsize,
                train_shuffle=self.shuffle,
                seed=self.seed,
            ).setup()

    def _get_dataloader(self):
        return {
            "train": self.data.train_dataloader(),
            "val": self.data.val_dataloader(),
            "test": self.data.test_dataloader(),
        }

    def get_image_tensors(self, batch_idx, dataloader_type="train"):
        dataloader = self._get_dataloader()[dataloader_type]
        filtered_batches = []
        for idx, batch in enumerate(dataloader):
            if idx in batch_idx:
                image, label = batch
                filtered_batches.append(batch)
                if len(filtered_batches) == len(batch_idx):
                    break
        return filtered_batches

    def show_batches(
        self, suptitle, batch_idx, hist_mode=False, dataloader_type="train"
    ):
        """
        Show batches of images from the dataloader
        Args:
            suptitle: Title of the plot
            batch_idx: List of batch indices to show, if list is empty, prints all batches and their labels
            hist_mode: If True, show histogram of the images
        """
        dataloader = self._get_dataloader()[dataloader_type]
        filtered_batches = []
        for idx, batch in enumerate(dataloader):
            if len(batch_idx) == 0:
                images, labels = batch
                print(f"Batch {idx} - Label: {labels}")

            if idx in batch_idx:
                image, label = batch
                filtered_batches.append(batch)
                if len(filtered_batches) == len(batch_idx):
                    break

        for batch in filtered_batches:
            image, label = batch
            plt.figure(figsize=(20, 20), dpi=200)
            for i in range(image.shape[0]):
                plt.subplot(4, 4, i + 1)
                plt.imshow(image[i].permute(1, 2, 0).int().clip(0, 255), cmap="gray")
                plt.axis("off")
                plt.title(f"Klasse: {label[i].item()}")
                plt.suptitle(suptitle, fontsize=24, y=1)
                plt.tight_layout()
            plt.show()

        if hist_mode:
            for batch in filtered_batches:
                image, label = batch
                plt.figure(figsize=(20, 20), dpi=200)
                for i in range(image.shape[0]):
                    plt.subplot(4, 4, i + 1)
                    plt.hist(image[i].flatten(), bins=100, alpha=0.7, color="blue")
                    plt.title(f"Klasse: {label[i].item()}")
                    plt.xlim([0, 255])
                    plt.tight_layout()
                plt.suptitle(f"{suptitle}-Histogramm", fontsize=24, y=1)
                plt.tight_layout()
                plt.show()

    def channel_distribution(
        self,
        dataset,
        binsize=255,
        density=True,
    ):
        """
        Plotet die Pixelverteilung jeder Partition, indem alle Bilder auf einen Vektor
        abgeflacht werden und die kombinierte Verteilung der Pixelwerte über alle
        Kanäle in Subplots dargestellt wird.
        """

        dataloader_types = ["train", "val", "test"]

        fig, axes = plt.subplots(
            1, len(dataloader_types), figsize=(5 * len(dataloader_types), 5), dpi=200
        )

        for i, dataloader_type in enumerate(dataloader_types):
            dataloader = self._get_dataloader()[dataloader_type]
            all_pixels = []
            image_count = 0

            # Pixelwerte aller Bilder in einer Liste speichern
            for batch in dataloader:
                images, labels = batch
                image_count += images.size(0)  # Anzahl der Bilder im Batch
                # Pixelwerte der Bilder abflachen
                flattened_pixels = images.view(images.shape[0], -1)
                all_pixels.extend(flattened_pixels)

            # Alle Pixelwerte in einem Tensor zusammenführen
            all_pixels = torch.cat(all_pixels, dim=0)

            # Tensor in Numpy Array umwandeln und Histogramm plotten
            axes[i].hist(
                all_pixels.numpy().flatten(),
                bins=binsize,
                alpha=0.7,
                color="blue",
                density=density,
            )

            axes[i].set_title(
                f"{dataloader_type.capitalize()} Datensatz ({image_count} Bilder)",
                fontsize=16,
            )
            axes[i].set_xlabel("Pixel Wert")
            if density:
                axes[i].set_ylabel("Relative Häufigkeit")
            else:
                axes[i].set_ylabel("Absolute Häufigkeit")

        fig.suptitle(
            f"Verteilung der Pixelwerte (Binsize: {binsize}) \n {dataset}",
            fontsize=24,
            y=1,
        )
        plt.tight_layout()
        plt.show()

    def mean_pixel_values(self):
        """
        Berechnet den Mittelwert der Pixelwerte für die Position der Pixelwerte über alle Channels und Bilder
        """
        dataloader_types = ["train", "val", "test"]

        dict_mean_pixel_values_per_dataloader = {}
        image_counts = {}  # Dict, um Anzahl der Bilder pro Partition zu speichern

        for dataloader_type in dataloader_types:
            dataloader = self._get_dataloader()[dataloader_type]

            pixel_sum = torch.zeros(224, 224)
            count_images = 0

            for images, _ in dataloader:
                # images Dimension: [batch_size, 3, 244, 244] -> Werte identisch über alle Channels

                if pixel_sum is None:
                    # Mittelwert über alle Channels
                    mean_over_channels = images.mean(dim=1)
                    # Summiere alle Pixelwerte über alle Bilder im Batch
                    pixel_sum = mean_over_channels.sum(dim=0)
                else:
                    mean_over_channels = images.mean(dim=1)
                    pixel_sum = mean_over_channels.sum(dim=0)

                # Anzahl der Bilder im Batch addieren
                count_images += images.size(0)

            # Speichern des Mittelwertes der Pixelwerte für die Position der Pixelwerte über alle Channels und Bilder
            dict_mean_pixel_values_per_dataloader[dataloader_type] = (
                pixel_sum / count_images
            )

            image_counts[dataloader_type] = count_images

        return dict_mean_pixel_values_per_dataloader, image_counts

    def plot_mean_pixel_heatmaps_together(self):
        """
        Visualizes the mean pixel values by creating heatmaps for each data partition.
        Uses self.mean_pixel_values() to get the mean values and counts of images.
        """

        mean_pixel_values, counts = self.mean_pixel_values()
        num_plots = len(mean_pixel_values)
        fig = make_subplots(
            rows=1,
            cols=num_plots,
            subplot_titles=[
                f"{key.capitalize()} ({counts[key]} Bilder)"
                for key in mean_pixel_values.keys()
            ],
        )

        col_index = 1
        for dataloader_type, pixel_values in mean_pixel_values.items():
            pixel_values_np = pixel_values.numpy()
            fig.add_trace(
                go.Heatmap(
                    z=pixel_values_np,
                    colorscale="Greys",
                    showscale=False,
                ),
                row=1,
                col=col_index,
            )
            col_index += 1

        fig.update_layout(
            title_text=f"Differenzenbilder der Pixelmittelwerte über Datenpartition für {self.dataset} ",
            height=400,
            width=300 * num_plots,
        )

        fig.show()
