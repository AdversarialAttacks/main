import torch
import torchvision
import matplotlib.pyplot as plt


class ExplorativeDataAnalysis:
    def __init__(
        self,
        datamodule,
        image_size,
        batchsize,
        seed=42,
        adversarial_matrix=None,
        dataloader_type="train",
    ):

        self.datamodule = datamodule
        self.dataloader_state = dataloader_type

        self.image_size = image_size
        self.batchsize = batchsize
        self.seed = seed

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size, antialias=True),
            ]
        )

        self.data = datamodule(
            transform=self.transform,
            batch_size=self.batchsize,
            train_shuffle=False,
            seed=self.seed,
        ).setup()

        self.adversarial_matrix = adversarial_matrix

    def _get_dataloader(self):
        dataloaders = {
            "train": self.data.train_dataloader(),
            "val": self.data.val_dataloader(),
            "test": self.data.test_dataloader(),
        }
        if self.dataloader_state not in dataloaders.keys():
            raise ValueError(
                f"Invalid dataloader_type. Choose from {dataloaders.keys()}"
            )
        return dataloaders[self.dataloader_state]

    def _create_adversarial_matrix(self, visualize=False):
        if self.adversarial_matrix is None:
            adversarial_matrix = torch.rand(
                (1, 3, self.image_size[0], self.image_size[1])
            )
            return adversarial_matrix
        return self.adversarial_matrix

    def show_batches(self, suptitle, batch_idx, hist_mode=False):
        """
        Show batches of images from the dataloader
        Args:
            suptitle: Title of the plot
            batch_idx: List of batch indices to show, if list is empty, prints all batches and their labels
            hist_mode: If True, show histogram of the images
        """
        dataloader = self._get_dataloader()
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


class AnalysePerturbation:
    def __init__(self):
        self.adversarial_matrix = None

    def _generate_adversarial_matrix(self, batch_size=16, image_size=(224, 224)):
        if self.adversarial_matrix is None:
            adversarial_matrix = torch.rand(
                (batch_size, 3, image_size[0], image_size[1])
            )
            return adversarial_matrix
        return self.adversarial_matrix

    def visualize(self):
        adversarial_matrix = self._generate_adversarial_matrix()
        plt.figure(figsize=(20, 20))
        for i in range(adversarial_matrix.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(adversarial_matrix[i].squeeze().permute(1, 2, 0).clip(0, 1))
            plt.axis("off")
        plt.show()

    def vis_histogram(self):
        pass

    def compare_perturbation(self):
        pass
