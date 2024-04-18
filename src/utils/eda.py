import torch
import torchvision
import matplotlib.pyplot as plt


class ExplorativeDataAnalysis:
    def __init__(
        self,
        datamodule,
        image_size,
        batchsize,
        adversarial_matrix=None,
        dataloader_type="train",
    ):

        self.datamodule = datamodule
        self.dataloader_state = dataloader_type

        self.image_size = image_size
        self.batchsize = batchsize
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size, antialias=True),
            ]
        )

        self.data = datamodule(
            transform=self.transform,
            batch_size=self.batchsize,
            train_shuffle=True,
            seed=42,
        ).setup()

        self.adversarial_matrix = adversarial_matrix

    def _get_dataloader(self):
        dataloaders = {
            "train": self.data.train_dataloader(),
            "val": self.data.val_dataloader(),
            "test": self.data.test_dataloader(),
        }
        if self.dataloader_state not in dataloaders.keys():
            raise ValueError(f"Invalid dataloader_type. Choose from {dataloaders.keys()}")
        return dataloaders[self.dataloader_state]

    def _create_adversarial_matrix(self, visualize=False):
        if self.adversarial_matrix is None:
            adversarial_matrix = torch.rand((1, 3, self.image_size[0], self.image_size[1]))
            return adversarial_matrix
        return self.adversarial_matrix

    def visualize_pertubation(self):
        adversarial_matrix = self._create_adversarial_matrix()
        plt.imshow(adversarial_matrix.squeeze().permute(1, 2, 0).clip(0, 1))
        plt.show()

    def show_batch(self):
        dataloader = self._get_dataloader()
        for batch in dataloader:
            image, label = batch
            plt.figure(figsize=(20, 20))
            for i in range(image.shape[0]):
                plt.subplot(4, 4, i + 1)
                plt.imshow(image[i].permute(1, 2, 0).int().clip(0, 255), cmap="gray")
                plt.axis("off")
                plt.title(label[i].item())
                plt.suptitle("Original Images", fontsize=16, y=1.02)
                plt.tight_layout()
            break

    def show_batch_with_adversarial(self):
        dataloader = self._get_dataloader()
        adversarial_matrix = self._create_adversarial_matrix()
        for batch in dataloader:
            image, label = batch
            image = torch.add(image, adversarial_matrix)
            plt.figure(figsize=(20, 20))
            for i in range(image.shape[0]):
                plt.subplot(4, 4, i + 1)
                plt.imshow(image[i].permute(1, 2, 0).int().clip(0, 255), cmap="gray")
                plt.axis("off")
                plt.title(label[i].item())
                plt.suptitle("Adversarial Noise added to images", fontsize=16, y=1.02)
                plt.tight_layout()
            break

    def compare_images(self, num_images=1, hist_mode=False):
        dataloader = self._get_dataloader()
        for batch in dataloader:
            images, labels = batch
            break

        plt.figure(figsize=(10, 5 * num_images))
        for i in range(num_images):
            image, label = images[i], labels[i]
            adversarial_matrix = self._create_adversarial_matrix()
            adversarial_image = torch.add(image, adversarial_matrix)

            if hist_mode:
                plt.subplot(num_images, 2, 2 * i + 1)
                plt.hist(image[0].flatten(), bins=100, color="blue", alpha=0.7)
                plt.title("Histogram of Original Image")

                plt.subplot(num_images, 2, 2 * i + 2)
                plt.hist(adversarial_image[0].flatten(), bins=100, color="blue", alpha=0.7)
                plt.title("Histogram of Adversarial Image")
                plt.suptitle("Original Images vs Adversarial Images", fontsize=16, y=1.01)
                plt.tight_layout()
                continue
            else:
                plt.subplot(num_images, 2, 2 * i + 1)
                plt.imshow(image.permute(1, 2, 0).clip(0, 1), cmap="gray")
                plt.axis("off")
                plt.title(f"Original Image - label {label.item()}")

                plt.subplot(num_images, 2, 2 * i + 2)
                plt.imshow(adversarial_image[0].permute(1, 2, 0).clip(0, 1), cmap="gray")
                plt.axis("off")
                plt.title(f"Adversarial Image - label {label.item()}")
                plt.suptitle("Original Images vs Adversarial Images", fontsize=16, y=1.01)
                plt.tight_layout()
                continue

    def compare_histograms(self):
        dataloader = self._get_dataloader()
        for batch in dataloader:
            images, labels = batch
            break
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))

        image, label = images[0], labels[0]
        adversarial_matrix = self._create_adversarial_matrix()
        adversarial_image = torch.add(image, adversarial_matrix)

        axs[0, 0].imshow(image.permute(1, 2, 0).int().clip(0, 255), cmap="gray")
        axs[0, 0].axis("off")
        axs[0, 0].set_title(f"Original Image - label {label.item()}")

        axs[0, 1].imshow(adversarial_image[0].permute(1, 2, 0).clip(0, 1), cmap="gray")
        axs[0, 1].axis("off")
        axs[0, 1].set_title(f"Adversarial Image - label {label.item()}")

        axs[1, 0].hist(image[0].flatten(), bins=100, color="blue", alpha=0.7)
        axs[1, 0].set_title("Histogram of Original Image")

        axs[1, 1].hist(adversarial_image[0].flatten(), bins=100, color="blue", alpha=0.7)
        axs[1, 1].set_title("Histogram of Adversarial Image")

        plt.suptitle("Original Images vs Adversarial Images", fontsize=16, y=1.01)
        plt.tight_layout()

        plt.show()


class AnalysePerturbation:
    def __init__(self):
        self.adversarial_matrix = None
        self

    def _generate_adversarial_matrix(self, batch_size=16, image_size=(224, 224)):
        if self.adversarial_matrix is None:
            adversarial_matrix = torch.rand((batch_size, 3, image_size[0], image_size[1]))
            return adversarial_matrix
        return self.adversarial_matrix

    def visualize(self):
        # Visualize for each image in the batch the adversarial matrix in a nx4 grid where n is the batch siz
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
