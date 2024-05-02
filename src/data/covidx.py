import torch
import torchvision
import pandas as pd
import lightning as L


class COVIDXDataset(torch.utils.data.Dataset):
    """
    A dataset class for COVIDX images that supports loading, shuffling, and transforming data.

    Attributes:
        path (str): The base directory of the dataset.
        split (str): The dataset split (e.g., 'train', 'val', 'test').
        transform (callable, optional): Optional transform to be applied on a sample.
        shuffle (bool, optional): Whether to shuffle the dataset upon initialization.

    Methods:
        __len__: Returns the size of the dataset.
        __getitem__: Provides the ability to access a data point through its index.
    """

    def __init__(
        self,
        path,
        split,
        transform=None,
        shuffle=False,
        sample_size=1,
        seed=None,
    ):
        self.path = path
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.sample_size = sample_size
        self.seed = seed

        self.data = pd.read_csv(f"{self.path}/{self.split}.txt", sep=" ", header=None)
        self.data.columns = ["pid", "filename", "class", "source"]
        if shuffle:
            self.data = self.data.sample(
                frac=self.sample_size, random_state=self.seed
            ).reset_index(drop=True)

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets the image and label for a given index.

        Parameters:
            idx (int): The index of the item.

        Returns:
            tuple: A tuple containing the image and its binary label (1 for 'positive', 0 for 'negative').
        """
        row = self.data.iloc[idx]
        filename = f"{self.path}/{self.split}/{row['filename']}"
        image = torchvision.io.read_image(
            filename, mode=torchvision.io.image.ImageReadMode.GRAY
        )
        image = image.int().float()
        image = image.expand(3, -1, -1)
        label = (row["class"] == "positive") * 1
        if self.transform:
            image = self.transform(image)
        return image, label


class COVIDXDataModule(L.LightningDataModule):
    """
    A data module for handling COVIDX dataset operations including setup, and data loaders creation.

    Attributes:
        path (str): Path to the dataset.
        transform (callable, optional): Transformations to apply on each sample.
        batch_size (int, optional): The size of each data batch.
        train_shuffle (bool, optional): Whether to shuffle the training data.
        seed (int, optional): Seed for reproducibility.

    Methods:
        setup: Prepares datasets for training, validation, and testing.
        train_dataloader: Returns a DataLoader for the training dataset.
        val_dataloader: Returns a DataLoader for the validation dataset.
        test_dataloader: Returns a DataLoader for the test dataset.
    """

    def __init__(
        self,
        path="data/raw/COVIDX-CXR4",
        transform=None,
        batch_size=32,
        num_workers=0,
        train_shuffle=False,
        train_sample_size=1,
        seed=None,
    ):
        super().__init__()
        self.path = path
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = num_workers > 0
        self.train_shuffle = train_shuffle
        self.train_sample_size = train_sample_size
        self.seed = seed
        if seed:
            torch.manual_seed(seed)

    def setup(self):
        """
        Sets up the dataset by creating train, validation, and test splits.

        Returns:
            COVIDXDataModule: self
        """
        self.train_dataset = COVIDXDataset(
            path=self.path,
            split="train",
            transform=self.transform,
            shuffle=self.train_shuffle,
            sample_size=self.train_sample_size,
            seed=self.seed,
        )

        self.val_dataset = COVIDXDataset(
            path=self.path,
            split="val",
            transform=self.transform,
        )

        self.test_dataset = COVIDXDataset(
            path=self.path,
            split="test",
            transform=self.transform,
        )

        return self

    def train_dataloader(self):
        """Returns a DataLoader for the training dataset."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.train_shuffle,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        """Returns a DataLoader for the validation dataset."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        """Returns a DataLoader for the test dataset."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def get_partition_info(self):
        """
        Returns a DataFrame containing information about the dataset partitions.
        """

        total = len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)

        len_train = len(self.train_dataset)
        len_val = len(self.val_dataset)
        len_test = len(self.test_dataset)

        train_labels = self.train_dataset.data["class"].value_counts()
        val_labels = self.val_dataset.data["class"].value_counts()
        test_labels = self.test_dataset.data["class"].value_counts()

        return pd.DataFrame(
            {
                "Dataset": ["COVIDX-CXR4", "COVIDX-CXR4", "COVIDX-CXR4"],
                "Partitiontype": ["Train", "Validation", "Test"],
                "n image absolute": [len_train, len_val, len_test],
                "n image relative": [len_train / total, len_val / total, len_test / total],
                "n Positive class": [
                    train_labels["positive"],
                    val_labels["positive"],
                    test_labels["positive"],
                ],
                "n Negative class": [
                    train_labels["negative"],
                    val_labels["negative"],
                    test_labels["negative"],
                ],
                "Postive Ratio": [
                    train_labels["positive"] / len_train,
                    val_labels["positive"] / len_val,
                    test_labels["positive"] / len_test,
                ],
            }
        )
