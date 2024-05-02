import os
import torch
import shutil
import torchvision
import pandas as pd
import lightning as L


class MRIDataset(torch.utils.data.Dataset):
    """
    A dataset class for MRI images that supports loading and transforming data.

    Attributes:
        path (str): The base directory of the dataset.
        split (str): The dataset split (e.g., 'train', 'val', 'test').
        transform (callable, optional): Optional transform to be applied on a sample.
        shuffle (bool, optional): Whether to shuffle the dataset.

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
        seed=None,
    ):
        self.path = path
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed

        self.data = pd.read_csv(f"{self.path}/{self.split}.txt", sep=" ", header=None)
        self.data.columns = ["filename", "class", "label"]
        if shuffle:
            self.data = self.data.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets the image and label for a given index.

        Parameters:
            idx (int): The index of the item.

        Returns:
            tuple: A tuple containing the image and its label.
        """
        row = self.data.iloc[idx]
        filename = f"{self.path}/{self.split}/{row['filename']}"
        image = torchvision.io.read_image(
            filename, mode=torchvision.io.image.ImageReadMode.GRAY
        )
        image = image.int().float()
        image = image.expand(3, -1, -1)
        label = row["label"].astype("int")
        if self.transform:
            image = self.transform(image)
        return image, label


class MRIDataModule(L.LightningDataModule):
    """
    A data module for handling MRI dataset operations including setup, and data loaders creation.

    Attributes:
        path (str): Path to the raw dataset.
        path_processed (str): Path where the processed dataset will be stored.
        transform (callable, optional): Transformations to apply on each sample.
        batch_size (int, optional): The size of each data batch.
        train_val_ratio (float, optional): The ratio of training to validation data.
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
        path="data/raw/Brain-Tumor-MRI",
        path_processed="data/processed/Brain-Tumor-MRI",
        transform=None,
        batch_size=32,
        num_workers=0,
        train_val_ratio=0.8,
        train_shuffle=False,
        seed=None,
    ):
        super().__init__()
        self.path = path
        self.path_processed = path_processed
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = num_workers > 0
        self.train_val_ratio = train_val_ratio
        self.train_shuffle = train_shuffle
        self.seed = seed
        if seed:
            torch.manual_seed(seed)

    def setup(self):
        """
        Sets up the dataset by creating train, validation, and test splits.

        Returns:
            MRIDataModule: self
        """
        if not os.path.exists(self.path_processed):
            print("Processing data...")
            self._copy_files()
            self._generate_labels_files()

        self.train_dataset = MRIDataset(
            self.path_processed,
            "train",
            transform=self.transform,
            shuffle=self.train_shuffle,
        )
        self.val_dataset = MRIDataset(
            self.path_processed, "val", transform=self.transform
        )
        self.test_dataset = MRIDataset(
            self.path_processed, "test", transform=self.transform
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

    def _copy_files(self):
        """Copies and organizes files into train, val, and test directories."""
        os.makedirs(f"{self.path_processed}/train", exist_ok=True)
        os.makedirs(f"{self.path_processed}/val", exist_ok=True)
        os.makedirs(f"{self.path_processed}/test", exist_ok=True)

        self._process_files(
            source_path=f"{self.path}/Training",
            dest_path=f"{self.path_processed}/train",
            dest_path_2=f"{self.path_processed}/val",
            is_training=True,
        )

        self._process_files(
            source_path=f"{self.path}/Testing",
            dest_path=f"{self.path_processed}/test",
            is_training=False,
        )

    def _process_files(
        self,
        source_path,
        dest_path,
        dest_path_2=None,
        is_training=True,
    ):
        """
        Processes and splits the files between training and validation sets, or organizes test files.

        Parameters:
            source_path (str): The directory of the source files.
            dest_path (str): The target directory for the main dataset split.
            dest_path_2 (str, optional): The target directory for the secondary dataset split (e.g., validation).
            is_training (bool): Flag indicating whether the processing is for training data.
        """
        for tumor_label in os.listdir(source_path):
            files = os.listdir(f"{source_path}/{tumor_label}")

            if is_training:
                num_val_files = int(len(files) * (1 - self.train_val_ratio))
                indices = torch.randperm(len(files))[:num_val_files]
                val_files = set(files[i] for i in indices)

            for file_path in files:
                new_file_name = f"{tumor_label}_{file_path}"
                file_dest_path = (
                    f"{dest_path_2}/{new_file_name}"
                    if is_training and file_path in val_files
                    else f"{dest_path}/{new_file_name}"
                )
                shutil.copy(f"{source_path}/{tumor_label}/{file_path}", file_dest_path)

    def _generate_labels_files(self):
        """
        Generates label files for train, val, and test datasets.
        """
        for dataset_type in ["train", "val", "test"]:
            path = f"{self.path_processed}/{dataset_type}"
            files = [f for f in os.listdir(path) if f.endswith(".jpg")]

            df = pd.DataFrame(
                {
                    "filename": files,
                    "class": ["_".join(f.split("_")[:1]) for f in files],
                    "label": [0 if "no_" in f else 1 for f in files],
                }
            )

            df.to_csv(f"{path}.txt", sep=" ", index=False, header=False)

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

        n_positive_train = train_labels.get("pituitary", 0) + train_labels.get("glioma", 0) + train_labels.get("meningioma", 0)
        n_positive_val = val_labels.get("pituitary", 0) + val_labels.get("glioma", 0) + val_labels.get("meningioma", 0)
        n_positive_test = test_labels.get("pituitary", 0) + test_labels.get("glioma", 0) + test_labels.get("meningioma", 0)

        return pd.DataFrame(
            {
                "Dataset": ["MRI-Brain-Tumor", "MRI-Brain-Tumor", "MRI-Brain-Tumor"],
                "Partitiontype": ["Train", "Validation", "Test"],
                "n image absolute": [len_train, len_val, len_test],
                "n image relative": [
                    len_train / total,
                    len_val / total,
                    len_test / total,
                ],
                "n Positive class": [
                    n_positive_train,
                    n_positive_val,
                    n_positive_test,
                ],
                "n Negative class": [
                    train_labels.get("no", 0),
                    val_labels.get("no", 0),
                    test_labels.get("no", 0),
                ],
                "Positive Ratio": [
                    n_positive_train / len_train,
                    n_positive_val / len_val,
                    n_positive_test / len_test,
                ],
                "n pituitary": [
                    train_labels.get("pituitary", 0),
                    val_labels.get("pituitary", 0),
                    test_labels.get("pituitary", 0),
                ],
                "n glioma": [
                    train_labels.get("glioma", 0),
                    val_labels.get("glioma", 0),
                    test_labels.get("glioma", 0),
                ],
                "n meningioma": [
                    train_labels.get("meningioma", 0),
                    val_labels.get("meningioma", 0),
                    test_labels.get("meningioma", 0),
                ],
                "n no_tumor": [
                    train_labels.get("no", 0),
                    val_labels.get("no", 0),
                    test_labels.get("no", 0),
                ],
            }
        )
