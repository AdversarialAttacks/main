import os

import torch
import torchvision
import matplotlib.pyplot as plt

from src.data.mri import MRIDataModule
from src.data.covidx import COVIDXDataModule
from src.utils.download import download_models
from src.utils.transform_perturbation import AddImagePerturbation
from src.utils.uap_helper import generate_adversarial_images_from_model_dataset, get_model
from src.models.imageclassifier import ImageClassifier

from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

def get_transform(perturbations: torch.Tensor = None, p: float = None, idx: int = None):
    if None in (perturbations, p, idx):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224), antialias=True),
            ]
        )
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224), antialias=True),
            AddImagePerturbation(perturbations, p, idx),
        ]
    )

def pipeline(
    modelname: str,
    dataset: str,
    n_robustifications: int,
    i: int,
    n: int,
    t: int,
    p: int,
    lambda_norm: int,
    r: float,
    eps: float,
    lr_uap: float,
    seed: int,
    num_workers: int,
    device: str,
    verbose: bool = False,
):
    """
    Runs the pipeline for generating Universal Adversarial Perturbations (UAP) using the specified model and dataset.

    Parameters:
    modelname (str): The name of the model to be used for generating perturbations.
    dataset (str): The dataset to be used for training and evaluating the UAP.
    n_robustifications (int): The number of robustifications to apply.
    i (int): The number of UAPs to generate.
    n (int): The number of images to be used for generating UAPs.
    t (int): The number of retries to fool an image on the algorithm.
    p (int): The norm to be used for measuring perturbations (e.g., L2 norm, L∞ norm).
    lambda_norm (int): The regularization parameter for the norm.
    r (float): The desired fooling rate, which when achieved, saves the UAP.
    eps (float): A small positive constant for numerical stability in the loss function.
    lr_uap (float): The learning rate for the UAP optimization problem.
    seed (int): The random seed for reproducibility.
    num_workers (int): Number of worker threads for data loading.
    device (str): The device to be used for computation (e.g., 'cpu' or 'cuda').
    verbose (bool): Whether to print additional information during the pipeline.

    Returns:
    None
    """
    model, hparams = get_model(modelname=modelname, dataset=dataset, output_size=1, return_hparams=True)

    for current_robustification in range(n_robustifications):
        modelfolder = f"{modelname}-{dataset}-n_{n}-robustification_{current_robustification}"

        # check if the model has already been robustified and evaluated, if so, skip current iteration
        metrics_path_i = f"robustified_models/{modelfolder}/06_eval_robustified_test_uap_{i-1}/metrics.csv"
        metrics_path_next = f"robustified_models/{modelfolder}/06_eval_robustified_test_uap_{i}/metrics.csv"
        if os.path.exists(metrics_path_i) and not os.path.exists(metrics_path_next):
            print(f"Skipping robustification {current_robustification}, since already exists") 

            # load best model checkpoint
            model = ImageClassifier.load_from_checkpoint(
                checkpoint_path=f"robustified_models/{modelname}-{dataset}-n_{n}-robustification_{current_robustification}/04_robustify_model/model.ckpt",
                modelname=modelname,
                output_size=1,
                p_dropout_classifier=hparams["p_dropout_classifier"],
                lr=hparams["lr"],
                weight_decay=hparams["weight_decay"],
            )
            model.freeze()
            model.eval()
            
            continue

        # generate UAP
        loggerUAP = pl_loggers.CSVLogger(
            "robustified_models",
            name=modelfolder,
            version="01_UAPs_pre_robustification",
            flush_logs_every_n_steps=1,
        )
        perturbations = generate_adversarial_images_from_model_dataset(
            model,
            modelname,
            dataset,
            logger=loggerUAP,
            transform=get_transform(),
            i=i,
            n=n,
            r=r,
            p=p,
            lambda_norm=lambda_norm,
            t=t,
            eps=eps,
            lr_uap=lr_uap,
            seed=seed,
            num_workers=num_workers,
            device=device,
            verbose=verbose,
        ).cpu()
        perturbations = perturbations.detach()
        loggerUAP.save()

        # evaluate model on testdata
        loggerEvalUnrobustifiedTest = pl_loggers.CSVLogger(
            "robustified_models",
            name=modelfolder,
            version="02_eval_unrobustified_test",
            flush_logs_every_n_steps=1,
        )
        datamodule = get_datamodule(
            dataset=dataset,
            transform=get_transform(),
            num_workers=num_workers,
            batch_size=32,
            seed=seed,
        )
        trainer = Trainer(
            logger=loggerEvalUnrobustifiedTest,
        )
        trainer.test(model, datamodule.test_dataloader())

        # evaluate model on testdata + uaps
        for perturbation_idx in range(i):
            loggerEvalUnrobustifiedTestUAP = pl_loggers.CSVLogger(
                "robustified_models",
                name=modelfolder,
                version=f"03_eval_unrobustified_test_uap_{perturbation_idx}",
                flush_logs_every_n_steps=1,
            )
            datamodule = get_datamodule(
                dataset=dataset,
                transform=get_transform(perturbations, p=1, idx=perturbation_idx),
                num_workers=num_workers,
                batch_size=32,
                seed=seed,
            )
            trainer = Trainer(
                logger=loggerEvalUnrobustifiedTestUAP,
            )
            trainer.test(model, datamodule.test_dataloader())

        # unfreeze model
        model.unfreeze()
        model.train()

        # robustify model
        loggerRobustify = pl_loggers.CSVLogger(
            "robustified_models",
            name=modelfolder,
            version="04_robustify_model",
            flush_logs_every_n_steps=1,
        )
        datamodule = get_datamodule(
            dataset=dataset,
            transform=get_transform(perturbations, p=0.5),
            num_workers=num_workers,
            batch_size=32,
            seed=seed,
        )
        trainer = Trainer(
            max_epochs=50,
            log_every_n_steps=1,
            gradient_clip_val=0.5,
            accelerator="auto",
            logger=loggerRobustify,
            fast_dev_run=False,  # set to True to test run
            enable_progress_bar=True,
            enable_model_summary=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=20),
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1, 
                    save_last=False,  
                    dirpath=f"robustified_models/{modelname}-{dataset}-n_{n}-robustification_{current_robustification}/04_robustify_model",
                    filename="model",
                )
            ],
        )

        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

        # load best model checkpoint
        model = ImageClassifier.load_from_checkpoint(
            checkpoint_path=f"robustified_models/{modelname}-{dataset}-n_{n}-robustification_{current_robustification}/04_robustify_model/model.ckpt",
            modelname=modelname,
            output_size=1,
            p_dropout_classifier=hparams["p_dropout_classifier"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        model.freeze()
        model.eval()

        # evaluate robustified model on testdata
        loggerEvalRobustifiedTest = pl_loggers.CSVLogger(
            "robustified_models",
            name=modelfolder,
            version="05_eval_robustified_test",
            flush_logs_every_n_steps=1,
        )
        datamodule = get_datamodule(
            dataset=dataset,
            transform=get_transform(perturbations, p=0.5),
            num_workers=num_workers,
            batch_size=32,
            seed=seed,
        )
        trainer = Trainer(
            logger=loggerEvalRobustifiedTest,
        )
        trainer.test(model, datamodule.test_dataloader())

        # evaluate robustified model on testdata + uaps
        for perturbation_idx in range(i):
            loggerEvalRobustifiedTestUAP = pl_loggers.CSVLogger(
                "robustified_models",
                name=modelfolder,
                version=f"06_eval_robustified_test_uap_{perturbation_idx}",
                flush_logs_every_n_steps=1,
            )
            datamodule = get_datamodule(
                dataset=dataset,
                transform=get_transform(perturbations, p=0.5, idx=perturbation_idx),
                num_workers=num_workers,
                batch_size=32,
                seed=seed,
            )
            trainer = Trainer(
                logger=loggerEvalRobustifiedTestUAP,
            )
            trainer.test(model, datamodule.test_dataloader())