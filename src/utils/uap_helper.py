import time
import torch
import torchvision
from tqdm.notebook import trange
from lightning.pytorch import loggers as pl_loggers
from src.data.mri import MRIDataModule
from src.data.covidx import COVIDXDataModule
from src.models.imageclassifier import ImageClassifier

default_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224), antialias=True),
    ]
)


def get_datamodule(dataset, transform=default_transform, num_workers=0, batch_size=1, seed=42):
    if dataset == "covidx_data":
        return COVIDXDataModule(
            path="data/raw/COVIDX-CXR4",
            transform=transform,
            num_workers=num_workers,
            batch_size=batch_size,
            train_sample_size=0.05,
            train_shuffle=False,
            seed=seed,
        ).setup()

    elif dataset == "mri_data":
        return MRIDataModule(
            path="data/raw/Brain-Tumor-MRI",
            path_processed="data/processed/Brain-Tumor-MRI",
            transform=transform,
            num_workers=num_workers,
            batch_size=batch_size,
            train_shuffle=False,
            seed=seed,
        ).setup()

    else:
        raise ValueError("Invalid dataset")


def get_model(modelname, dataset, output_size=1):
    return ImageClassifier.load_from_checkpoint(
        checkpoint_path=f"models/{modelname}-{dataset}/model.ckpt",
        modelname=modelname,
        output_size=output_size,
        p_dropout_classifier=0.0,
        lr=0.0,
        weight_decay=0.0,
    )


def check_if_image_fooled(model, image, v, v_temp):
    with torch.no_grad():

        x_adv = image + v + v_temp
        x_adv = torch.clamp(x_adv, 0, 255)

        y_pred = model(image.round()).sigmoid()
        y_adv = model(x_adv.round()).sigmoid()

        return y_pred.round() != y_adv.round()


def fool_image(model, image, v, p, lambda_norm, t, eps, verbose, device):
    bce_f = torch.nn.BCELoss().to(device)
    norm_f = lambda x: torch.functional.norm(input=x, p=p)
    loss_bce_inv_f = lambda y_pred, y_adv: 1 / (bce_f(y_pred, y_adv) + eps)

    # initialize temporary adversarial perturbation
    delta_v = torch.zeros((1, 3, 224, 224), device=device, requires_grad=True)
    optim = torch.optim.Adam([delta_v], lr=0.1)

    # iterate till the image is fooled or limit
    for _ in range(t):

        # check if the image is fooled, if so, break
        if check_if_image_fooled(model, image, v, delta_v):
            print("Image fooled! Adding perturbation...") if verbose else None
            return v + delta_v

        # if the image is not fooled, update the adversarial perturbation
        optim.zero_grad()

        x_adv = image + v + delta_v
        x_adv = torch.clamp(x_adv, 0, 255)

        y_pred = model(image).sigmoid()
        y_adv = model(x_adv).sigmoid()

        loss = loss_bce_inv_f(y_pred, y_adv) + lambda_norm * norm_f(v + delta_v)
        (
            print(f"Loss: {loss}, Norm: {lambda_norm * norm_f(delta_v)}, InvBCE: {loss_bce_inv_f(y_pred, y_adv)}")
            if verbose
            else None
        )

        loss.backward()
        optim.step()

    # image could not be fooled, return original perturbation
    return v


def calculate_fooling_rate(model, dataloader, v, num_images, verbose, device):
    n_image = 0
    fooling_rate = 0
    for batch in dataloader:
        if n_image == num_images:
            break

        image, label = batch
        image, label = image.to(device), label.to(device)

        x_adv = image + v
        x_adv = torch.clamp(x_adv, 0, 255)

        y_pred = model(image.round()).sigmoid()
        y_adv = model(x_adv.round()).sigmoid()

        if y_pred.round() != y_adv.round():
            fooling_rate += 1

        n_image += 1

    fooling_rate = fooling_rate / num_images
    print(f"Fooling rate: {fooling_rate}") if verbose else None
    return fooling_rate


def generate_adversarial_image(
    model,
    dataloader,
    logger: pl_loggers.Logger,
    n,
    r,
    p,
    lambda_norm,
    t,
    eps,
    verbose,
    device,
):
    v = torch.zeros((1, 3, 224, 224), device=device, requires_grad=False)

    fooling_rate = 0.0
    step = 0
    # iterate, till the fooling rate is reached
    while fooling_rate < r:
        # measure time
        t0 = time.perf_counter()

        step += 1
        (print(f"Starting step {step} for epoch {logger.current_epoch}...") if verbose else None)

        # iterate over images
        n_image = 0
        for batch in dataloader:

            # break on image limit
            if n_image == n:
                break

            print(f"Image {n_image}...") if verbose else None

            # get image and label and move to device
            image, _ = batch
            image = image.to(device)

            # fool the image
            v = fool_image(
                model,
                image,
                v,
                p=p,
                lambda_norm=lambda_norm,
                t=t,
                eps=eps,
                verbose=verbose,
                device=device,
            )
            n_image += 1

        # calculate fooling rate
        fooling_rate = calculate_fooling_rate(
            model,
            dataloader=dataloader,
            v=v,
            num_images=n,
            verbose=verbose,
            device=device,
        )

        t1 = time.perf_counter()
        time_elapsed = t1 - t0

        logger.log_metrics(
            {
                "fooling_rate": fooling_rate,
                "time_elapsed": time_elapsed,
                "uap": logger.current_epoch,
                "uap_step": step,
            },
            step=logger.current_step,
        )
        logger.current_step += 1

    return v


def generate_adversarial_images_from_model_dataset(
    model,
    modelname,
    dataset,
    logger: pl_loggers.Logger,
    transform=default_transform,
    i=50,
    n=50,
    r=0.8,
    p=2,
    lambda_norm=0.001,
    t=20,
    eps=1e-6,
    seed=None,
    verbose=False,
    num_workers=0,
    device="cpu",
):
    if seed is None:
        seed = torch.randint(0, int(1e9), (1,)).item()

    model.freeze()
    model.eval()
    model.to(device)

    logger.log_hyperparams(
        {
            "modelname": modelname,
            "dataset": dataset,
            "transform": transform,
            "i": i,
            "n": n,
            "r": r,
            "p": p,
            "lambda_norm": lambda_norm,
            "t": t,
            "eps": eps,
            "seed": seed,
            "verbose": verbose,
            "num_workers": num_workers,
            "device": device,
        }
    )

    v = []
    logger.current_step = 0
    for i_iteration in trange(i, desc="Universal Pertubation", position=0):
        t0 = time.perf_counter()
        logger.current_epoch = i_iteration

        dataloader = get_datamodule(dataset, transform=transform, seed=seed + i_iteration, num_workers=num_workers)
        v_i = generate_adversarial_image(
            model,
            dataloader.train_dataloader(),
            logger,
            n=n,
            r=r,
            p=p,
            lambda_norm=lambda_norm,
            t=t,
            eps=eps,
            verbose=verbose,
            device=device,
        )
        v.append(v_i)

        t1 = time.perf_counter()
        time_elapsed = t1 - t0
        logger.log_metrics(
            {
                "uap_time_elapsed": time_elapsed,
            },
            step=logger.current_step - 1,
        )

    v = torch.stack(v, dim=1).squeeze(0)
    torch.save(v, f"{logger.log_dir}/UAPs_tensor.pt")
    return v
