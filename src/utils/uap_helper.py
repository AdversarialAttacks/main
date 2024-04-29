import torch
import torchvision
from tqdm.notebook import tqdm, trange
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


def fool_image(model, image, v, norm_p, norm_alpha, max_iter_image, eps, verbose, device):
    bce_f = torch.nn.BCELoss().to(device)
    norm_f = lambda x: torch.functional.norm(input=x, p=norm_p)
    loss_bce_inv_f = lambda y_pred, y_adv: 1 / (bce_f(y_pred, y_adv) + eps)

    # initialize temporary adversarial perturbation
    v_temp = torch.zeros((1, 3, 224, 224), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([v_temp], lr=0.1)

    # iterate till the image is fooled or limit
    for _ in range(max_iter_image):

        # check if the image is fooled, if so, break
        if check_if_image_fooled(model, image, v, v_temp):
            print("Image fooled! Adding perturbation...") if verbose else None
            return v + v_temp

        # if the image is not fooled, update the adversarial perturbation
        optimizer.zero_grad()

        x_adv = image + v + v_temp
        x_adv = torch.clamp(x_adv, 0, 255)

        y_pred = model(image).sigmoid()
        y_adv = model(x_adv).sigmoid()

        loss = loss_bce_inv_f(y_pred, y_adv) + norm_alpha * norm_f(v + v_temp)
        (
            print(f"Loss: {loss}, Norm: {norm_alpha * norm_f(v_temp)}, InvBCE: {loss_bce_inv_f(y_pred, y_adv)}")
            if verbose
            else None
        )

        loss.backward()
        optimizer.step()

    # image could not be fooled, return original perturbation
    return v


def calculate_fooling_rate(model, dataloader, v, num_images, verbose, device):
    n_image = 0
    fooling_rate = 0
    for batch in tqdm(dataloader, desc="Image", position=1, total=num_images):
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
    num_pertubation_images,
    desired_fooling_rate,
    norm_p,
    norm_alpha,
    max_iter_image,
    eps,
    verbose,
    device,
):
    v = torch.zeros((1, 3, 224, 224), device=device, requires_grad=False)

    fooling_rate = 0.0
    epoch = 0
    # iterate, till the fooling rate is reached
    while fooling_rate < desired_fooling_rate:
        epoch += 1
        print(f"Starting epoch {epoch}...") if verbose else None

        # iterate over images
        n_image = 0
        for batch in tqdm(dataloader, desc="Image", position=1, total=num_pertubation_images):

            # break on image limit
            if n_image == num_pertubation_images:
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
                norm_p=norm_p,
                norm_alpha=norm_alpha,
                max_iter_image=max_iter_image,
                eps=eps,
                verbose=verbose,
                device=device,
            )
            n_image += 1

        # calculate fooling rate
        fooling_rate = calculate_fooling_rate(
            model, dataloader=dataloader, v=v, num_images=num_pertubation_images, verbose=verbose, device=device
        )

    return v


def generate_adversarial_images_from_model_dataset(
    modelname,
    dataset,
    num_universal_pertubation_images=50,
    num_pertubation_images=50,
    desired_fooling_rate=0.8,
    norm_p=2,
    norm_alpha=0.001,
    max_iter_image=20,
    seed=None,
    eps=1e-6,
    verbose=False,
    device="cpu",
):
    if seed is None:
        seed = torch.randint(0, int(1e9), (1,)).item()

    model = get_model(modelname, dataset)
    model.freeze()
    model.eval()
    model.to(device)

    perturbations = []
    for i in trange(num_universal_pertubation_images, desc="Universal Pertubation", position=0):
        dataloader = get_datamodule(dataset, seed=seed + i)
        v = generate_adversarial_image(
            model,
            dataloader.train_dataloader(),
            num_pertubation_images=num_pertubation_images,
            desired_fooling_rate=desired_fooling_rate,
            norm_p=norm_p,
            norm_alpha=norm_alpha,
            max_iter_image=max_iter_image,
            eps=eps,
            verbose=verbose,
            device=device,
        )

        perturbations.append(v)

    return torch.stack(perturbations, dim=1).squeeze(0)
