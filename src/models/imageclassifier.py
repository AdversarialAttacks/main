import torch
import torchvision
import torch.optim as optim
import lightning as L

from src.utils.metrics import metrics


class ImageClassifier(L.LightningModule):
    def __init__(
        self,
        modelname,
        output_size,
        p_dropout_classifier,
        lr=0.01,
        weight_decay=0,
        first_unfreeze_epoch=0,
        second_unfreeze_epoch=0,
    ):
        super().__init__()
        self.modelname = modelname
        self.output_size = output_size
        self.p_dropout_classifier = p_dropout_classifier
        self.lr = lr
        self.weight_decay = weight_decay
        self.first_unfreeze_epoch = first_unfreeze_epoch
        self.second_unfreeze_epoch = second_unfreeze_epoch

        self.resize = None

        self.metrics = metrics
        self.logging = {
            "train_prediction": [],
            "train_target": [],
            "val_prediction": [],
            "val_target": [],
            "test_prediction": [],
            "test_target": [],
        }

        try:
            # check if model exists
            if modelname not in torchvision.models.list_models():
                raise ValueError(
                    f"Model {modelname} doesn't exist! These are the available torchvision models: {torchvision.models.list_models()}"
                )

            # load the model
            self.model = torchvision.models.get_model(modelname, weights="DEFAULT")

            # unfreeze all the layers (they probably already are xD)
            for param in self.model.parameters():
                param.requires_grad = False

            # replace the last classifier layer on alexnet
            if modelname.startswith("alexnet"):
                self.model.classifier[-1] = torch.nn.Sequential(
                    torch.nn.Dropout(p=p_dropout_classifier),
                    torch.nn.Linear(self.model.classifier[-1].in_features, output_size),
                )

            # replace the last classifier layer on vgg
            elif modelname.startswith("vgg"):
                self.model.classifier[-2] = torch.nn.Dropout(p=p_dropout_classifier)
                self.model.classifier[-1] = torch.nn.Linear(self.model.classifier[-1].in_features, output_size)

            # replace the fc layer on resnet
            elif modelname.startswith("resnet"):
                self.model.fc = torch.nn.Sequential(
                    torch.nn.Dropout(p=p_dropout_classifier),
                    torch.nn.Linear(self.model.fc.in_features, output_size),
                )

            # replace classifier layer on densenet
            elif modelname.startswith("densenet"):
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=p_dropout_classifier),
                    torch.nn.Linear(self.model.classifier.in_features, output_size),
                )

            # replace the classifier layer on EfficientNet
            elif modelname.startswith("efficientnet_v2"):
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=p_dropout_classifier),
                    torch.nn.Linear(self.model.classifier[-1].in_features, output_size),
                )

            # replace the head layer on ViT
            elif modelname.startswith("vit"):
                self.resize = torchvision.transforms.Resize((224, 224), antialias=True)
                self.model.heads.head = torch.nn.Sequential(
                    torch.nn.Dropout(p=p_dropout_classifier),
                    torch.nn.Linear(self.model.heads.head.in_features, output_size),
                )

            else:
                raise ValueError(
                    f"Learning on Model {modelname} not implemented! Please choose between alexnet, vgg, resnet, densenet, efficientnet_v2 or vit."
                )

            self.layers = list(self.model.children())
            self.total_layers = len(self.layers)

            if self.first_unfreeze_epoch == 0:
                self.unfreeze_layers(0.5)
                print("Unfreezing 50% of the layers")

            if self.second_unfreeze_epoch == 0:
                self.unfreeze_layers(1.0)
                print("Unfreezing 100% of the layers")

        except Exception as e:
            raise ValueError(f"Cannot load model {modelname}!") from e

    def unfreeze_layers(self, percentage):
        num_layers_to_unfreeze = int(self.total_layers * percentage)
        for layer in self.layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        if self.resize:
            x = self.resize(x)

        y_hat = self.model(x)
        y_hat = torch.clamp(y_hat, min=-9, max=9)
        return y_hat

    def predict(self, x):
        self.eval()
        y_hat = self.forward(x)
        return torch.sigmoid(y_hat)

    def __step(self, batch, state):
        x, y = batch
        y = y.float()
        y_hat = self(x).squeeze(1).float()

        self.logging[f"{state}_prediction"].append(y_hat)
        self.logging[f"{state}_target"].append(y)

        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, _):
        return self.__step(batch, "train")  # Loss

    def validation_step(self, batch, _):
        return self.__step(batch, "val")  # Loss

    def test_step(self, batch, _):
        return self.__step(batch, "test")  # Loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def __on_epoch_end(self, state):
        predictions = torch.cat(self.logging[f"{state}_prediction"])
        predictions = torch.sigmoid(predictions)
        targets = torch.cat(self.logging[f"{state}_target"])

        loss = torch.nn.functional.binary_cross_entropy(predictions, targets)
        self.log(f"{state}_loss", loss)

        metrics_dict = self.metrics(predictions, targets.int())
        for metric, value in metrics_dict.items():
            self.log(f"{state}_{metric}", value)

        self.logging[f"{state}_prediction"] = []
        self.logging[f"{state}_target"] = []

    def on_train_epoch_end(self):
        self.__on_epoch_end("train")

        if (self.current_epoch - 1) == self.first_unfreeze_epoch:
            self.unfreeze_layers(0.5)
            print("Unfreezing 50% of the layers")

        if (self.current_epoch - 1) == self.second_unfreeze_epoch:
            self.unfreeze_layers(1.0)
            print("Unfreezing 100% of the layers")

    def on_validation_epoch_end(self):
        self.__on_epoch_end("val")

    def on_test_epoch_end(self):
        self.__on_epoch_end("test")
