import torch
import torchvision
import torch.optim as optim
import lightning as L

from src.utils.metrics import metrics


class ImageClassifier(L.LightningModule):
    def __init__(
        self, modelname, output_size, p_dropout_classifier, lr=0.01, weight_decay=0
    ):
        super().__init__()
        self.modelname = modelname
        self.output_size = output_size
        self.p_dropout_classifier = p_dropout_classifier
        self.resize = None
        self.lr = lr
        self.weight_decay = weight_decay

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
                param.requires_grad = True

            # replace the last classifier layer on alexnet
            if modelname.startswith("alexnet"):
                self.model.classifier[-1] = torch.nn.Sequential(
                    torch.nn.Dropout(p=p_dropout_classifier),
                    torch.nn.Linear(self.model.classifier[-1].in_features, output_size),
                )

            # replace the last classifier layer on vgg
            elif modelname.startswith("vgg"):
                self.model.classifier[-2] = torch.nn.Dropout(p=p_dropout_classifier)
                self.model.classifier[-1] = torch.nn.Linear(
                    self.model.classifier[-1].in_features, output_size
                )

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

        except Exception as e:
            raise ValueError(f"Cannot load model {modelname}!") from e

    def forward(self, x):
        if self.resize:
            x = self.resize(x)

        result = self.model(x)
        result = torch.sigmoid(result)
        return result

    def predict(self, x):
        self.eval()
        return self.forward(x)

    def __step(self, batch, state):
        x, y = batch
        y = y.float()
        y_hat = self(x).squeeze(1).float()

        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        self.log(f"{state}_loss", loss)

        self.logging[f"{state}_prediction"].append(y_hat)
        self.logging[f"{state}_target"].append(y)
        return loss

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
        targets = torch.cat(self.logging[f"{state}_target"]).int()
        metrics_dict = self.metrics(predictions, targets)
        for metric, value in metrics_dict.items():
            self.log(f"{state}_{metric}", value)

    def on_train_epoch_end(self):
        self.__on_epoch_end("train")

    def on_validation_epoch_end(self):
        self.__on_epoch_end("val")

    def on_test_epoch_end(self):
        self.__on_epoch_end("test")
