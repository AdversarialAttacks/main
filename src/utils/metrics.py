import torchmetrics

metrics = torchmetrics.MetricCollection(
    [
        torchmetrics.Accuracy(task="binary", average="macro"),
        torchmetrics.Precision(task="binary", average="macro"),
        torchmetrics.Recall(task="binary", average="macro"),
        torchmetrics.F1Score(task="binary", average="macro"),
        torchmetrics.Specificity(task="binary", average="macro"),
        torchmetrics.AUROC(task="binary", average="macro"),
    ]
)

