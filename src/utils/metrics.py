import torchmetrics

metrics = torchmetrics.MetricCollection(
    [
        torchmetrics.Accuracy(task="binary"),
        torchmetrics.Precision(task="binary"),
        torchmetrics.Recall(task="binary"),
        torchmetrics.F1Score(task="binary"),
        torchmetrics.Specificity(task="binary"),
        torchmetrics.AUROC(task="binary")
    ]
)
