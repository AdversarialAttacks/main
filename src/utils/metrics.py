import torch
import torchmetrics
from matplotlib.pyplot import plt

metrics = torchmetrics.MetricCollection(
    [
        torchmetrics.Accuracy(task="binary", average="macro"),
        torchmetrics.Precision(task="binary", average="macro"),
        torchmetrics.Recall(task="binary", average="macro"),
        torchmetrics.F1Score(task="binary", average="macro"),
        torchmetrics.Specificity(task="binary", average="macro"),
        torchmetrics.AUROC(task="binary"),
    ]
)


class Metrics:
    def __init__(
        self,
        y_trues,
        y_logits,
        metrics=metrics,
    ):
        self.metrics = metrics
        self.y_trues = y_trues
        self.y_logits = y_logits
        self.thresholds = torch.linspace(0, 1, 100)

        # self.compute_metrices_at_different_thresholds()

        def compute_metrices_at_different_thresholds(self):
            """
            compute metrics at different thresholds and store them in self.threshold_metrics
            """
            for threshold in self.thresholds:
                y_preds = (self.y_pred > threshold).float()
                self.threshold_metrics = self.metrics(y_preds, self.y_trues)

        def visualize_threshold_metric_plot(self, metric):
            """
            function to visualize the metric at different thresholds using matplotlib

            Args:
            metric: str, metric to visualize, check self.metrics for available metrics
            """
            plt.plot(self.thresholds, self.threshold_metrics[metric].compute().numpy())
            plt.xlabel("Threshold")
            plt.ylabel(metric)
            plt.show()
