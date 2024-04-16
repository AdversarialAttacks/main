import torch
import torchmetrics
from matplotlib.pyplot import plt


def get_metrics(threshold=0.5):
    return torchmetrics.MetricCollection(
        [
            torchmetrics.Accuracy(task="binary", average="macro", threshold=threshold),
            torchmetrics.Precision(task="binary", average="macro", threshold=threshold),
            torchmetrics.Recall(task="binary", average="macro", threshold=threshold),
            torchmetrics.F1Score(task="binary", average="macro", threshold=threshold),
            torchmetrics.Specificity(
                task="binary", average="macro", threshold=threshold
            ),
            torchmetrics.AUROC(task="binary"),
        ]
    )


metrics = get_metrics()


def metrics_threshold(y_logit, y_true, threshold):
    metrics = get_metrics(threshold)
    return metrics(y_logit, y_true)


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

        self.metrics_at_different_thresholds()

    def metrics_at_different_thresholds(self):
        """
        compute metrics at different thresholds and store them in self.threshold_metrics
        """
        for threshold in self.thresholds:
            self.threshold_metrics = metrics_threshold(
                self.y_logits, self.y_trues, threshold
            )

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
