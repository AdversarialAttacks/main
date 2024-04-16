import torch
import torchmetrics
import matplotlib.pyplot as plt
from collections import defaultdict 


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
        y_preds,
        metrics=metrics,
    ):
        self.metrics = metrics
        self.y_trues = y_trues
        self.y_preds = y_preds
        self.thresholds = torch.linspace(0.01, 0.99, 99)
        self.metrics_at_different_thresholds()

    def metrics_at_different_thresholds(self):
        """
        compute metrics at different thresholds and store them in self.threshold_metrics
        """
        self.threshold_metrics = defaultdict(list)
        
        for threshold in self.thresholds:
            metrics = metrics_threshold(self.y_preds, self.y_trues, threshold.item())
            for metric, value in metrics.items():
                self.threshold_metrics[metric].append(value)
        
        self.threshold_metrics = {metric: torch.stack(values) for metric, values in self.threshold_metrics.items()}

    def visualize_threshold_metric_plot(self, metric):
        """
        function to visualize the metric at different thresholds using matplotlib

        Args:
        metric: str, metric to visualize, check self.metrics for available metrics
        """
        plt.plot(self.thresholds.numpy(), self.threshold_metrics[metric].numpy())
        plt.xlabel("Threshold")
        plt.ylabel(metric)
        plt.show()
