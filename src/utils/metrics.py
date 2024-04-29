import torch
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


def get_metrics(threshold=0.5):
    return torchmetrics.MetricCollection(
        [
            torchmetrics.Accuracy(task="binary", average="macro", threshold=threshold),
            torchmetrics.Precision(task="binary", average="macro", threshold=threshold),
            torchmetrics.Recall(task="binary", average="macro", threshold=threshold),
            torchmetrics.F1Score(task="binary", average="macro", threshold=threshold),
            torchmetrics.Specificity(task="binary", average="macro", threshold=threshold),
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

    def visualize_threshold_metric_plot(self, metric, model, dataset):
        """
        function to visualize the metric at different thresholds using matplotlib

        Args:
        metric: str, metric to visualize, check self.metrics for available metrics
        """
        plt.figure(figsize=(10, 5), dpi=200)
        plt.plot(self.thresholds.numpy(), self.threshold_metrics[metric].numpy())
        plt.title(f"{metric} at different thresholds for {model} on {dataset}")
        best_threshold = self.thresholds[self.threshold_metrics[metric].argmax()].item()
        plt.axvline(best_threshold, color="green", linestyle="--", label="Best Threshold")
        default_threshold = 0.5
        plt.axvline(default_threshold, color="red", linestyle="--", label="Default Threshold")
        plt.xlabel("Threshold")
        plt.ylabel(metric)
        plt.legend()
        plt.xticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.grid(axis="x", alpha=0.5)
        plt.show()

    def visualize_confusion_matrix(self, model, dataset):
        """
        function to visualize the confusion matrix using matplotlib
        """
        confusion_matrix = torchmetrics.ConfusionMatrix(task="binary")(self.y_preds.cpu(), self.y_trues.cpu()).numpy()
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for {model} on {dataset}")
        plt.xlabel("Predicted Labels")
        plt.xticks([0.5, 1.5], ["False", "True"])
        plt.ylabel("True Labels")
        plt.yticks([0.5, 1.5], ["False", "True"])
        plt.show()
