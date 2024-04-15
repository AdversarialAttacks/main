import wandb
import pandas as pd


class WeightsandBiasEval:
    def __init__(self, entity_project_name):
        """
        Initializes the WandBManager with the specified project.

        :param entity_project_name: A string specifying the 'entity/project-name' for WandB.
        """
        self.api = wandb.Api()
        self.project_name = entity_project_name
        self.metrics = [
            "train_BinaryAUROC",
            "train_BinaryAccuracy",
            "train_BinaryF1Score",
            "train_BinaryPrecision",
            "train_BinaryRecall",
            "train_BinarySpecificity",
            "train_loss",
            "val_BinaryAUROC",
            "val_BinaryAccuracy",
            "val_BinaryF1Score",
            "val_BinaryPrecision",
            "val_BinaryRecall",
            "val_BinarySpecificity",
            "val_loss",
        ]

        self.last_runs = None
        self.best_epochs = None

        self.fetch_runs()
        self.get_config()

    def fetch_runs(self):
        """
        Fetches all runs from the specified project and constructs a DataFrame.
        """
        runs = self.api.runs(self.project_name)
        data = {
            "id": [run.id for run in runs],
            "name": [run.name for run in runs],
            "config": [{k: v for k, v in run.config.items() if not k.startswith("_")} for run in runs],
            "summary": [run.summary._json_dict for run in runs],
        }
        self.runs_df = pd.DataFrame(data)

        return self.runs_df

    def get_config(self):
        """
        Constructs and returns a full DataFrame of all run configurations.
        """
        self.config = self.runs_df["config"].apply(pd.Series)
        self.config["id"] = self.runs_df["id"]

        return self.config

    def get_last_runs(self):
        """
        Constructs and returns the last runs for each configuration.
        """
        result_df = self.runs_df["summary"].apply(pd.Series)[self.metrics]
        self.last_runs = pd.concat([self.config, result_df], axis=1)

        first_columns = ["id", "model", "dataset"]
        self.last_runs = self.__sort_columns(self.last_runs, first_columns)

        return self.last_runs

    def get_best_epochs(self):
        """
        Constructs and returns the best runs for each configuration at the lowest val_loss.
        """
        results = [self.__get_best_epoch_with_runid(run_id) for run_id in self.runs_df["id"]]

        best_epochs = pd.concat(results, ignore_index=True)
        self.best_epochs = pd.merge(self.config, best_epochs, on="id", how="outer")

        first_columns = ["id", "model", "dataset", "epoch"]
        self.best_epochs = self.__sort_columns(self.best_epochs, first_columns)

        return self.best_epochs

    def __get_best_epoch_with_runid(self, run):
        run_hist = self.api.run(f"{self.project_name}/{run}").history()
        run_hist = run_hist[["epoch"] + self.metrics]
        run_hist["id"] = run

        run_hist = run_hist.groupby("epoch").first()
        run_hist = run_hist.sort_values("val_loss")

        return run_hist.head(1).reset_index()

    def __sort_columns(self, df, first_columns):
        return df[first_columns + [col for col in df if col not in first_columns]]
