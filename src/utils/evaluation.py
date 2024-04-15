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

        self.last_runs = None
        self.best_epochs = None

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

        self.fetch_runs()
        self.get_config()

    def fetch_runs(self):
        """
        Fetches all runs from the specified project and constructs a DataFrame.
        """
        runs = self.api.runs(self.project_name)
        summary_list, config_list, name_list, id_list = [], [], [], []

        for run in runs:
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )
            name_list.append(run.name)
            id_list.append(run.id)

        self.runs_df = pd.DataFrame(
            {
                "summary": summary_list,
                "config": config_list,
                "name": name_list,
                "id": id_list,
            }
        )

    def get_config(self):
        """
        Constructs and returns a full DataFrame of all run configurations.
        """
        self.config = pd.concat(
            [pd.DataFrame([d]) for d in self.runs_df["config"]], ignore_index=True
        )
        self.config["id"] = self.runs_df["id"]
        return self.config

    def get_last_runs(self):
        """
        Constructs and returns the last runs for each configuration.
        """
        result_df = pd.DataFrame(columns=self.metrics)

        for i in range(len(self.runs_df)):
            first_row = self.runs_df["summary"].iloc[i]

            dict = {}
            for metric in self.metrics:
                dict[metric] = first_row[metric]

            temp_df = pd.DataFrame(dict, index=[0])

            result_df = pd.concat([result_df, temp_df], ignore_index=True)

        self.last_runs = pd.concat([self.config, result_df], axis=1)

        return self.last_runs

    def get_best_epoch(self):
        """
        Constructs and returns the best runs for each configuration at lowest val_loss.
        """

        best_epochs = pd.DataFrame(columns=self.metrics + ["epoch"])

        results = [self._get_best_epoch(run) for run in self.runs_df["id"]]
        for result in results:
            best_epochs = pd.concat([best_epochs, result], ignore_index=True)

        self.best_epochs = pd.merge(self.config, best_epochs, on="id", how="outer")

        return self.best_epochs

    def _get_best_epoch(self, run):

        run_hist = self.api.run(f"{self.project_name}/{run}").history()
        run_hist = run_hist[self.metrics + ["epoch"]]
        run_hist["id"] = run

        return (
            run_hist.groupby("epoch")
            .first()
            .sort_values("val_loss")
            .head(1)
            .reset_index()
        )
