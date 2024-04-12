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
        self.runs_df = None
        self.config = None
        self.w_b_df = None

    def fetch_runs(self):
        """
        Fetches all runs from the specified project and constructs a DataFrame.
        """
        runs = self.api.runs(self.project_name)
        summary_list, config_list, name_list = [], [], []

        for run in runs:
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )
            name_list.append(run.name)

        self.runs_df = pd.DataFrame(
            {"summary": summary_list, "config": config_list, "name": name_list}
        )

    def save_runs_to_csv(self, path="src/data", filename="project", format="csv"):
        """
        Saves the runs DataFrame to a CSV file.

        :param filename: A string specifying the filename for the CSV.
        """
        if self.runs_df is not None:
            self.runs_df.to_csv(f"{path}/{filename}.{format}")
        else:
            print(
                "No runs data to save. Please fetch the runs first by using method fetch_runs()"
            )

    def get_full_config_df(self):
        """
        Constructs and returns a full DataFrame of all run configurations.
        """
        if self.runs_df is None:
            raise ValueError("No runs data available. Please fetch the runs first.")

        self.config = pd.concat(
            [pd.DataFrame([d]) for d in self.runs_df["config"]], ignore_index=True
        )
        return self.config


    def get_full_metrics_df(self):
        """
        Constructs and returns a full Dataframe of all run metrics
        """
        if self.runs_df is None:
            raise ValueError("No runs data available. Please fetch the runs first.")
        
        config_df = self.get_full_config_df()

        metrics = [
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
        
        result_df = pd.DataFrame(columns=metrics)
        
        for i in range(len(self.runs_df)):
            first_row = self.runs_df["summary"].iloc[i]

            # create a dict with key (metric) and its value in first_row
            dict = {}
            for metric in metrics:
                dict[metric] = first_row[metric]

            # turn dict into dataframe 
            temp_df = pd.DataFrame(dict, index=[0])

            # concat to result_df and emptu temp_df
            result_df = pd.concat([result_df, temp_df], ignore_index=True)

        # concat result_df with config_df
        result_df = pd.concat([config_df, result_df], axis=1)

        self.w_b_df = result_df
        return self.w_b_df



    def get_best_run(self, sweep_id):
        """
        Returns the best run from a specified sweep.
        """
        if sweep_id is None:
            raise ValueError(
                "Sweep ID is not set. Please provide a sweep ID during initialization."
            )

        sweep = self.api.sweep(f"{self.project_name}/{sweep_id}")
        return sweep.best_run()
    
