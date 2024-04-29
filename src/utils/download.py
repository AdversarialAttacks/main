import os
import wandb
from tqdm import tqdm
from src.utils.evaluation import WeightsandBiasEval


def download_models(entity, project):
    models = os.listdir("models")
    models = [model for model in models if os.path.isdir(f"models/{model}")]
    if len(models) != 22:
        evaluator = WeightsandBiasEval(entity_project_name=f"{entity}/{project}")
        best_models = evaluator.get_best_models()

        for idx, metadata in tqdm(
            best_models.iterrows(), desc="Model - Dataset Pair", position=0, total=len(best_models)
        ):
            print(f"\n---\nPair: {idx} - Model: {metadata.model} - Dataset: {metadata.dataset}")
            model_artifact = wandb.Api().artifact(f"{entity}/{project}/model-{metadata.id}:best", type="model")
            model_path = model_artifact.file(root=f"models/{metadata.model}-{metadata.dataset}/")

    return models
