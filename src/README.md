# src

In diesem Ordner befinden sich die Python-Module, die für die Bearbeitung unserer Thesis verwendet wurden. Die Module sind in folgende Unterordner aufgeteilt:

- `data`: Enthält Module für die Datenverarbeitung
- `models`: Enthält Module für die 
- `utils`: Enthält Module für die Hilfsfunktionen


## data

`data` enthält Module für die Datenverarbeitung. 

| Datei | Beschreibung |
| --- | --- |
| [covidx.py](data/covidx.py) | Enthält die Klassen `COVIDXDataset` und `COVIDXDataModule` |
| [mri.py](data/mri.py) | Enthält die Klassen `MRIDataset` und `MRIDataModule` |

## models

`models` enthält Module für die Modelle.

| Datei | Beschreibung |
| --- | --- |
| [imageclassifier.py](models/imageclassifier.py) | Enthält die Klassen `ImageClassifier`, die uns erlaubt dynamisch das Modell zu trainieren und zu loggen |

## utils

`utils` enthält Module für die Hilfsfunktionen.

| Datei | Beschreibung |
| --- | --- |
| [adv_training.py](utils/adv_training.py) | Pipeline für Adversarial Attack und Robustifizierung |
| [download.py](utils/download.py) | Download von Modellen aus Weights & Bias |
| [eda.py](utils/eda.py) | Explorative Datenanalyse |
| [evaluation.py](utils/evaluation.py) | Evaluierung der Modelle |
| [gpu_setup.py](utils/gpu_setup.py) | Einstellung und Configuration der GPU Setup |
| [metrics.py](utils/metrics.py) | Berechnung der Metriken |
| [transform_perturbation.py](utils/transform_perturbation.py) | Custom PyTorch Preprocessing Klasse |
| [uap-eda.py](utils/uap-eda.py) | Explorative Datenanalyse der UAPs |
| [uap_helper.py](utils/uap_helper.py) | Hilfsfunktionen für die Generierung der UAPs |

