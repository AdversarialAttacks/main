# 24FS_I4DS27: Adversarial Attacks

Dies ist das Hauptrepository für die Bachelorarbeit 24FS_I4DS27: Adversarial Attacks. <br>
Hier trainieren, validieren, attackieren, verteidigen und evaluiren wir unsere Modelle.

## Setup

1. Klone das Repository.
2. Installiere die benötigten Python Packages mit `make reqs` oder `make reqs-cuda`.
3. Erstelle einen Kaggle API Access Token und lege ihn ab unter `~/.kaggle/kaggle.json`. Die Anleitung dafür findet man [hier](https://www.kaggle.com/docs/api).
4. Lade die Daten herunter. Dies kann mit den Download Befehlen im Makefile gemacht werden. Diese können unter `make help` eingesehen werden.
 
## Projektstruktur

    ├── data
    │   ├── processed           <- Die finalen Datensätze für die Modellierung.
    │   └── raw                 <- Der ursprüngliche, unveränderliche Datendump.
    │
    ├── models                  <- Trainierte Modelle.
    ├── notebooks               <- Jupyter-Notebooks. Die Namenskonvention ist eine Nummer und der
    │                              Use-Case des Notebooks. Zum Beispiel: '03-eda.ipynb'
    │
    ├── reports                 <- Verweisende Links zum Bericht.
    ├── src                     <- Quellcode für die Verwendung in diesem Projekt.
    ├── .gitignore              <- Dateien, die beim Gebrauch von git ignoriert werden sollen.
    ├── LICENSE                 <- MIT-Lizenz.
    ├── Makefile                <- Makefile mit Befehlen wie 'make reqs' oder 'make reqs-cuda'.
    ├── requirements.txt        <- Die Requirementsdatei zur Reproduktion der Analyse.
    ├── requirements-cuda.txt   <- Die Requirementsdatei zur Reproduktion der Analyse mit CUDA Support.
    ├── README.md               <- Die oberste README-Datei für Entwickler, die dieses Projekt verwenden.
