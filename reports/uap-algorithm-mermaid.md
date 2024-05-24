%%{init: {'theme': 'base', 'themeVariables': {'backgroundColor': 'white', 'primaryColor': 'white', 'secondaryColor': 'white', 'lineColor': 'grey', 'primaryBorderColor': 'grey', 'tertiaryBorderColor': 'grey'}}}%%

graph TD
    subgraph UAP Algorithmus 
        Start((Start))
        InitializeV(Initialisiere Liste von Perturbationen v)
        subgraph Erzeuge i universelle Perturbationen[ ]
            Iteriere(Iteriere durch die spezifizierte Anzahl i der gewünschten universellen Perturbationsbilder v)
            Initialize(Initialisiere Perturbationstensor v_i als Nulltensorbild)
            subgraph Epochen[ ]
                FürJede(Bereite n Bilder von Trainingsdatensatz vor)
                subgraph Durchlaufe Bilder[ ]
                    HolBild(Hole das nächste Bild im Trainingsdatensatz)
                    InitialisiereVTemp(Initialisiere bildspezifische Perturbation delta_v als Nulltensorbild)
                    Prüfe(Prüfe, ob das Modell mit dem Bild durch die Perturbation v + delta_v getäuscht wird)
                    subgraph Optimierungsproblem[ ]
                        Vorhersage("Mache Modellvorhersage mit dem originalen und perturbierten Bild (img + v_i + delta_v)")
                        Abbrechen(t mal wiederholt?)
                        Verlust(Berechne den Loss mit der spezifizierten Norm und inversen BCE)
                        Rückpropagiere(Rückpropagiere den Verlust, um näher an die Decision Boundary zu gelangen)
                        PrüfeDecisionBoundary(Prüfe, ob die Decision Boundary erreicht wurde)
                        Hinzufügen(Füge lokale Perturbation delta_v zur universellen Perturbation v_i hinzu)
                    end
                    Wiederholen(Prüfe, ob alle n Trainingsbilder durchlaufen wurden)
                end
                Berechne(Berechne die Fooling Rate der Perturbation v_i und prüfe, ob die gewünschte Fooling Rate r erreicht ist)
            end
            Speichern(Füge Perturbation v_i zur Liste v hinzu)
            PrüfeNPerturbationen(Prüfe, ob die gewünschte Anzahl von universellen Perturbationen i erreicht wurde)
        end
        
        Rückgabe((Return v))
    end

    Start --> InitializeV
    InitializeV --> Iteriere
    Iteriere --> Initialize
    Initialize --> FürJede
    FürJede --> HolBild
    HolBild --> InitialisiereVTemp
    InitialisiereVTemp --> Prüfe
    Prüfe -->|Ja| Wiederholen
    Prüfe -->|Nein| Vorhersage
    Vorhersage --> Verlust
    Verlust --> Rückpropagiere
    Rückpropagiere --> PrüfeDecisionBoundary
    PrüfeDecisionBoundary --> |Ja| Hinzufügen
    PrüfeDecisionBoundary --> |Nein| Abbrechen
    Abbrechen --> |Nein| Vorhersage
    Abbrechen -.-> |Ja| Wiederholen
    Hinzufügen --> Wiederholen
    Wiederholen -->|Ja| Berechne
    Wiederholen -->|Nein| HolBild
    Berechne -->|Ja| Speichern
    Berechne -->|Nein| FürJede
    Speichern --> PrüfeNPerturbationen
    PrüfeNPerturbationen -->|Nein| Iteriere
    PrüfeNPerturbationen -->|Ja| Rückgabe
