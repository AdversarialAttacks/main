%%{init: {'flowchart': {'htmlLabels': true, 'curve': 'linear'}, 'theme': 'base', 'themeVariables': {'backgroundColor': 'white', 'primaryColor': 'white', 'secondaryColor': 'white', 'lineColor': 'grey', 'primaryBorderColor': 'grey', 'tertiaryBorderColor': 'grey'}}}%%

graph LR
    subgraph a[ ]
    direction TB
        subgraph Modelltraining[Modelle trainieren]
        direction TB
            ModellTraining(Modell auf unseren \nDatensatz trainieren)
            UapGen(UAPs generieren)
            UapSave(UAPs speichern)
            Inference(Inference)
            InferenceAdv(Inference mit UAPs\nAdversarial Attack)
            Evaluation(Evaluieren der \nFooling-Performance)
        end

        subgraph Schutzmechanismen[Modelle schützen]

        direction TB
            ModellFinetuning(Modell mit UAPs und \nSchutzmechanismus finetunen)
            UapGen2(UAPs generieren)
            UapSave2(UAPs speichern)
            Inference2(Inference)
            InferenceAdv2(Inference mit UAPs\nAdversarial Attack)
            Evaluation2(Evaluieren der \nFooling-Performance)
            Robust2(Modell weiter \nrobustifizieren)
        end
    end
 

ModellTraining ==> Inference
ModellTraining ==> UapGen
UapGen ==> InferenceAdv
Inference ==> Evaluation
InferenceAdv ==> Evaluation
UapGen ==> UapSave
UapSave ==> ModellFinetuning

%%Modelltraining ===> Schutzmechanismen

ModellFinetuning ==> Inference2
ModellFinetuning ==> UapGen2
UapGen2 ==> InferenceAdv2
Inference2 ==> Evaluation2
InferenceAdv2 ==> Evaluation2
UapGen2 ==> UapSave2
UapSave2 ==> Robust2
Evaluation2 ==> Robust2
Robust2 .-> | x Mal | ModellFinetuning
