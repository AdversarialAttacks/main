%%{init: {'flowchart': {'htmlLabels': true, 'curve': 'basis'}, 'theme': 'base', 'themeVariables': {'backgroundColor': 'white', 'primaryColor': 'white', 'secondaryColor': 'white', 'lineColor': 'grey', 'primaryBorderColor': 'grey', 'tertiaryBorderColor': 'grey'}}}%%

graph 
subgraph a[ ]
    direction TB
    Train(Train model)
    Dataset[(Dataset)]
    GenAdv(Generate universal adversarial attack)
    CombineData(Combine data)
    AdversarialAttack[(UAP)]
end

Train -.-> GenAdv
Train -.-> Dataset
GenAdv -.-> |store| AdversarialAttack
AdversarialAttack -.-> CombineData
Dataset -.-> GenAdv
Dataset -.-> CombineData
CombineData --> |repeat x times| Train
