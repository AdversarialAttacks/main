graph TD
    subgraph UAP Algorithm
        Start((Start))
        InitializeV[Initialize list of perturbations v]
        subgraph Generate i universal perturbations
            Iterate[Iterate through specified number i of desired universal perturbation images v]
            Initialize[Initialize perturbation tensor v_i as an image with zeroes]
            subgraph Epochs
                ForEach[For each n perturbation images in the training dataset]
                subgraph Iterate through images
                    GetImage[Get the next image in training dataset]
                    InitializeVTemp[Initialize image specific perturbation v_temp with zeroes]
                    Check[Check if model is fooled by the perturbation v + v_temp]
                    subgraph Optimization Problem
                        Predict["Make model prediction with original and perturbed image (image + v_i + v_temp)"]
                        Break[Retried too often?]
                        Loss[Calculate the loss with the specified norm and inverse BCE]
                        Backpropagate[Backpropagate on the loss to get closer to the decision boundary]
                        CheckDecisionBoundary[Check if the decision boundary was reached]
                    end
                    Add[Add local perturbation v_temp to universal perturbation v_i]
                    Repeat[Check if all n train images have been iterated through]
                end
                Calculate[Calculate fooling rate of perturbation v_i and check if desired fooling rate is reached]
            end
            Save[Append perturbation v_i to list v]
            CheckNPerturbations[Check if desired amount of universal perturbations i has been reached]
        end
        
        Return((Return v))
    end

    Start --> InitializeV
    InitializeV --> Iterate
    Iterate --> Initialize
    Initialize --> ForEach
    ForEach --> GetImage
    GetImage --> InitializeVTemp
    InitializeVTemp --> Check
    Check -->|Yes| Add
    Check -->|No| Predict
    Predict --> Loss
    Loss --> Backpropagate
    Backpropagate --> CheckDecisionBoundary
    CheckDecisionBoundary --> |Yes| Add
    CheckDecisionBoundary --> |No| Break
    Break --> |No| Predict
    Break -.-> |Yes| Repeat
    Add --> Repeat
    Repeat -->|Yes| Calculate
    Repeat --> |No| GetImage
    Calculate --> |Yes| Save
    Calculate --> |No| ForEach
    Save --> CheckNPerturbations
    CheckNPerturbations -->|No| Iterate
    CheckNPerturbations -->|Yes| Return