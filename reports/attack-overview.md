%%{init: {'flowchart': {'htmlLabels': true, 'curve': 'basis'}, 'theme': 'base', 'themeVariables': {'backgroundColor': 'white', 'primaryColor': 'white', 'secondaryColor': 'white', 'lineColor': 'grey', 'primaryBorderColor': 'grey', 'tertiaryBorderColor': 'grey'}}}%%

graph TB
    subgraph idee[" "]
        direction LR
            Imageoriginal("$$x_0 $$")
            subgraph Adversarial Attack
            Attackwithsmallpertub("$$ + v $$")
            AdversarialImage("$$ x_0^{*} = x_0 + v $$")
            end
            Modelf(Modell c)
            Modelf2(Modell c)
            output1("$$c(x_0)=y_0$$")
            outputadv("$$c(x_0^{*})=y_0^{*}$$")
            ungleich("$$ ≠ $$")
    end


Imageoriginal --> Modelf 
AdversarialImage --> Modelf2

Imageoriginal -->  Attackwithsmallpertub
Attackwithsmallpertub --> AdversarialImage 

Modelf --> output1
Modelf2 --> outputadv

output1 --> ungleich
outputadv --> ungleich
