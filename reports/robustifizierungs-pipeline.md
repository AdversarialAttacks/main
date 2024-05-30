%%{init: {'flowchart': {'htmlLabels': true, 'curve': 'basis'}, 'theme': 'base', 'themeVariables': {'backgroundColor': 'white', 'primaryColor': 'white', 'secondaryColor': 'white', 'lineColor': 'grey', 'primaryBorderColor': 'grey', 'tertiaryBorderColor': 'grey'}}}%%

graph TD
    subgraph Robustifizierungs Pipeline[" "]
        direction TB

            ModellTrained(Modell)
            UAP(UAPs generieren)

            subgraph normal[" "]
                TestdatamitUAP(Test Datensatz pro UAP)
                Testdata(Test Datensatz)
                Inferencemit(Inference mit ...)
            end

            ModellRobustifizieren(Modell robustifizieren)
            subgraph robustifizierung[" "]
                Inferencemit2(Inference mit ...)
                Testdata2(Test Datensatz)
                TestdatamitUAP2(Test Datensatz pro UAP)
            end

            Evaluierung(Speicherung der Metriken & 
            Evaluierung)
            Modelersetzen(Modell mit robustifizierten 
            Modell ersetzen)
    end


ModellTrained --> UAP
UAP --> ModellRobustifizieren

Inferencemit -.-> Testdata
Inferencemit -.-> TestdatamitUAP

UAP --> Inferencemit

ModellRobustifizieren --> Inferencemit2

Inferencemit2 -.-> Testdata2
Inferencemit2 -.-> TestdatamitUAP2

Testdata --> Evaluierung
TestdatamitUAP --> Evaluierung

TestdatamitUAP2 --> Evaluierung
Testdata2 --> Evaluierung

Evaluierung --> Modelersetzen
Modelersetzen -.-> |x mal| ModellTrained

style ModellRobustifizieren fill:#D3D3D3
style UAP fill:#D3D3D3
style TestdatamitUAP fill:#D3D3D3
style TestdatamitUAP2 fill:#D3D3D3
style Modelersetzen fill:#D3D3D3