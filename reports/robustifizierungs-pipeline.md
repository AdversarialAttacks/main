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

            subgraph robustifizierung[" "]
                ModellRobustifizieren(Modell robustifizieren)
                Inferencemit2(Inference mit ...)
                Testdata2(Test Datensatz)
                TestdatamitUAP2(Test Datensatz pro UAP)
            end

            Evaluierung(Evaluierung)
            Modelersetzen(Modell mit robustifizierten Modell ersetzen)
    end


ModellTrained --> Inferencemit
ModellTrained --> UAP
UAP --> ModellRobustifizieren
ModellTrained --> ModellRobustifizieren

Inferencemit -.-> Testdata
Inferencemit -.-> TestdatamitUAP

UAP --> TestdatamitUAP
UAP --> TestdatamitUAP2

ModellRobustifizieren --> Inferencemit2

Inferencemit2 -.-> Testdata2
Inferencemit2 -.-> TestdatamitUAP2

Testdata --> Evaluierung
TestdatamitUAP --> Evaluierung

TestdatamitUAP2 --> Evaluierung
Testdata2 --> Evaluierung

Evaluierung --> Modelersetzen
Modelersetzen -.-> |x mal| ModellTrained




