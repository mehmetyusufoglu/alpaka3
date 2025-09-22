# How to render PlantUML diagrams (Java only)

1) Download PlantUML JAR

```bash
wget -q https://github.com/plantuml/plantuml/releases/download/v1.2024.6/plantuml-1.2024.6.jar -O plantuml.jar
```

2) Generate PNG from a .puml file

From this diagrams/ folder:

```bash
java -jar plantuml.jar -tpng *.puml
```

Or render a single file to the parent output folder:

```bash
java -jar plantuml.jar -tpng -o ../output yourDiagram.puml
```

Notes:
- Java (JRE) must be installed and on PATH.
- Graphviz (dot) is optional; without it, PlantUML still renders most diagrams.
