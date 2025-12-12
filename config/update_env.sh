#!/bin/bash

# --- Skript zur automatischen Aktualisierung oder Neuerstellung der Conda-Umgebung ---

ENV_NAME="Quant"
YAML_FILE="environment.yml"

echo "Starte Conda Umgebungs-Management..."

# 1. Prüfen, ob die Umgebung existiert
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Umgebung '$ENV_NAME' gefunden. Führe Update durch (kann dauern)..."
    # Führt ein Update auf Basis der YML-Datei durch (optimale Lösung)
    conda env update -f "$YAML_FILE"
else
    echo "Umgebung '$ENV_NAME' nicht gefunden. Erstelle Umgebung neu..."
    # Erstellt die Umgebung komplett neu
    conda env create -f "$YAML_FILE"
fi

# 2. Umgebung aktivieren
echo "Aktiviere Umgebung '$ENV_NAME'."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 3. Bestätigung
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================"
    echo "ERFOLG: Conda Umgebung ist bereit und aktualisiert."
    echo "Sie sind nun in der Umgebung '$ENV_NAME'."
    echo "Führen Sie Streamlit aus mit: streamlit run app.py"
    echo "========================================================"
else
    echo "FEHLER: Conda konnte die Umgebung nicht aktualisieren/erstellen."
fi
