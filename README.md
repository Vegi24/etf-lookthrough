# ETF Look-Through – Streamlit App

Diese App berechnet aus einem ETF-Portfolio das effektive **Einzelaktien-Exposure**
(„Look-Through-Analyse“).

## Features

- Einlesen eines Portfolios aus `portfolio.json`
- Automatisches Laden der Holdings für
  - iShares-ETFs
  - Invesco-ETFs  
- Einlesen von Amundi-Holdings aus CSV-Dateien im Ordner `data/`
- Aggregation der Einzelaktien-Gewichte über alle ETFs
- Anzeige der Top-Aktien + Detailtabelle (welche Aktie steckt in welchem ETF?)

## Installation

```bash
git clone <dein-repo-url>
cd etf-lookthrough

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
