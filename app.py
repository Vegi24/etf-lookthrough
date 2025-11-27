import json
from pathlib import Path
from io import StringIO

import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

DATA_DIR = Path("data")

# Länder-Zentroid-Koordinaten (für die Punktkarte)
COUNTRY_COORDS = {
    "USA": (37.1, -95.7),
    "Vereinigte Staaten": (37.1, -95.7),

    "Germany": (51.1657, 10.4515),
    "Deutschland": (51.1657, 10.4515),

    "France": (46.2276, 2.2137),
    "Frankreich": (46.2276, 2.2137),

    "United Kingdom": (55.3781, -3.4360),
    "Großbritannien": (55.3781, -3.4360),
    "Vereinigtes Königreich": (55.3781, -3.4360),

    "Netherlands": (52.1326, 5.2913),
    "Niederlande": (52.1326, 5.2913),

    "Austria": (47.5162, 14.5501),
    "Österreich": (47.5162, 14.5501),

    "Switzerland": (46.8182, 8.2275),
    "Schweiz": (46.8182, 8.2275),

    "Sweden": (60.1282, 18.6435),
    "Schweden": (60.1282, 18.6435),

    "Italy": (41.8719, 12.5674),
    "Italien": (41.8719, 12.5674),

    "Japan": (36.2048, 138.2529),

    "Australia": (-25.2744, 133.7751),
    "Australien": (-25.2744, 133.7751),

    "China": (35.8617, 104.1954),
}

# --------------------------------------------------------------------------------------
#  iShares: Holdings-CSV-Links (offizielle iShares-Downloads)
# --------------------------------------------------------------------------------------
ISHARES_HOLDINGS_URLS = {
    # iShares MSCI World Information Technology Sector Advanced UCITS ETF (WITS)
    "IE00BJ5JNY98": (
        "https://www.ishares.com/ch/privatkunden/de/produkte/308858/"
        "fund/1495092304805.ajax?dataType=fund&fileName=WITS_holdings&fileType=csv"
    ),
    # iShares Automation & Robotics UCITS ETF (RBOT)
    "IE00BYZK4552": (
        "https://www.ishares.com/ch/privatkunden/de/produkte/284219/"
        "fund/1495092304805.ajax?dataType=fund&fileName=RBOT_holdings&fileType=csv"
    ),
    # iShares MSCI World Health Care Sector Advanced UCITS ETF (WHCS)
    "IE00BJ5JNZ06": (
        "https://www.ishares.com/ch/privatkunden/de/produkte/308909/"
        "fund/1495092304805.ajax?dataType=fund&fileName=WHCS_holdings&fileType=csv"
    ),
}

# --------------------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------------------

def load_portfolio(path: Path) -> pd.DataFrame:
    """Lädt dein Portfolio aus portfolio.json und berechnet ETF-Gewichte im Depot."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    total = df["value_eur"].sum()
    df["weight_in_portfolio"] = df["value_eur"] / total
    return df


def load_ishares_holdings_from_url(isin: str) -> pd.DataFrame:
    """
    Lädt die Holdings direkt von der iShares-Seite über den CSV-Link.

    Format wie in deinem Beispiel:
      Fondsposition per,"25.Nov.2025"
      <Leerzeile>
      Emittententicker,Name,Sektor,Anlageklasse,Marktwert,Gewichtung (%),...,Standort,Börse,Marktwährung
    """
    if isin not in ISHARES_HOLDINGS_URLS:
        raise ValueError(f"Keine iShares-Holdings-URL für ISIN {isin} konfiguriert.")

    url = ISHARES_HOLDINGS_URLS[isin]
    resp = requests.get(url)
    resp.raise_for_status()

    text = resp.text
    lines = text.splitlines()

    # Header-Zeile finden (fängt mit "Emittententicker," oder "Ticker," an)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Emittententicker,") or line.startswith("Ticker,"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Konnte Header-Zeile in iShares-CSV nicht finden.")

    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(csv_text))

    # Spaltennamen lowercased für robuste Suche
    lower_map = {c.lower(): c for c in df.columns}

    # Name-Spalte
    if "name" in lower_map:
        name_col = lower_map["name"]
    else:
        raise ValueError(f"Spalte 'Name' in iShares-CSV nicht gefunden. Spalten: {list(df.columns)}")

    # Gewichtung-Spalte (in %)
    weight_candidates = [
        "gewichtung (%)",
        "gewichtung%",
        "gewicht (%)",
        "gewicht",
        "weight (%)",
        "weight%",
        "weight",
    ]
    weight_col = next((lower_map[k] for k in weight_candidates if k in lower_map), None)
    if weight_col is None:
        raise ValueError(
            f"Keine Gewichtsspalte in iShares-CSV gefunden. Spalten: {list(df.columns)}"
        )

    # Länder-Spalte (Standort)
    country_candidates = [
        "standort",
        "land",
        "country",
        "issuer country",
        "domicile",
    ]
    country_col = next((lower_map[k] for k in country_candidates if k in lower_map), None)

    cols = [name_col, weight_col]
    if country_col is not None:
        cols.append(country_col)

    result = df[cols].copy()
    result.rename(columns={name_col: "name", weight_col: "weight_pct"}, inplace=True)

    if country_col is not None:
        result.rename(columns={country_col: "country"}, inplace=True)
    else:
        result["country"] = pd.NA

    # Gewichtung (%) -> Anteil [0,1]
    cleaned = (
        result["weight_pct"]
        .astype(str)
        .str.replace("’", "", regex=False)   # 220’671 -> 220671
        .str.replace("'", "", regex=False)
        .str.replace(",", ".", regex=False)  # falls 19,01 vorkommt
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    result["weight_pct"] = numeric / 100.0

    result = result.dropna(subset=["weight_pct"])

    return result


def load_invesco_local(isin: str) -> pd.DataFrame:
    """
    Liest Invesco-Holdings aus einer lokalen Datei im Ordner data/.

    Erlaubt:
      - data/invesco_<ISIN>.xlsx  (Excel)
      - data/invesco_<ISIN>.csv   (CSV)

    Unterstützt sowohl Prozent-Formate (z.B. "5,43" oder "5.43 %")
    als auch das neue Format mit Anteilen (z.B. "0,051580471").

    Beispiel-CSV:
      Full name;ISIN;Weight
      CIPHER MINING INC ...;US17253J1060;0,051580471
    """
    xlsx_path = DATA_DIR / f"invesco_{isin}.xlsx"
    csv_path = DATA_DIR / f"invesco_{isin}.csv"

    if xlsx_path.exists():
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Lesen der Invesco-Excel {xlsx_path}: {e}")
        source_path = xlsx_path

    elif csv_path.exists():
        # CSV: explizit Semikolon, Python-Engine, kaputte Zeilen überspringen
        try:
            try:
                df = pd.read_csv(
                    csv_path,
                    sep=";",
                    engine="python",
                    on_bad_lines="skip",  # pandas >= 1.3
                )
            except TypeError:
                # Fallback für ältere pandas-Versionen
                df = pd.read_csv(
                    csv_path,
                    sep=";",
                    engine="python",
                    error_bad_lines=False,
                    warn_bad_lines=True,
                )
        except Exception as e:
            raise RuntimeError(f"Fehler beim Lesen der Invesco-CSV {csv_path}: {e}")
        source_path = csv_path

    else:
        raise FileNotFoundError(
            f"Keine Datei für Invesco-ETF {isin} gefunden. "
            f"Erwarte data/invesco_{isin}.xlsx oder data/invesco_{isin}.csv"
        )

    lower_map = {c.lower(): c for c in df.columns}

    # Kandidaten für die Namensspalte – "full name" inklusive
    name_candidates = [
        "full name",
        "name",
        "titel",
        "security name",
        "issuer name",
        "position",
        "bezeichnung",
        "holding",
        "constituent",
    ]
    # Kandidaten für Gewichts-Spalte (in % oder als Anteil)
    weight_candidates = [
        "weight",
        "weight (%)",
        "weight%",
        "gewichtung (%)",
        "gewichtung",
        "gewicht (%)",
        "gewicht",
        "portfolio weight",
        "portfolio weight (%)",
        "gewicht in %",
    ]

    name_col = next((lower_map[k] for k in name_candidates if k in lower_map), None)
    weight_col = next((lower_map[k] for k in weight_candidates if k in lower_map), None)

    if name_col is None or weight_col is None:
        raise ValueError(
            f"Konnte Name/Weight-Spalten in {source_path} nicht erkennen. "
            f"Spalten: {list(df.columns)}"
        )

    resu
