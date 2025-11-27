import json
from pathlib import Path
from io import StringIO

import pandas as pd
import requests
import streamlit as st

DATA_DIR = Path("data")

# --------------------------------------------------------------------------------------
#  iShares: Holdings-CSV-Links (offizielle iShares-Downloads)
#  Struktur wie in deinem WITS-Beispiel (erste 2 Zeilen Meta, dann CSV mit Header
#  "Emittententicker,Name,Sektor,...,Gewichtung (%)" etc.). 
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


def _clean_percent_series(s: pd.Series) -> pd.Series:
    """
    Nimmt eine Spalte mit Prozentwerten (z.B. '19.01', '3,45', '4’321,23')
    und gibt einen Float in [0,1] zurück.
    """
    return (
        s.astype(str)
        .str.replace("’", "", regex=False)   # tausender-Trennzeichen wie 220’671
        .str.replace("'", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .pipe(lambda x: pd.to_numeric(x, errors="coerce")) / 100.0
    )


def load_ishares_holdings_from_url(isin: str) -> pd.DataFrame:
    """
    Lädt die Holdings direkt von der iShares-Seite über den CSV-Link.

    Format wie in deiner WITS-Datei:
      - Zeile 0: "Fondsposition per,..." (Meta)
      - Zeile 1: Leerzeile
      - Ab Zeile 2: CSV mit Header "Emittententicker,Name,...,Gewichtung (%)"
    """
    if isin not in ISHARES_HOLDINGS_URLS:
        raise ValueError(f"Keine iShares-Holdings-URL für ISIN {isin} konfiguriert.")

    url = ISHARES_HOLDINGS_URLS[isin]
    resp = requests.get(url)
    resp.raise_for_status()

    text = resp.text
    lines = text.splitlines()

    # Header-Zeile finden (fängt mit "Emittententicker," an – siehe WITS CSV)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Emittententicker,") or line.startswith("Ticker,"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Konnte Header-Zeile in iShares-CSV nicht finden.")

    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(csv_text))

    # Spalten identifizieren
    # Name
    if "Name" in df.columns:
        name_col = "Name"
    else:
        raise ValueError("Spalte 'Name' in iShares-CSV nicht gefunden.")

    # Gewichtung
    weight_col = None
    weight_candidates = [
        "Gewichtung (%)",
        "Gewichtung (%)",
        "Weight (%)",
        "Weighting (%)",
        "Gewicht (%)",
        "Gewicht %",
    ]
    for c in weight_candidates:
        if c in df.columns:
            weight_col = c
            break

    if weight_col is None:
        raise ValueError(
            f"Keine Gewichtsspalte in iShares-CSV gefunden. Spalten: {list(df.columns)}"
        )

    result = df[[name_col, weight_col]].copy()
    result.rename(columns={name_col: "name", weight_col: "weight_pct"}, inplace=True)

    # Prozentwerte in [0,1] umrechnen
    result["weight_pct"] = _clean_percent_series(result["weight_pct"])
    result = result.dropna(subset=["weight_pct"])

    return result

def load_invesco_local(isin: str) -> pd.DataFrame:
    """
    Liest Invesco-Holdings aus einer lokalen Datei im Ordner data/.
    Erlaubt:
      - data/invesco_<ISIN>.xlsx  (Excel)
      - data/invesco_<ISIN>.csv   (CSV)
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
        df = pd.read_csv(csv_path)
        source_path = csv_path
    else:
        raise FileNotFoundError(
            f"Keine Datei für Invesco-ETF {isin} gefunden. "
            f"Erwarte data/invesco_{isin}.xlsx oder data/invesco_{isin}.csv"
        )

    lower_map = {c.lower(): c for c in df.columns}

    name_candidates = [
        "name", "titel", "security name", "issuer name",
        "position", "bezeichnung", "holding", "constituent"
    ]
    weight_candidates = [
        "weight (%)", "gewichtung (%)", "gewichtung",
        "gewicht (%)", "gewicht", "portfolio weight",
        "portfolio weight (%)", "gewicht in %"
    ]

    name_col = next((lower_map[k] for k in name_candidates if k in lower_map), None)
    weight_col = next((lower_map[k] for k in weight_candidates if k in lower_map), None)

    if name_col is None or weight_col is None:
        raise ValueError(
            f"Konnte Name/Weight-Spalten in {source_path} nicht erkennen. "
            f"Spalten: {list(df.columns)}"
        )

    result = df[[name_col, weight_col]].copy()
    result.rename(columns={name_col: "name", weight_col: "weight_pct"}, inplace=True)

    result["weight_pct"] = (
        result["weight_pct"]
        .astype(str)
        .str.replace("’", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .pipe(lambda s: pd.to_numeric(s, errors="coerce") / 100.0)
    )

    result = result.dropna(subset=["weight_pct"])
    return result


def load_invesco_local(isin: str) -> pd.DataFrame:
    """
    Liest Invesco-Holdings aus einer lokalen Datei im Ordner data/.
    Erlaubt:
      - data/invesco_<ISIN>.xlsx  (Excel)
      - data/invesco_<ISIN>.csv   (CSV)
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
        df = pd.read_csv(csv_path)
        source_path = csv_path
    else:
        raise FileNotFoundError(
            f"Keine Datei für Invesco-ETF {isin} gefunden. "
            f"Erwarte data/invesco_{isin}.xlsx oder data/invesco_{isin}.csv"
        )

    lower_map = {c.lower(): c for c in df.columns}

    name_candidates = [
        "name", "titel", "security name", "issuer name",
        "position", "bezeichnung", "holding", "constituent"
    ]
    weight_candidates = [
        "weight (%)", "gewichtung (%)", "gewichtung",
        "gewicht (%)", "gewicht", "portfolio weight",
        "portfolio weight (%)", "gewicht in %"
    ]

    name_col = next((lower_map[k] for k in name_candidates if k in lower_map), None)
    weight_col = next((lower_map[k] for k in weight_candidates if k in lower_map), None)

    if name_col is None or weight_col is None:
        raise ValueError(
            f"Konnte Name/Weight-Spalten in {source_path} nicht erkennen. "
            f"Spalten: {list(df.columns)}"
        )

    result = df[[name_col, weight_col]].copy()
    result.rename(columns={name_col: "name", weight_col: "weight_pct"}, inplace=True)

    result["weight_pct"] = (
        result["weight_pct"]
        .astype(str)
        .str.replace("’", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .pipe(lambda s: pd.to_numeric(s, errors="coerce") / 100.0)
    )

    result = result.dropna(subset=["weight_pct"])
    return result


def load_invesco_local(isin: str) -> pd.DataFrame:
    """
    Liest Invesco-Holdings aus einer lokalen Datei im Ordner data/.
    Erlaubt:
      - data/invesco_<ISIN>.xlsx  (Excel)
      - data/invesco_<ISIN>.csv   (CSV)
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
        df = pd.read_csv(csv_path)
        source_path = csv_path
    else:
        raise FileNotFoundError(
            f"Keine Datei für Invesco-ETF {isin} gefunden. "
            f"Erwarte data/invesco_{isin}.xlsx oder data/invesco_{isin}.csv"
        )

    lower_map = {c.lower(): c for c in df.columns}

    name_candidates = [
        "name", "titel", "security name", "issuer name",
        "position", "bezeichnung", "holding", "constituent"
    ]
    weight_candidates = [
        "weight (%)", "gewichtung (%)", "gewichtung",
        "gewicht (%)", "gewicht", "portfolio weight",
        "portfolio weight (%)", "gewicht in %"
    ]

    name_col = next((lower_map[k] for k in name_candidates if k in lower_map), None)
    weight_col = next((lower_map[k] for k in weight_candidates if k in lower_map), None)

    if name_col is None or weight_col is None:
        raise ValueError(
            f"Konnte Name/Weight-Spalten in {source_path} nicht erkennen. "
            f"Spalten: {list(df.columns)}"
        )

    result = df[[name_col, weight_col]].copy()
    result.rename(columns={name_col: "name", weight_col: "weight_pct"}, inplace=True)

    result["weight_pct"] = (
        result["weight_pct"]
        .astype(str)
        .str.replace("’", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .pipe(lambda s: pd.to_numeric(s, errors="coerce") / 100.0)
    )

    result = result.dropna(subset=["weight_pct"])
    return result



def load_amundi_local(isin: str) -> pd.DataFrame:
    """
    Liest Amundi-Holdings aus einer lokalen Datei im Ordner data/.
    Erlaubt:
      - data/amundi_<ISIN>.xlsx  (Excel)
      - data/amundi_<ISIN>.csv   (CSV)

    Erwartet: irgendeine Spalte mit Name + eine mit Gewicht (%) o.ä.
    """
    # mögliche Dateien
    xlsx_path = DATA_DIR / f"amundi_{isin}.xlsx"
    csv_path = DATA_DIR / f"amundi_{isin}.csv"

    if xlsx_path.exists():
        # Excel einlesen
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Lesen der Amundi-Excel {xlsx_path}: {e}")
        source_path = xlsx_path
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        source_path = csv_path
    else:
        raise FileNotFoundError(
            f"Keine Datei für Amundi-ETF {isin} gefunden. "
            f"Erwarte data/amundi_{isin}.xlsx oder data/amundi_{isin}.csv"
        )

    # Spaltennamen ins Lowercase mappen, damit wir robust suchen können
    lower_map = {c.lower(): c for c in df.columns}

    # Kandidaten für die Namensspalte
    name_candidates = [
        "name",
        "titel",
        "titelname",
        "security name",
        "position",
        "bezeichnung",
        "issuer",
        "issuer name",
    ]

    # Kandidaten für Gewichts-Spalte (in %)
    weight_candidates = [
        "gewichtung (%)",
        "gewichtung%",
        "gewichtung",
        "gewicht (%)",
        "gewicht",
        "weight (%)",
        "weight%",
        "weight",
        "portfolio weight",
        "portfolio weight (%)",
        "gewicht in %",
    ]

    name_col = None
    for key in name_candidates:
        if key in lower_map:
            name_col = lower_map[key]
            break

    weight_col = None
    for key in weight_candidates:
        if key in lower_map:
            weight_col = lower_map[key]
            break

    if name_col is None or weight_col is None:
        raise ValueError(
            f"Konnte Name/Weight-Spalten in {source_path} nicht erkennen. "
            f"Spalten: {list(df.columns)}"
        )

    result = df[[name_col, weight_col]].copy()
    result.rename(columns={name_col: "name", weight_col: "weight_pct"}, inplace=True)

    # Prozent-Spalte in Float 0–1 umwandeln
    result["weight_pct"] = (
        result["weight_pct"]
        .astype(str)
        .str.replace("’", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .pipe(lambda s: pd.to_numeric(s, errors="coerce") / 100.0)
    )

    result = result.dropna(subset=["weight_pct"])

    return result



def compute_lookthrough(portfolio_df: pd.DataFrame):
    """
    Berechnet die Look-Through-Einzelaktien-Gewichte.
    Gibt zwei DataFrames zurück:
      - aggregated: je Aktie Gesamtgewicht im Portfolio
      - detailed: Detailansicht mit ETF-Zuordnung
    """
    all_rows = []

    for _, row in portfolio_df.iterrows():
        provider = row["provider"]
        isin = row["isin"]
        etf_weight = row["weight_in_portfolio"]

        try:
            if provider == "ishares":
                holdings = load_ishares_holdings_from_url(isin)
            elif provider == "amundi":
                holdings = load_amundi_local(isin)
            elif provider == "invesco":
                holdings = load_invesco_local(isin)
            else:
                st.warning(f"Unbekannter Provider: {provider} für {row['name']}")
                continue
        except Exception as e:
            st.error(f"Fehler beim Laden der Holdings für {row['name']} ({isin}): {e}")
            continue

        h = holdings.copy()
        h["lookthrough_weight"] = h["weight_pct"] * etf_weight
        h["etf_name"] = row["name"]
        all_rows.append(h)

    if not all_rows:
        return (
            pd.DataFrame(columns=["name", "weight_in_portfolio"]),
            pd.DataFrame(),
        )

    combined = pd.concat(all_rows, ignore_index=True)

    # gleiche Aktiennamen zusammenfassen
    grouped = combined.groupby("name", as_index=False)["lookthrough_weight"].sum()
    grouped.rename(columns={"lookthrough_weight": "weight_in_portfolio"}, inplace=True)
    grouped.sort_values("weight_in_portfolio", ascending=False, inplace=True)

    return grouped, combined


# --------------------------------------------------------------------------------------
# Streamlit App
# --------------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="ETF Look-Through", layout="wide")
    st.title("ETF Look-Through Analyse – Einzelaktien-Exposure")

    st.markdown(
        """
Diese Version lädt die Holdings **direkt** über die Links der Anbieter:

- iShares-ETFs (WITS, RBOT, WHCS) über die offiziellen CSV-Links
- Amundi & Invesco über lokale CSV-Dateien im Ordner `data/`
"""
    )

    portfolio_path = Path("portfolio.json")
    if not portfolio_path.exists():
        st.error("`portfolio.json` nicht gefunden – bitte Datei im Projektordner anlegen.")
        st.stop()

    portfolio_df = load_portfolio(portfolio_path)

    st.subheader("Dein ETF-Portfolio")
    st.dataframe(
        portfolio_df[["name", "isin", "value_eur", "weight_in_portfolio"]]
        .assign(weight_pct=lambda df: (df["weight_in_portfolio"] * 100).round(2))
        .rename(columns={"weight_pct": "Gewicht im Depot (%)"})
    )

    st.markdown("---")
    st.subheader("Look-Through-Berechnung")

    if st.button("Holdings laden & Look-Through berechnen"):
        aggregated_df, detailed_df = compute_lookthrough(portfolio_df)

        if aggregated_df.empty:
            st.warning("Keine Holdings geladen – bitte Fehlermeldungen oben prüfen.")
            st.stop()

        col1, col2 = st.columns([2, 3])

        with col1:
            st.markdown("### Top-Einzelaktien im Gesamtportfolio")
            st.dataframe(
                aggregated_df.assign(
                    weight_pct=lambda df: (df["weight_in_portfolio"] * 100).round(2)
                )[["name", "weight_pct"]]
                .rename(columns={"weight_pct": "Gewicht im Portfolio (%)"})
                .head(50)
            )

        with col2:
            st.markdown("### Chart: Top-20 Aktien (Gewicht im Portfolio)")
            top20 = aggregated_df.head(20).copy()
            top20["Gewicht (%)"] = (top20["weight_in_portfolio"] * 100).round(2)
            st.bar_chart(
                data=top20.set_index("name")["Gewicht (%)"]
            )

        st.markdown("---")
        st.subheader("Detailansicht: welche Aktien stecken in welchem ETF?")

        if not detailed_df.empty:
            detailed_display = detailed_df.assign(
                gewicht_im_etf_pct=lambda df: (df["weight_pct"] * 100).round(2),
                beitrag_portfolio_pct=lambda df: (df["lookthrough_weight"] * 100).round(3),
            )[["etf_name", "name", "gewicht_im_etf_pct", "beitrag_portfolio_pct"]]

            detailed_display = detailed_display.rename(
                columns={
                    "etf_name": "ETF",
                    "name": "Aktie",
                    "gewicht_im_etf_pct": "Gewicht im ETF (%)",
                    "beitrag_portfolio_pct": "Beitrag zum Gesamtportfolio (%)",
                }
            )

            st.dataframe(detailed_display)

        st.success("Berechnung abgeschlossen.")


if __name__ == "__main__":
    main()
