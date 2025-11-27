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
# Länder-Koordinaten (Zentroid) für die Kartenansicht
# --------------------------------------------------------------------------------------
COUNTRY_COORDS = {
    "USA": (37.1, -95.7),
    "Vereinigte Staaten": (37.1, -95.7),

    "Deutschland": (51.1657, 10.4515),
    "Frankreich": (46.2276, 2.2137),
    "Großbritannien": (55.3781, -3.4360),
    "Vereinigtes Königreich": (55.3781, -3.4360),
    "Niederlande": (52.1326, 5.2913),

    "Österreich": (47.5162, 14.5501),
    "Schweiz": (46.8182, 8.2275),
    "Schweden": (60.1282, 18.6435),
    "Italien": (41.8719, 12.5674),

    "Japan": (36.2048, 138.2529),
    "Australien": (-25.2744, 133.7751),
    "China": (35.8617, 104.1954),
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
      - Ab Zeile 2: CSV mit Header
        "Emittententicker,Name,Sektor,Anlageklasse,Marktwert,Gewichtung (%),...,Standort,..."
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

    # Gewichtung-Spalte
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
        .str.replace(",", ".", regex=False)  # 19,01 -> 19.01 (falls irgendwo vorkommt)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")

    # hier sind es bei iShares wirklich Prozentwerte (z.B. 19.01)
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

    # Kandidaten für die Namensspalte – "full name" neu mit drin
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

    result = df[[name_col, weight_col]].copy()
    result.rename(columns={name_col: "name", weight_col: "weight_pct"}, inplace=True)

    # Country-Spalte (falls vorhanden)
    country_col = next(
        (lower_map[k] for k in ["land", "country", "domicile", "issuer country"] if k in lower_map),
        None,
    )
    if country_col is not None:
        result["country"] = df[country_col]
    else:
        result["country"] = pd.NA

    # Text bereinigen -> numerisch
    cleaned = (
        result["weight_pct"]
        .astype(str)
        .str.replace("’", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(",", ".", regex=False)   # 0,05158 -> 0.05158
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")

    # Heuristik:
    # - Wenn max > 1.5 -> Prozentangaben (z.B. 5.43 = 5,43 % -> /100)
    # - Wenn max <= 1.5 -> bereits normierte Anteile (z.B. 0.0515)
    max_val = numeric.max(skipna=True)
    if pd.notna(max_val) and max_val > 1.5:
        numeric = numeric / 100.0

    result["weight_pct"] = numeric
    result = result.dropna(subset=["weight_pct"])

    return result


def load_amundi_local(isin: str) -> pd.DataFrame:
    """
    Liest Amundi-Holdings aus einer lokalen Datei im Ordner data/.

    Erlaubt:
      - data/amundi_<ISIN>.xlsx  (Excel)
      - data/amundi_<ISIN>.csv   (CSV)

    Unterstützt sowohl alte Prozent-Formate (z.B. "5,43" oder "5.43 %")
    als auch das neue Format mit Anteilen (z.B. "0,054629263").

    Beispiel-Amundi-CSV (dein neues Format):
      ISIN;Name;Anlageklasse;Währung;Gewichtung;Sektor;Land
      US92338C1036;VERALTO CORP;EQUITY;USD;0,054629263;Industrie;USA
      ...
    """
    xlsx_path = DATA_DIR / f"amundi_{isin}.xlsx"
    csv_path = DATA_DIR / f"amundi_{isin}.csv"

    if xlsx_path.exists():
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Lesen der Amundi-Excel {xlsx_path}: {e}")
        source_path = xlsx_path

    elif csv_path.exists():
        # CSV: Semikolon, Python-Engine, kaputte Zeilen überspringen
        try:
            try:
                df = pd.read_csv(
                    csv_path,
                    sep=";",
                    engine="python",
                    on_bad_lines="skip",  # pandas >= 1.3
                )
            except TypeError:
                df = pd.read_csv(
                    csv_path,
                    sep=";",
                    engine="python",
                    error_bad_lines=False,
                    warn_bad_lines=True,
                )
        except Exception as e:
            raise RuntimeError(f"Fehler beim Lesen der Amundi-CSV {csv_path}: {e}")
        source_path = csv_path
    else:
        raise FileNotFoundError(
            f"Keine Datei für Amundi-ETF {isin} gefunden. "
            f"Erwarte data/amundi_{isin}.xlsx oder data/amundi_{isin}.csv"
        )

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

    # Kandidaten für Gewichts-Spalte (in % oder als Anteil)
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

    # Länderspalte
    country_candidates = [
        "land",
        "country",
        "issuer country",
        "domicile",
        "domizilland",
    ]

    name_col = next((lower_map[k] for k in name_candidates if k in lower_map), None)
    weight_col = next((lower_map[k] for k in weight_candidates if k in lower_map), None)
    country_col = next((lower_map[k] for k in country_candidates if k in lower_map), None)

    if name_col is None or weight_col is None:
        raise ValueError(
            f"Konnte Name/Weight-Spalten in {source_path} nicht erkennen. "
            f"Spalten: {list(df.columns)}"
        )

    cols = [name_col, weight_col]
    if country_col is not None:
        cols.append(country_col)

    result = df[cols].copy()
    result.rename(columns={name_col: "name", weight_col: "weight_pct"}, inplace=True)

    if country_col is not None:
        result.rename(columns={country_col: "country"}, inplace=True)
    else:
        result["country"] = pd.NA

    # Gewicht-Spalte bereinigen
    cleaned = (
        result["weight_pct"]
        .astype(str)
        .str.replace("’", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")

    # Heuristik:
    # - Wenn max > 1.5 -> Prozentangaben (z.B. 5.43 = 5,43 % -> /100)
    # - Wenn max <= 1.5 -> bereits normierte Anteile (z.B. 0.0546)
    max_val = numeric.max(skipna=True)
    if pd.notna(max_val) and max_val > 1.5:
        numeric = numeric / 100.0

    result["weight_pct"] = numeric
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
    st.title("ETF Look-Through Analyse – Einzelaktien- & Länder-Exposure")

    st.markdown(
        """
Diese Version lädt die Holdings **direkt** über die Links der Anbieter:

- iShares-ETFs (WITS, RBOT, WHCS) über die offiziellen CSV-Links
- Amundi & Invesco über lokale CSV-Dateien im Ordner `data/`

Zusätzlich:
- Aggregation nach **Ländern**
- Weltkarte mit der Verteilung nach Ländern (Look-Through-Gewicht)
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

        # ----------------------------------------------------------------------------------
        # Länder-Statistik & Karte
        # ----------------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Länder-Statistik (Look-Through)")

        if "country" not in detailed_df.columns:
            st.info("Keine Länderdaten vorhanden (Spalte 'country').")
        else:
            country_df = (
                detailed_df.copy()
                .assign(
                    country=lambda df: df["country"].fillna("Unbekannt"),
                )
                .groupby("country", as_index=False)["lookthrough_weight"]
                .sum()
            )

            country_df["weight_pct"] = (country_df["lookthrough_weight"] * 100).round(2)

            st.markdown("### Gewicht nach Land")
            st.dataframe(
                country_df[["country", "weight_pct"]]
                .rename(
                    columns={
                        "country": "Land",
                        "weight_pct": "Gewicht im Portfolio (%)",
                    }
                )
                .sort_values("Gewicht im Portfolio (%)", ascending=False)
            )

            # Karte vorbereiten
            map_rows = []
            for _, row in country_df.iterrows():
                land = row["country"]
                weight = row["weight_pct"]
                coords = COUNTRY_COORDS.get(land)
                if coords is None:
                    continue
                lat, lon = coords
                map_rows.append(
                    {
                        "Land": land,
                        "lat": lat,
                        "lon": lon,
                        "Gewicht im Portfolio (%)": weight,
                    }
                )

            if map_rows:
                map_df = pd.DataFrame(map_rows)
                st.markdown("### Weltkarte: Ländergewichtung (Look-Through)")
                st.map(
                    map_df,
                    latitude="lat",
                    longitude="lon",
                )
            else:
                st.info("Für keine Länder konnten Koordinaten gefunden werden (COUNTRY_COORDS erweitern?).")

        st.success("Berechnung abgeschlossen.")


if __name__ == "__main__":
    main()
