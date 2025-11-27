import json
from pathlib import Path

import pandas as pd
import streamlit as st
from etf_scraper import ETFScraper  # für iShares & Invesco


DATA_DIR = Path("data")


# -------------------------
# Hilfsfunktionen
# -------------------------

def load_portfolio(path: Path) -> pd.DataFrame:
    """Lädt dein Portfolio aus portfolio.json und berechnet ETF-Gewichte im Depot."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    total = df["value_eur"].sum()
    df["weight_in_portfolio"] = df["value_eur"] / total
    return df


def fetch_holdings_ishares_invesco(ticker: str, scraper: ETFScraper) -> pd.DataFrame:
    """
    Nutzt etf_scraper, um die aktuellen Holdings für iShares/Invesco zu laden.
    ticker: Börsenticker bei iShares/Invesco, z.B. WITS, BCHN, RBOT...
    """
    df = scraper.query_holdings(ticker, holdings_date=None)

    # Spaltennamen vereinheitlichen
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    # Gewichtsspalte erkennen
    if "weight" in df.columns:
        weight_col = "weight"
    elif "weight_percent" in df.columns:
        weight_col = "weight_percent"
    else:
        raise ValueError(f"Keine Gewichtsspalte in Holdings für {ticker} gefunden")

    name_col = "name" if "name" in df.columns else "security_name"

    result = df[[name_col, weight_col]].copy()
    result.rename(columns={name_col: "name", weight_col: "weight_pct"}, inplace=True)

    # Wenn Werte > 1.5, ist es wahrscheinlich in Prozent (0–100), sonst schon 0–1
    if result["weight_pct"].max() > 1.5:
        result["weight_pct"] = result["weight_pct"] / 100.0

    return result


def load_holdings_amundi(isin: str) -> pd.DataFrame:
    """
    Für Amundi: CSV-Datei aus Ordner data/amundi_<ISIN>.csv einlesen,
    die du vorher von der Amundi-Website heruntergeladen hast.
    Erwartet: Spalten für Name & Gewicht (%).
    """
    csv_path = DATA_DIR / f"amundi_{isin}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Keine CSV für Amundi-ETF {isin} gefunden. "
            f"Erwarte Datei: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    possible_name_cols = ["name", "titel", "titelname", "security", "title"]
    possible_weight_cols = ["gewicht", "weight", "gewichtung", "gewicht_%", "weight_pct"]

    name_col = next((c for c in possible_name_cols if c in df.columns), None)
    weight_col = next((c for c in possible_weight_cols if c in df.columns), None)

    if name_col is None or weight_col is None:
        raise ValueError(
            f"Konnte Name/Weight-Spalten in {csv_path} nicht erkennen. "
            f"Bitte Spaltennamen prüfen."
        )

    result = df[[name_col, weight_col]].copy()
    result.rename(columns={name_col: "name", weight_col: "weight_pct"}, inplace=True)

    if result["weight_pct"].max() > 1.5:
        result["weight_pct"] = result["weight_pct"] / 100.0

    return result


def compute_lookthrough(portfolio_df: pd.DataFrame, scraper: ETFScraper):
    """
    Berechnet die Look-Through-Einzelaktien-Gewichte.
    Gibt zwei DataFrames zurück:
      - aggregated: je Aktie Gesamtgewicht im Portfolio
      - detailed: Detailansicht mit ETF-Zuordnung
    """
    all_rows = []

    for _, row in portfolio_df.iterrows():
        provider = row["provider"]
        ticker = row["ticker"]
        isin = row["isin"]
        etf_weight = row["weight_in_portfolio"]

        try:
            if provider in ["ishares", "invesco"]:
                if not ticker:
                    st.warning(f"Kein Ticker für {row['name']} angegeben – übersprungen.")
                    continue
                holdings = fetch_holdings_ishares_invesco(ticker, scraper)
            elif provider == "amundi":
                holdings = load_holdings_amundi(isin)
            else:
                st.warning(f"Unbekannter Provider: {provider} für {row['name']}")
                continue
        except Exception as e:
            st.error(f"Fehler beim Laden der Holdings für {row['name']} ({isin}): {e}")
            continue

        holdings = holdings.copy()
        holdings["lookthrough_weight"] = holdings["weight_pct"] * etf_weight
        holdings["etf_name"] = row["name"]

        all_rows.append(holdings)

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


# -------------------------
# Streamlit App
# -------------------------

def main():
    st.set_page_config(page_title="ETF Look-Through", layout="wide")
    st.title("ETF Look-Through Analyse – Einzelaktien-Exposure")

    st.markdown(
        """
Diese App liest dein ETF-Portfolio ein, lädt (wo möglich) die aktuellen Holdings 
der ETFs und berechnet, welche **Einzelaktien** du effektiv hältst.

- iShares & Invesco: Holdings über `etf_scraper`
- Amundi: Holdings aus CSV-Dateien im Ordner `data/`
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
        scraper = ETFScraper()

        aggregated_df, detailed_df = compute_lookthrough(portfolio_df, scraper)

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
