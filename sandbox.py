from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def plot_patient_expenses_by(df: pd.DataFrame, versnr: int, column: str) -> None:
    """Plot the yearly expenses for a patient by a given column."""
    patient_data = df[df["Versnr"] == versnr]
    grouped_data = (
        patient_data.groupby(["Jahr", column])["Ausgaben"].sum().unstack(fill_value=0)
    )
    ax = grouped_data.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")

    # Set plot labels and title
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Ausgaben (aufsummiert)")
    geburtsjahr = patient_data["Gebjahr"].iloc[0]
    ax.set_title(
        f"Jährliche Ausgaben für Versnr {versnr} (geb. {geburtsjahr}) nach {column}"
    )
    ax.legend(title=f"{column} Code", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def filter_patient_data(df: pd.DataFrame, versnr: int) -> pd.DataFrame:
    """Filter the DataFrame for a specific patient and calculate cumulative expenses."""
    patient_data = df[df["Versnr"] == versnr].copy()
    patient_data["Jahresausgaben"] = patient_data.groupby("Jahr")["Ausgaben"].transform(
        "sum"
    )
    patient_data["Kumulative_Ausgaben"] = patient_data.groupby("Jahr")[
        "Ausgaben"
    ].cumsum()
    return patient_data.reset_index()


def kosten_pro_medikament(
    df: pd.DataFrame, column: List[str], ascending: bool
) -> pd.DataFrame:
    """Berechnet die Gesamtkosten, Anzahl der Verschreibungen und Anzahl der Versicherten pro Medikament und Jahr."""
    df_non_null = df.dropna(subset=column)

    # Gruppieren nach Jahr und Medikament und Aggregieren der Daten
    aggregated_data = (
        df_non_null.groupby(column)
        .agg(
            Gesamtkosten=("Ausgaben", "sum"),
            Anzahl_Verschreibungen=("Medikament", "count"),
            Anzahl_Versicherte=("Versnr", "nunique"),
        )
        .sort_values("Gesamtkosten", ascending=ascending)
    )

    # Berechnung der durchschnittlichen Kosten pro Versicherten und pro Verschreibung
    aggregated_data["Durchschnittliche_Kosten_pro_Verschreibung"] = (
        aggregated_data["Gesamtkosten"] / aggregated_data["Anzahl_Verschreibungen"]
    )
    aggregated_data["Durchschnittliche_Kosten_pro_Versicherten"] = (
        aggregated_data["Gesamtkosten"] / aggregated_data["Anzahl_Versicherte"]
    )

    return aggregated_data


def kosten_pro_eingriff(
    df: pd.DataFrame, column: List[str], ascending: bool
) -> pd.DataFrame:
    """Berechnet die Gesamtkosten, Anzahl der Eingriffe und Anzahl der Versicherten pro Eingriff und Jahr."""
    df_non_null = df.dropna(subset=column)

    # Gruppieren nach Jahr und Eingriff und Aggregieren der Daten
    aggregated_data = (
        df_non_null.groupby(column)
        .agg(
            Gesamtkosten=("Ausgaben", "sum"),
            Anzahl_Eingriffe=("Eingriff", "count"),
            Anzahl_Versicherte=("Versnr", "nunique"),
        )
        .sort_values("Gesamtkosten", ascending=ascending)
    )

    # Berechnung der durchschnittlichen Kosten pro Versicherten und pro Eingriff
    aggregated_data["Durchschnittliche_Kosten_pro_Eingriff"] = (
        aggregated_data["Gesamtkosten"] / aggregated_data["Anzahl_Eingriffe"]
    )
    aggregated_data["Durchschnittliche_Kosten_pro_Versicherten"] = (
        aggregated_data["Gesamtkosten"] / aggregated_data["Anzahl_Versicherte"]
    )

    return aggregated_data


def kosten_pro_diagnose(
    df: pd.DataFrame, column: List[str], ascending: bool
) -> pd.DataFrame:
    """Berechnet die Gesamtkosten, Anzahl der Diagnosen und Anzahl der Versicherten pro Diagnose und Jahr."""
    df_non_null = df.dropna(subset=column)

    # Gruppieren nach Jahr und Diagnose und Aggregieren der Daten
    aggregated_data = (
        df_non_null.groupby(column)
        .agg(
            Gesamtkosten=("Ausgaben", "sum"),
            Anzahl_Diagnosen=("Diagnose", "count"),
            Anzahl_Versicherte=("Versnr", "nunique"),
        )
        .sort_values("Gesamtkosten", ascending=ascending)
    )

    # Berechnung der durchschnittlichen Kosten pro Versicherten und pro Diagnose
    aggregated_data["Durchschnittliche_Kosten_pro_Diagnose"] = (
        aggregated_data["Gesamtkosten"] / aggregated_data["Anzahl_Diagnosen"]
    )
    aggregated_data["Durchschnittliche_Kosten_pro_Versicherten"] = (
        aggregated_data["Gesamtkosten"] / aggregated_data["Anzahl_Versicherte"]
    )

    return aggregated_data
