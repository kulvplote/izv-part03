#!/usr/bin/python3.12
# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt


def plot_collision_consequences(df: pd.DataFrame, fig_location: str = None):
    collision_types = {
        1: "čelní",
        2: "boční",
        3: "z boku",
        4: "zezadu",
    }

    consequences = {
        1: "s následky na životě",
        2: "pouze s hmotnou škodou",
    }

    # Filter df to only include valid collision types and consequences
    filtered_df = df[
        (df["p7"].isin(collision_types.keys())) & (df["p9"].isin(consequences.keys()))
    ]

    grouped_counts = filtered_df.groupby(["p7", "p9"]).size().unstack(fill_value=0)

    grouped_counts.index = [collision_types[code] for code in grouped_counts.index]

    grouped_counts.columns = [consequences[code] for code in grouped_counts.columns]

    # Plot the grouped bar chart
    grouped_counts.plot(kind="bar", figsize=(12, 6), color=["lightcoral", "skyblue"])
    plt.xlabel("Druh srážky jedoucích vozidel", fontsize=12)
    plt.ylabel("Počet nehod", fontsize=12)
    plt.title("Počet nehod podle druhu srážky a následků nehody", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Charakter nehody", fontsize=10)
    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location, bbox_inches="tight")
        print(f"Figure saved to {fig_location}")


def head_on_table(df: pd.DataFrame):
    causes = {
        range(201, 210): "nepřiměřená rychlost jízdy",
        range(301, 312): "nesprávné předjíždění",
        range(401, 415): "nedání přednosti v jízdě",
        range(501, 517): "nesprávný způsob jízdy",
        range(601, 616): "technická závada vozidla",
        100: "nezaviněná řidičem",
    }

    consequences = {1: "s následky na životě", 2: "pouze s hmotnou škodou"}

    # Filter only 'čelní srážky' and create a copy to avoid SettingWithCopyWarning
    celni_srazky = df[df["p7"] == 1].copy()

    # Categorize causes
    def categorize_cause(value):
        for key, category in causes.items():
            if isinstance(key, range) and value in key:  # If p12 is within the range
                return category
            elif value == key:  # If p12 matches a specific key
                return category
        return "jiné"  # For uncategorized causes

    celni_srazky["cause_category"] = celni_srazky["p12"].apply(categorize_cause)
    celni_srazky["consequence_category"] = celni_srazky["p9"].map(consequences)

    pivot_table = celni_srazky.pivot_table(
        index="cause_category",
        columns="consequence_category",
        aggfunc="size",
        fill_value=0,
    )

    pivot_table["Celkem"] = pivot_table.sum(axis=1)
    pivot_table.loc["Celkem"] = pivot_table.sum(axis=0)

    return pivot_table


def cause_percentage(df: pd.DataFrame, cause_range: range):
    # Filter data for 'čelní srážky' and 'nedání přednosti v jízdě'
    filtered_df = df[(df["p7"] == 1) & (df["p12"].isin(cause_range))]

    total_accidents = len(filtered_df)
    accidents_na_zivote = len(filtered_df[filtered_df["p9"] == 1])

    percentage = (accidents_na_zivote / total_accidents) * 100

    return round(percentage, 2)


def junction_percentage(df: pd.DataFrame):
    # Filter data for 'čelní srážky' and 'nedání přednosti v jízdě'
    filtered_df = df[(df["p7"] == 1) & (df["p12"].between(401, 414))]

    total_accidents = len(filtered_df)
    relevant_accidents = len(
        filtered_df[filtered_df["p28"].isin([4, 5, 6, 7])]
    )  # Filter for values representing different types of junctions

    percentage = (relevant_accidents / total_accidents) * 100

    return round(percentage, 2)


if __name__ == "__main__":
    df_accidents = pd.read_pickle("accidents.pkl.gz")

    plot_collision_consequences(df_accidents, fig_location="collision_consequences.png")

    print(head_on_table(df_accidents))

    nedani_prednosti = cause_percentage(df_accidents, range(401, 415))
    print(
        f"Percento nehôd s následkami na živote - 'nedání přednosti v jízdě': {nedani_prednosti}%"
    )

    nespravne_predjizdeni = cause_percentage(df_accidents, range(301, 312))
    print(
        f"Percento nehôd s následkami na živote - 'nesprávné předjíždění': {nespravne_predjizdeni}%"
    )

    neprimerana_rychlost = cause_percentage(df_accidents, range(201, 210))
    print(
        f"Percento nehôd s následkami na živote - 'nepřiměřená rychlost jízdy': {neprimerana_rychlost}%"
    )

    junctions = junction_percentage(df_accidents)
    print(
        f"Percento čelných zrážok na križovatkách spôsobených nedaním prednosti v jazde: {junctions}%"
    )
