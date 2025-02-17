#!/usr/bin/python3.12
# coding=utf-8
# %%%
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import numpy as np


def make_geo(
    df_accidents: pd.DataFrame, df_locations: pd.DataFrame
) -> geopandas.GeoDataFrame:
    df = pd.merge(df_accidents, df_locations, on="p1", how="inner")
    df = df[(df["d"].notnull()) & (df["e"].notnull()) & (df["d"] != 0) & (df["e"] != 0)]

    swapped = df["d"] < df["e"]
    df.loc[swapped, ["d", "e"]] = df.loc[swapped, ["e", "d"]].values

    gdf = geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df["d"], df["e"]),
        crs="EPSG:5514",
    )

    return gdf

    # Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani Pozor na mozne prohozeni d a e!


def plot_geo(
    gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False
):
    # Filter accidents involving alcohol in JHM region
    gdf = gdf[(gdf["p11"] >= 4) & (gdf["region"] == "JHM")]

    januaryGDF = gdf[gdf["date"].dt.month == 1]
    decemberGDF = gdf[gdf["date"].dt.month == 12]

    # calculate boundaries for both GDFs
    combined_bounds = geopandas.GeoDataFrame(
        pd.concat([decemberGDF, januaryGDF]), crs=gdf.crs
    ).total_bounds

    padding_x = (combined_bounds[2] - combined_bounds[0]) * 0.05
    padding_y = (combined_bounds[3] - combined_bounds[1]) * 0.05

    padded_bounds = [
        combined_bounds[0] - padding_x,  # minx
        combined_bounds[1] - padding_y,  # miny
        combined_bounds[2] + padding_x,  # maxx
        combined_bounds[3] + padding_y,  # maxy
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    decemberGDF.plot(
        ax=ax1,
        marker="o",
        color="red",
        markersize=5,
        alpha=0.7,
        label="Nehody s prítomnosťou alkoholu",
    )

    ctx.add_basemap(
        ax1, crs=decemberGDF.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik
    )

    ax1.set_title("Nehody s prítomnosťou alkoholu v kraji JHM (December)", fontsize=15)

    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_xlim(padded_bounds[0], padded_bounds[2])
    ax1.set_ylim(padded_bounds[1], padded_bounds[3])

    januaryGDF.plot(
        ax=ax2,
        marker="o",
        color="red",
        markersize=5,
        alpha=0.7,
        label="Nehody s prítomnosťou alkoholu",
    )

    ctx.add_basemap(
        ax2, crs=januaryGDF.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik
    )

    ax2.set_title("Nehody s prítomnosťou alkoholu v kraji JHM (Január)", fontsize=15)

    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.set_xlim(padded_bounds[0], padded_bounds[2])
    ax2.set_ylim(padded_bounds[1], padded_bounds[3])

    ax1.legend()
    ax2.legend()

    if fig_location:
        plt.savefig(fig_location, bbox_inches="tight")

    if show_figure:
        plt.show()


def plot_cluster(
    gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False
):
    gdf = gdf[(gdf["p10"] == 4) & (gdf["region"] == "JHM")]

    # Using k-means for clustering because it's simple and efficient for identifying hotspots
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    kmeans = sklearn.cluster.KMeans(n_clusters=8).fit(coords)

    gdf["cluster"] = kmeans.labels_

    accidents_per_cluster = gdf.groupby("cluster").size()

    polygons = (
        gdf.dissolve(by="cluster").geometry.apply(lambda g: g.convex_hull).reset_index()
    )

    polygons["accident_count"] = polygons["cluster"].map(accidents_per_cluster)

    bounds = geopandas.GeoDataFrame(gdf).total_bounds

    padding_x = (bounds[2] - bounds[0]) * 0.05
    padding_y = (bounds[3] - bounds[1]) * 0.05

    padded_bounds = [
        bounds[0] - padding_x,
        bounds[1] - padding_y,
        bounds[2] + padding_x,
        bounds[3] + padding_y,
    ]

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    cmap = plt.cm.viridis

    polygons.plot(
        ax=ax,
        column="accident_count",
        cmap=cmap,
        alpha=0.5,
        legend=True,
        legend_kwds={
            "label": "Počet nehôd v úseku",
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 30,
            "pad": 0.01,
            "anchor": (0.5, -0.1),
        },
    )

    gdf.plot(
        ax=ax, color="red", markersize=7, alpha=0.7, label="Místa nehod", marker="o"
    )

    ctx.add_basemap(
        ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik
    )

    ax.set_xlim(padded_bounds[0], padded_bounds[2])
    ax.set_ylim(padded_bounds[1], padded_bounds[3])

    ax.set_axis_off()

    ax.set_title("Nehody v Juhomoravskom kraji zavinené lesnou zverou", fontsize=15)
    ax.legend(loc="upper right")

    if fig_location:
        plt.savefig(fig_location, bbox_inches="tight")

    if show_figure:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    df_accidents = pd.read_pickle("accidents.pkl.gz")
    df_locations = pd.read_pickle("locations.pkl.gz")
    gdf = make_geo(df_accidents, df_locations)

    plot_geo(gdf, "geo1.png", False)
    plot_cluster(gdf, "geo2.png", False)

    # testovani splneni zadani
    # import os

    # assert os.path.exists("geo1.png")
    # assert os.path.exists("geo2.png")
