from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import shapely
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


admin_BD_shp_path = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/bgd_adm_bbs_shp/bgd_adm_bbs_20201113_SHP/bgd_admbnda_adm0_bbs_20201113.shp"
weather_station_csv = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd Data/weather_station_info.csv"
bd_plot_country = gpd.read_file(admin_BD_shp_path)
bd_weather_stations = pd.read_csv(weather_station_csv)

region = gpd.GeoDataFrame(geometry=bd_plot_country["geometry"], crs=bd_plot_country.crs)
x = bd_weather_stations["Longitude"]
y = bd_weather_stations["Latitude"]
coords = np.vstack((x, y)).T
coords = np.append(coords, [[999.0, 999.0], [-999.0, 999.0], [999.0, -999.0], [-999.0, -999.0]], axis=0)

vor = Voronoi(coords)


lines = [shapely.geometry.LineString(vor.vertices[line]) for line in
         vor.ridge_vertices if -1 not in line]

# print(lines)
polys = shapely.ops.polygonize(lines)
print(polys)
voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys), crs=region.crs)
print(voronois.shape)
voronois = gpd.overlay(voronois, region)
voronois["area"] = voronois['geometry'].area / 10**6

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

bd_plot_country.plot(ax=ax, color="white", edgecolor='k', alpha=0.5)
voronois.plot(column="area", ax=ax, cmap="Blues_r", edgecolor='k', alpha=0.5)
bd_weather_stations = bd_weather_stations.sort_values(by=["Longitude"], ascending=False)
plt.scatter(x=bd_weather_stations["Longitude"], y=bd_weather_stations["Latitude"], marker="*", s=64, edgecolors="k")
# for i, txt in enumerate(bd_weather_stations["Location"]):
#     plt.annotate(txt, (bd_weather_stations["Longitude"][i], bd_weather_stations["Latitude"][i]), fontsize=8, fontweight=3)


plt.show()
