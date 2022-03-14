import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio as rio
from rasterio.plot import show
import geopandas as gpd
from rasterio.plot import show_hist
from shapely.geometry import Polygon, mapping
from rasterio.mask import mask

admin_shp_path = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/bgd_adm_bbs_shp/bgd_adm_bbs_20201113_SHP/bgd_admbnda_adm2_bbs_20201113.shp"
weather_station_csv = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd Data/weather_station_info.csv"
bd_plot_locations = gpd.read_file(admin_shp_path)
bd_weather_stations = pd.read_csv(weather_station_csv)
print(bd_plot_locations.columns)
print(bd_weather_stations.columns)
print(f"BD ADMIN CRS {bd_plot_locations.crs}")

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(10, 10))
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
#Create custom color map
station_color = ListedColormap('blue')
map_color = ListedColormap('lightgrey')
# create map
bd_plot_locations.plot(column='ADM2_EN', cmap=map_color, linewidth=0.8, ax=ax, edgecolor='.9')
# bd_plot_elev.plot(column='ELEV_M', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, cax=cax)
# bd_plot_locations.geometry.boundary.plot(color=None, edgecolor='k', linewidth=0.8, ax=ax) #Use your second dataframe
plt.scatter(x=bd_weather_stations["Longitude"], y=bd_weather_stations["Latitude"], marker="*", s=64, cmap=station_color)
# plt.scatter(x=91.7, y=22.6, marker="o", cmap=mycolor)
for i, txt in enumerate(bd_weather_stations["Location"]):
    plt.annotate(txt, (bd_weather_stations["Longitude"][i], bd_weather_stations["Latitude"][i]), fontsize=8, fontweight=3)
# remove the axis
ax.axis('off')
# add a title
ax.set_title('BMD Weather Station Locations', fontdict={'fontsize': '25', 'fontweight' : '3'})
plt.savefig("bd_weather_station_locations")
plt.show()