import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import rasterio as rio
from rasterio.plot import show
import geopandas as gpd
import georaster
from rasterio.plot import show_hist
from shapely.geometry import Polygon, mapping
from rasterio.mask import mask
from raster2xyz.raster2xyz import Raster2xyz

# define path to digital terrain model
admin_shp_path = '/home/joyanta/Documents/MSc/weather data interpolation/GeoInterpolBD/data/bgd_adm_bbs_SHP/bgd_admbnda_adm2_bbs_20201113.shp'
bd_plot_locations = gpd.read_file(admin_shp_path)

tif_file_path = "/home/joyanta/Documents/MSc/weather data interpolation/GeoInterpolBD/data/srtm_54_07/srtm_54_07.tif"
elev_shape_path = "/home/joyanta/Documents/MSc/weather data interpolation/GeoInterpolBD/data/bgd_elevation/bgd_elev.shp"
bd_plot_elev = gpd.read_file(elev_shape_path)
# out_csv = "demo_out_xyz.csv"
# rtxyz = Raster2xyz()
# rtxyz.translate(tif_file_path, out_csv)
# myRasterDF = pd.read_csv(out_csv)

print(bd_plot_locations.columns)
print(bd_plot_elev.columns)
print(bd_plot_elev.head(5))
print(f"BD ADMIN CRS {bd_plot_locations.crs}")
print(f"BD ELEV CRS {bd_plot_elev.crs}")
print(f"Admin Shape {bd_plot_locations.shape}")
print(f" Elev shape {bd_plot_elev.shape}")
# print(myRasterDF.shape)
# print(myRasterDF.head(5))
# image_plot = plt.imshow(myRasterDF)

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(10, 10))
divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
#Create custom color map
mycolor = ListedColormap('blue')
# create map
# bd_plot_locations.plot(column='ADM2_EN', facecolor='none', linewidth=0.8, ax=ax, edgecolor='0.8')
# bd_plot_elev.plot(column='ELEV_M', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, cax=cax)
# bd_plot_locations.geometry.boundary.plot(color=None, edgecolor='k', linewidth=0.8, ax=ax) #Use your second dataframe

# remove the axis
ax.axis('off')
# add a title
ax.set_title('Districts Of Bangladesh', fontdict={'fontsize': '25', 'fontweight' : '3'})

plt.show()

