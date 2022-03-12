import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show
from rasterio.plot import show_hist
from shapely.geometry import Polygon, mapping
from rasterio.mask import mask

# define path to digital terrain model
sjer_dtm_path = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/spatial-vector-lidar/california/neon-soap-site/2013/lidar/SOAP_lidarDTM.tif"

# read in all of the data without specifying a band
with rio.open(sjer_dtm_path) as src:
    # convert / read the data into a numpy array:
    lidar_dem_im = src.read(1, masked=True)
    sjer_ext = rio.plot.plotting_extent(src)

print(lidar_dem_im.shape)
print(sjer_ext)
# BoundingBox(left=296906.0, bottom=4100038.0, right=300198.0, top=4101554.0)

# plot the dem using raster.io
fig, ax = plt.subplots(figsize=(10, 8))
show(lidar_dem_im,
     title="Lidar Digital Elevation Model (DEM) \n Boulder Flood 2013",
     ax=ax)
ax.set_axis_off()

plt.show()
