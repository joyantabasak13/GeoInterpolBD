import pandas as pd
import re
import csv

min_temp_file = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd Data/Daily Minimum Temperature.ods"
write_csv_file = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd Data/weather_station_info.csv"
df = pd.read_excel(min_temp_file, engine="odf", header=None)
print(df.shape[0])
ws_name = []
ws_code = []
ws_lat = []
ws_lon = []
i = 0
while i < df.shape[0]:
    line = re.split(r' +', df.iloc[i, 0])
    code = int(df.iloc[i+5, 0])
    print(f"{line[4]} {line[5]}")
    print(f"Lat {len(line[4])} and Lon {len(line[5])}")
    degree = float(line[4][4:6])
    if len(line[4]) == 17:
        minutes = float(line[4][10:12])
    elif len(line[4]) == 16:
        minutes = float(line[4][10:11])
    else:
        minutes = 0.0
    lat = degree + minutes/60
    print(f"LAT Degree {degree} and Minutes {minutes} ")
    degree = float(line[5][5:7])

    if len(line[5]) == 18:
        minutes = float(line[5][11:13])
    elif len(line[5]) == 17:
        minutes = float(line[5][11:12])
    else:
        minutes = 0.0
    long = degree + minutes / 60
    print(f"Long Degree {degree} and Minutes {minutes} ")

    ws_lat.append(lat)
    ws_lon.append(long)
    ws_name.append(line[3])
    ws_code.append(code)
    i = i + 75

rows = zip(ws_code, ws_name, ws_lat, ws_lon)

with open(write_csv_file, "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
