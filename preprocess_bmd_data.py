import pandas as pd

temperature_file = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd " \
                   "Data/Daily Avg Dry-bulb Temperature.ods"
station_info_file = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd Data/weather_station_info.csv"

write_to_csv = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd Data/temp_info.csv"

temp_df = pd.read_excel(temperature_file, engine="odf")
bd_weather_stations = pd.read_csv(station_info_file)

# years = [2016, 2017, 2018, 2019, 2020, 2021]
# col_names = ["Station", "Code", "Latitude", "Longitude"]
#
# for year in years:
#     for month in range(0, 12):
#         for day in range(0, 31):
#             col_name = f"Y_{year}_M_{month}_D_{day}"
#             col_names.append(col_name)

# curated_temp_df = [col_names]
st_code = []
st_latitude = []
st_longitude = []
j = 0
for i in range(0, temp_df.shape[0]):
    j = i // 72
    station_name = bd_weather_stations.iloc[j, 1]
    code = bd_weather_stations.iloc[j, 0]
    latitude = bd_weather_stations.iloc[j, 2]
    longitude = bd_weather_stations.iloc[j, 3]
    temp_df.loc[i, "Station"] = station_name
    st_code.append(code)
    st_latitude.append(latitude)
    st_longitude.append(longitude)

temp_df["Code"] = st_code
temp_df["latitude"] = st_latitude
temp_df["longitude"] = st_longitude

temp_df.to_csv(write_to_csv, sep='\t', encoding='utf-8', index=False)

