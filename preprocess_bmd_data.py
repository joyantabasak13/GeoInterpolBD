import pandas as pd

temperature_file = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd " \
                   "Data/Daily Avg Dry-bulb Temperature.ods"
df = pd.read_excel(temperature_file, engine="odf")
print(df["Station"])

