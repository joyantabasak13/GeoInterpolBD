import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import pandas as pd
import seaborn as sns
import math


def get_rmse_error(error_list):
    sq_error = 0.0
    if len(error_list) == 0:
        return 0.0
    for x in error_list:
        sq_error += float(x)**2
    mean_sq_error = sq_error / len(error_list)
    root_mean_sum_error = math.sqrt(mean_sq_error)
    print(f"Mean sq error is {mean_sq_error} and RMSE is {root_mean_sum_error}")
    return root_mean_sum_error


def find_nearest_station(st_target, st_all):
    st_nearest = []
    for t in st_target:
        nearest = ''
        nearest_val = 99999.0
        st_target_entry = st_all.loc[st_all['Location'] == t]
        st_target_entry_index = st_all.loc[st_all['Location'] == t].index
        st_target_removed = st_all.drop(st_target_entry_index)
        for i in range(st_target_removed.shape[0]):
            t_lat = float(st_target_entry.iloc[0, 2])
            t_lon = float(st_target_entry.iloc[0, 3])
            r_lat = float(st_target_removed.iloc[i, 2])
            r_lon = float(st_target_removed.iloc[i, 3])
            dist = (t_lat - r_lat)**2 + (t_lon - r_lon)**2
            if dist < nearest_val:
                nearest_val = dist
                nearest = st_target_removed.iloc[i, 1]
        st_nearest.append(nearest)
    return st_nearest

# Load datasets and set target
weather_station_csv = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd Data/weather_station_info.csv"
temp_info_file = '/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/Bmd Data/temp_info.csv'
bd_weather_stations = pd.read_csv(weather_station_csv)
temp_info_df = pd.read_csv(temp_info_file, sep='\t')
st_target_location = ['Khulna']
st_nearest = find_nearest_station(st_target_location, bd_weather_stations)

# start calculating errors
error_flat_t = []
error_all_year_t = []
error_year_t = []
error_month_t = []
#Considering only one target
for i, t in enumerate(st_target_location):
    st_t_df = temp_info_df.loc[temp_info_df['Station'] == t]
    st_nearest_t = st_nearest[i]
    st_nearest_t_df = temp_info_df.loc[temp_info_df['Station'] == st_nearest_t]

    for r in range(min(st_t_df.shape[0], st_nearest_t_df.shape[0])):
        if r > 0 and r % 12 == 0:
            error_all_year_t.append(error_year_t)
            error_year_t = []
        for c in range(3, 34):
            t_val = st_t_df.iloc[r, c]
            n_val = st_nearest_t_df.iloc[r, c]
            is_not_null = not (pd.isnull(t_val) or pd.isnull(n_val))
            is_not_star = True
            if t_val == "****" or n_val == "****":
                is_not_star = False
            if is_not_star and is_not_null:
                error = float(Decimal(t_val) - Decimal(n_val))
                # print(f"temp in {st_target_location[i]} is {t_val} and temp in {st_nearest[i]} is {n_val}")
                # print(f"temp Error is {error}")
                error_month_t.append(error)
                error_flat_t.append(error)
        error_year_t.append(error_month_t)
        error_month_t = []
    error_all_year_t.append(error_year_t)
    error_year_t = []

# Whole dataset RMSE result
# rmse_error = get_rmse_error(error_flat_t)

# calculate month wise and day wise error
indexes = np.array(temp_info_df["Year"].drop_duplicates())
day_colnames = np.arange(1, 367)
month_colnames = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

day_error_df = pd.DataFrame(np.nan, index=indexes, columns=day_colnames)
month_error_df = pd.DataFrame(np.nan, index=indexes, columns=month_colnames)

d_max = -999999
d_min = 9999999
m_max = -999999
m_min = 9999999

for i, y in enumerate(error_all_year_t):
    day_c = 0
    for j, m in enumerate(y):
        month_error_df.iloc[i, j] = get_rmse_error(m)
        if month_error_df.iloc[i, j] > m_max:
            m_max = month_error_df.iloc[i, j]
        if month_error_df.iloc[i, j] < m_min:
            m_min = month_error_df.iloc[i, j]

        for d in m:
            day_error_df.iloc[i, day_c] = d
            day_c += 1
            if d > d_max:
                d_max = d
            if d < d_min:
                d_min = d

# print(f"Monthly max {m_max} and min {m_min}")
# print(f"Daily max {d_max} or {max(error_flat_t)} and min {d_min} or {min(error_flat_t)}")

normalized_day_error_df = day_error_df.copy(deep=True)
normalized_month_error_df = month_error_df.copy(deep=True)
# print(f"largest val in df {normalized_month_error_df.max()} and min {normalized_month_error_df.min()}")

###### Were working here ############
for i, y in enumerate(normalized_month_error_df):
    for j, m in enumerate(y):
        normalized_month_error_df.iloc[i,j] =

normalized_month_error_df=(normalized_month_error_df-normalized_month_error_df.min())/(normalized_month_error_df.max()-normalized_month_error_df.min())
# print(normalized_month_error_df.head())


# df = pd.DataFrame(data, columns=['A', 'B', 'C'])
#
# np_temp = np.array([np.array(xi) for xi in x])
# ax = sns.heatmap(uniform_data, linewidth=0.5)
# plt.show()

