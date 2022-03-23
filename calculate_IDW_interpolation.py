import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import pandas as pd
import seaborn as sns
import math
import sklearn.neighbors as skn


def get_distance(lat1, lon1, lat2, lon2):
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2

    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r


def get_rmse_error(error_list):
    sq_error = 0.0
    if len(error_list) == 0:
        return 0.0
    for x in error_list:
        sq_error += float(x)**2
    mean_sq_error = sq_error / len(error_list)
    root_mean_sum_error = math.sqrt(mean_sq_error)
    # print(f"Mean sq error is {mean_sq_error} and RMSE is {root_mean_sum_error}")
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
            dist = get_distance(t_lat, t_lon, r_lat, r_lon)
            if dist < nearest_val:
                nearest_val = dist
                nearest = st_target_removed.iloc[i, 1]
        st_nearest.append(nearest)
    return st_nearest


def get_IDW_val(row, column, n_weights, nearest_stations_df_list):
    value = 0
    for i in range(len(nearest_stations_df_list)):
        n_val = nearest_stations_df_list[i].iloc[row, column]
        if pd.isnull(n_val) or n_val == "****":
            return None
        value += n_weights[i] * float(Decimal(n_val))
    return value


def get_error_df(target_station, nearest_stations, n_weights, var_info_df):
    # start calculating errors
    error_flat_t = []
    error_all_year_t = []
    error_year_t = []
    error_month_t = []
    # Considering only one target
    target_station_df = var_info_df.loc[temp_info_df['Station'] == target_station]
    nearest_stations_df_list = []
    for i, x in enumerate(nearest_stations):
        nearest_stations_df_list.append(temp_info_df.loc[temp_info_df['Station'] == nearest_stations[i]])
    min_row_num = target_station_df.shape[0]
    for i in range(len(nearest_stations_df_list)):
        if min_row_num < nearest_stations_df_list[i].shape[0]:
            min_row_num = nearest_stations_df_list[i].shape[0]

    for r in range(min_row_num):
        if r > 0 and r % 12 == 0:
            error_all_year_t.append(error_year_t)
            error_year_t = []
        for c in range(3, 34):
            t_val = target_station_df.iloc[r, c]
            if not pd.isnull(t_val):
                if t_val != "****":
                    n_val = get_IDW_val(r, c, n_weights, nearest_stations_df_list)
                    if n_val is not None:
                        error = float(Decimal(t_val) - Decimal(n_val))
                        error_month_t.append(error)
                        error_flat_t.append(error)
        error_year_t.append(error_month_t)
        error_month_t = []
    error_all_year_t.append(error_year_t)
    return error_flat_t, error_all_year_t


def get_year_day_error(day_colnames, year_indexes, error_all_year_t):
    day_error_df = pd.DataFrame(np.nan, index=year_indexes, columns=day_colnames)
    d_max = -999999
    d_min = 9999999
    for i, y in enumerate(error_all_year_t):
        day_c = 0
        for j, m in enumerate(y):
            for d in m:
                day_error_df.iloc[i, day_c] = d
                day_c += 1
                if d > d_max:
                    d_max = d
                if d < d_min:
                    d_min = d
    return day_error_df, d_max, d_min


def get_year_month_error(month_colnames, year_indexes, error_all_year_t ):
    month_error_df = pd.DataFrame(np.nan, index=year_indexes, columns=month_colnames)
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
    return month_error_df, m_max, m_min


def get_target_data(target, weather_stations):
    for i, x in enumerate(weather_stations["Location"]):
        if target == x:
            return weather_stations.iloc[i].values.flatten().tolist()


def calculate_dist_from_target(target_data, wst_df):
    dist = []
    for j in range(wst_df.shape[0]):
        t_lat = target_data[2]
        t_lon = target_data[3]
        n_lat = wst_df.iloc[j, 2]
        n_lon = wst_df.iloc[j, 3]
        distance = get_distance(t_lat, t_lon, n_lat, n_lon)
        dist_row = [target_data[1], wst_df.iloc[j, 1], distance]
        dist.append(dist_row)
    return dist


def get_k_neighbours(dist_from_target_to_all, neighbours_num):
    k_neighbours = []
    temp_df = pd.DataFrame(dist_from_target_to_all, columns=["Source", "Destination", "Distance"])
    temp_df = temp_df.sort_values("Distance")
    for j in range(1, neighbours_num+1):
        k_neighbours.append(temp_df.iloc[j][1])
    return k_neighbours


def get_k_neighbours_from_target(target, weather_stations, neighbour_num):
    target_data = get_target_data(target[0], weather_stations)
    dist_all_st_from_target = calculate_dist_from_target(target_data, weather_stations)
    k_neighbours = get_k_neighbours(dist_all_st_from_target, neighbour_num)
    return k_neighbours


def get_weight_vector(target, neighbour_stations, weather_stations):
    dist_vec = [0]*len(neighbour_stations)
    weight_vec = [0]*len(neighbour_stations)
    target_data = get_target_data(target[0], weather_stations)
    sum = 0
    for i, x in enumerate(neighbour_stations):
        n_data = get_target_data(x, weather_stations)
        t_lat = target_data[2]
        t_lon = target_data[3]
        n_lat = n_data[2]
        n_lon = n_data[3]
        distance = get_distance(t_lat, t_lon, n_lat, n_lon)
        dist_vec[i] = distance
        sum += distance
    weight_sum = 0
    for i in range(len(dist_vec)):
        weight_vec[i] = sum/dist_vec[i]
        weight_sum += weight_vec[i]
    for i in range(len(weight_vec)):
        weight_vec[i] = weight_vec[i]/weight_sum
    return weight_vec


# Load datasets and set target
weather_station_csv = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd Data/weather_station_info.csv"
temp_info_file = '/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/Bmd Data/temp_info.csv'
bd_weather_stations = pd.read_csv(weather_station_csv)
temp_info_df = pd.read_csv(temp_info_file, sep='\t')

st_target_location = ['Mymensingh']

neighbors_num = 5
# Only single station is considered
# For multiple station run a loop
neighbours = get_k_neighbours_from_target(st_target_location, bd_weather_stations, neighbors_num)

target_temp_df = temp_info_df.loc[temp_info_df['Station'] == st_target_location[0]]

neighbour_weights = get_weight_vector(st_target_location, neighbours, bd_weather_stations)

error_flat_t, error_all_year_t = get_error_df(st_target_location[0], neighbours, neighbour_weights, temp_info_df)

# Whole dataset RMSE result
rmse_error = get_rmse_error(error_flat_t)
print(f"RMSE error: {rmse_error} for location {st_target_location[0]} interpolated from {neighbours} ")

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

d_spread = d_max - d_min
m_spread = m_max - m_min

print(f"Day: Max {d_max} Min {d_min} and spread {d_spread}")
print(f"Month: Max {m_max} Min {m_min} and spread {m_spread}")

# ###### Were working here ############
# for i in range(len(normalized_month_error_df)):
#     for j in range(len(normalized_month_error_df.iloc[0])):
#         normalized_month_error_df.iloc[i, j] = (normalized_month_error_df.iloc[i, j] - m_min)/m_spread

# print(normalized_month_error_df.head())
ax = sns.heatmap(normalized_day_error_df, cmap="PiYG", center=0)
# ax = sns.heatmap(normalized_day_error_df, linewidth=0.5)
ax.set_title(f'Nearest Neighbour Daily Temperature Estimation Error \n Target: {st_target_location[0]} Neighbours: {neighbours}', fontdict={'fontsize': '20', 'fontweight' : '3'})
plt.savefig("daily_error_Bhola_Barishal")

plt.show()
