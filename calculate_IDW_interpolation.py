import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import pandas as pd
import seaborn as sns
import math
import sklearn.neighbors as skn


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
            dist = (t_lat - r_lat)**2 + (t_lon - r_lon)**2
            if dist < nearest_val:
                nearest_val = dist
                nearest = st_target_removed.iloc[i, 1]
        st_nearest.append(nearest)
    return st_nearest


def get_error_df(st_target_location, st_nearest, temp_info_df):
    # start calculating errors
    error_flat_t = []
    error_all_year_t = []
    error_year_t = []
    error_month_t = []
    # Considering only one target
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
        if target[0] == x:
            return weather_stations.iloc[i].values.flatten().tolist()


def calculate_dist_from_target(target_data, wst_df):
    dist = []
    for j in range(wst_df.shape[0]):
        t_lat = target_data[2]
        t_lon = target_data[3]
        n_lat = wst_df.iloc[j, 2]
        n_lon = wst_df.iloc[j, 3]
        distance = math.sqrt(((t_lat - n_lat) ** 2) + ((t_lon - n_lon) ** 2))
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
    k_neighbours_target = []
    target_data = [get_target_data(target, weather_stations)]
    dist_all_st_from_target = calculate_dist_from_target(target_data[0], weather_stations)
    k_neighbours = get_k_neighbours(dist_all_st_from_target, neighbour_num)
    k_neighbours_target.append(k_neighbours)
    return k_neighbours_target[0]


# Load datasets and set target
weather_station_csv = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd Data/weather_station_info.csv"
temp_info_file = '/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/Bmd Data/temp_info.csv'
bd_weather_stations = pd.read_csv(weather_station_csv)
temp_info_df = pd.read_csv(temp_info_file, sep='\t')

st_target_location = ['Khulna']

neighbors_num = 5
# Only single station is considered
# For multiple station run a loop
neighbours = get_k_neighbours_from_target(st_target_location, bd_weather_stations, neighbors_num)

target_temp_df = temp_info_df.loc[temp_info_df['Station'] == st_target_location[0]]
neighbours_temp_df = temp_info_df.loc[temp_info_df['Station'].isin(neighbours)]

print(target_temp_df.head(5))
print(neighbours_temp_df["Station"])

# # Initialize KNN regressor
# knn_regressor = skn.KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
#
# Fit to data
# knn_regressor.fit(coords_rain_train, value_rain_train)
# neigh_dist, neigh_index = knn_regressor.kneighbors(X=bd_weather_stations.iloc[: , 2:], n_neighbors=neighbors, return_distance=True)
# print(neigh_dist)
# print(neigh_index)
# st_nearest = find_nearest_station(st_target_location, bd_weather_stations)
#
# #get error df
# error_flat_t, error_all_year_t = get_error_df(st_target_location, st_nearest, temp_info_df)
#
# # Whole dataset RMSE result
# rmse_error = get_rmse_error(error_flat_t)
# print(f"RMSE error: {rmse_error} for location {st_target_location[0]} interpolated from {st_nearest[0]}")
#
# # calculate month wise and day wise error
# indexes = np.array(temp_info_df["Year"].drop_duplicates())
# day_colnames = np.arange(1, 367)
# month_colnames = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
#
# month_error_df, m_max, m_min = get_year_month_error(month_colnames, indexes, error_all_year_t)
# day_error_df, d_max, d_min = get_year_day_error(day_colnames, indexes, error_all_year_t)

# ax = sns.heatmap(month_error_df, linewidth=0.5)
# # ax = sns.heatmap(normalized_day_error_df)
# plt.show()

