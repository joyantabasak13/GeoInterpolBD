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

# Load datasets and set target
weather_station_csv = "/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/data/Bmd Data/weather_station_info.csv"
temp_info_file = '/home/joyanta/Documents/HDSS_Documents/Weather Data Interpolation/GeoInterpolBD/Bmd Data/temp_info.csv'
bd_weather_stations = pd.read_csv(weather_station_csv)
temp_info_df = pd.read_csv(temp_info_file, sep='\t')
st_target_location = ['Mymensingh']
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
rmse_error = get_rmse_error(error_flat_t)
print(f"RMSE error: {rmse_error} for location {st_target_location[0]} interpolated from {st_nearest[0]} ")
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
ax.set_title(f'Nearest Neighbour Daily Temperature Estimation Error \n Target: {st_target_location} Neighbours: {st_nearest}', fontdict={'fontsize': '20', 'fontweight' : '3'})
# plt.savefig("daily_error_Bhola_Barishal")

plt.show()

# ValueError: 'PiBu' is not a valid value for name; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
