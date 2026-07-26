"""
    Purpose: defines telemetry data and fills in missing data, if required. Computes necessary parameters useful for training the neutral network that may not necessarily be measured by the hardware. Finally, it visualizes the data through Plotly
    Main takeaway (Jul 26): raw telemetry data from telemetry_dumps/telemetry_001.csv is not enough on it's own; need to compute concepts such as Power, delta-time, and Consumption (how much energy is used per second)
"""
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np

pd.options.plotting.backend = 'plotly'
# pio.templates.default = 'plotly_dark'

df = pd.read_csv('BE_sample_data.csv', sep=',', low_memory=False)
df.columns = [col.lower() for col in df.columns]
df.reset_index(drop=True, inplace=True)
df.ffill(inplace=True)
df.head(10)
df.plot(x='gps_longitude', y='gps_latitude')
df.plot(x='gps_longitude', y='gps_latitude', color='lap_lap')
df.loc[df.lap_lap.isin([1, 4])].plot(x='lap_dist', y='gps_speed', color='lap_lap')
df.loc[df.lap_lap.isin([1, 4])].plot(x='gps_longitude', y='gps_latitude', color='jm3_current', kind='scatter', facet_col='lap_lap')
df['jm3_power'] = df['jm3_voltage']*df['jm3_current']/10**6

df.plot(x='lap_dist', y='jm3_power', color='lap_lap')

#initalising new collum:
df['jm3_power_alternative'] = .0

window_len = 2

#manually adding data to our new collum
for ind in range(window_len, len(df)):
    ind_prev = ind-window_len
    df.loc[ind, 'jm3_power_alternative'] = (df.loc[ind, 'jm3_netjoule'] - df.loc[ind_prev, 'jm3_netjoule']) / (df.loc[ind, 'lap_obc_timestamp'] - df.loc[ind_prev, 'lap_obc_timestamp'])
df.plot(x='lap_dist', y='jm3_power_alternative', color='lap_lap')

df['acceleration'] = 0.
df['consumption'] = 0.

window_len = 50

for ind in range(window_len, len(df)):
    ind_prev = ind-window_len
    df.loc[ind, 'acceleration'] = (df.loc[ind, 'gps_speed'] - df.loc[ind_prev, 'gps_speed'])/(df.loc[ind, 'obc_timestamp'] - df.loc[ind_prev, 'obc_timestamp'])
    df.loc[ind, 'consumption'] = (df.loc[ind, 'jm3_netjoule'] - df.loc[ind_prev, 'jm3_netjoule'])/(df.loc[ind, 'obc_timestamp'] - df.loc[ind_prev, 'obc_timestamp'])

for col in df.columns:
    print(col)