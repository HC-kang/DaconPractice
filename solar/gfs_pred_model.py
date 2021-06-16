import numpy as np
import pandas as pd
import pvlib
from pvlib import forecast
print(pvlib.__version__)
import warnings
import datetime as dt
warnings.filterwarnings('ignore')

#visualization
import matplotlib.pyplot as plt


# set location and timezone of target site
lat, long, tz = 37.05075270, 126.5102993, 'Asia/Seoul'
# set time period of target data
start = pd.Timestamp(dt.date.today(), tz='Asia/Seoul')
end = start + pd.Timedelta(days=5)

print(start,end)

# get GFS forecast
GFSmodel = forecast.GFS()
GFSfcst = GFSmodel.get_data(lat, long,
                            start=start,
                            end=end)

GFSfcst = GFSfcst.resample('1h').interpolate()
GFSfcst = GFSmodel.process_data(GFSfcst)

print(GFSfcst.head())
weather = GFSfcst[['temp_air','wind_speed']] 
cloud = GFSfcst[['total_clouds', 'low_clouds',
              'mid_clouds', 'high_clouds']]
irr = GFSfcst[['ghi','dni','dhi']]

weather.plot(figsize=(12,5))
plt.ylabel('Value')
plt.xlabel('Forecast Time ($UTC+9$)')
plt.title(f'GFS 0.25 degree forecast for lat={lat:.4f}, lon={long:.4f}')
plt.legend()
plt.tight_layout()
plt.show()

cloud.plot(figsize=(12,5))
plt.ylabel('Value')
plt.xlabel('Forecast Time ($UTC+9$)')
plt.title(f'GFS 0.25 degree forecast for lat={lat:.4f}, lon={long:.4f}')
plt.legend()
plt.tight_layout()
plt.show()

irr.plot(figsize=(12,5))
plt.ylabel('Irradiance ($W/m^2$)')
plt.xlabel('Forecast Time ($UTC+9$)')
plt.title(f'GFS 0.25 degree forecast for lat={lat:.4f}, lon={long:.4f}')
plt.legend()
plt.tight_layout()
plt.show()


def getCSirradiance(site_location, startdate, enddate, tilt):
    times = pd.date_range(startdate, enddate, freq='h')
    clearsky = site_location.get_clearsky(times)
    solar_position = site_location.get_solarposition(times=times)
    POA_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=180,
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'])

    return ({'Clear Sky': clearsky, 'Solar Position': solar_position, 'POA': POA_irradiance})

ulsan = pvlib.location.Location(lat, long, tz='Asia/Seoul')
csirr = getCSirradiance(ulsan, str(start), str(end), 30)

for key, data in csirr.items():
    data.plot(figsize=(12,5))
    plt.ylabel('Value')
    plt.xlabel('Forecast Time ($UTC+9$)')
    plt.title(f'GFS 0.25 degree {key} forecast for lat={lat:.4f}, lon={long:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.show()