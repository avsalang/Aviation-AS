import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import cartopy.crs as ccrs
import matplotlib.patches as mpatches

airports = pd.read_csv("data/airports.dat", delimiter=',',
                       names=['id', 'name', 'city', 'country', 'iata',
                              'icao', 'lat', 'long', 'altitude', 'timezone',
                              'dst', 'tz', 'type', 'source'])
airports.head()