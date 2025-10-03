# Global Flight Routes & Airport Network
### Visualizing the World's Air Travel Routes Using Open Source Data and Python

![Project Image](https://raw.githubusercontent.com/imtiajiqbalmahfuj/imtiajiqbal-portfolio/refs/heads/main/Projects/25016%20Global%20Flight%20Routes%20%26%20Airport%20Network%20Map/Global_Airport_Routes_Final.png)

![Date](https://img.shields.io/badge/19/09/2025-19/09/2025-blue) 
![Location](https://img.shields.io/badge/Location-Global-green) 

---

## 📝 Overview
The COVID-19 pandemic drastically reduced global air travel, but it is now showing signs of recovery. In 2019, the total number of passengers carried by planes hit an all-time high of 4.5 billion. While flight routes are often overlooked, visualizing them provides fascinating insights into popular travel corridors, international hubs, and global connectivity.  

This project demonstrates how to generate eye-catching visualizations of global airways using open source data and Python, leveraging data manipulation, geospatial analysis, and visualization libraries. Through this exercise, you will gain a better understanding of libraries like **Pandas, GeoPandas, Matplotlib, Shapely, and Cartopy**, and how to handle geospatial data at a global scale.

---

## 🛠️ Tools & Technologies
![Python](https://img.shields.io/badge/Python-3.10-blue)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Manipulation-orange)  
![GeoPandas](https://img.shields.io/badge/GeoPandas-Geospatial%20Analysis-red)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)  
![Shapely](https://img.shields.io/badge/Shapely-Geometries-lightgrey)  
![Cartopy](https://img.shields.io/badge/Cartopy-Geospatial%20Mapping-purple)  

---

## ⚙️ Methodology
| Step | Description |
|------|-------------|
| 1. Data Collection | OpenFlights datasets: airports and routes CSV files, under Database Contents License. The airports dataset contains 7,608 unique airports with location, city, country, and unique IATA codes. The routes dataset provides source-destination pairs for over 66,000 routes worldwide. |
| 2. Preprocessing   | Cleaned datasets, assigned continents/regions to airports, calculated total routes per airport, and prepared data for geospatial analysis. |
| 3. Analysis        | Created **GeoDataFrames** for airports and **LineStrings** for routes using Shapely. Ensured routes properly account for source-destination pairs and merged airport information. |
| 4. Visualization   | Plotted global airways using Matplotlib and Cartopy, applied **Robinson projection** for realistic global visualization, accounted for Earth's curvature, and colored routes by region. Airports were sized by connectivity (number of routes) to highlight major hubs. |

---

## 📊 Results & Codes
Import Libraries
``` python
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
```
- **Airports Distribution**: Scatter plot showing the global distribution of airports. Reflects population density and travel hubs.
``` python
airports = pd.read_csv("data/airports.dat", delimiter=',', 
                       names=['id', 'name', 'city', 'country', 'iata',
                              'icao', 'lat', 'long', 'altitude', 'timezone',
                              'dst', 'tz', 'type', 'source'])

# Rough manual mapping by country region
region_map = {
    'North America': ['United States','Canada','Mexico'],
    'South America': ['Brazil','Argentina','Chile','Colombia','Peru'],
    'Europe': ['United Kingdom','France','Germany','Italy','Spain','Netherlands'],
    'Africa': ['South Africa','Nigeria','Egypt','Kenya','Ethiopia'],
    'Asia': ['China','Japan','India','Thailand','Singapore','United Arab Emirates'],
    'Oceania': ['Australia','New Zealand']
}

def get_region(country):
    for r, c_list in region_map.items():
        if country in c_list:
            return r
    return 'Other'

airports['region'] = airports['country'].apply(get_region)
fig, ax = plt.subplots(figsize = (16,8))

ax.scatter(airports['long'], airports['lat'], s=2, alpha=1, edgecolors='none')

# Add legend: create a dummy point
ax.scatter([], [], s=2, color='blue', label='Airports')  # invisible point for legend
ax.legend(loc='lower left', fontsize=12, facecolor='white', edgecolor='black', title='Legend')

ax.axis('off')
ax.set_title("Airpors of the world", fontsize = 18)

plt.savefig("Airports of the world.png", dpi=300, bbox_inches='tight')
plt.show
```
![1](https://raw.githubusercontent.com/imtiajiqbalmahfuj/imtiajiqbal-portfolio/refs/heads/main/Projects/25016%20Global%20Flight%20Routes%20%26%20Airport%20Network%20Map/Airports%20of%20the%20world.png)
- **Flat Airways Map**: Direct straight-line connections between airports to illustrate the raw route network.
``` python
routes = pd.read_csv("data/routes.dat", delimiter=',', names=['airline', 'id', 'source_airport', 'source_airport_id',
                                                               'destination_airport', 'destination_airport_id', 'codeshare',
                                                               'stops', 'equitment'])
# Count routes per airport
src_counts = routes['source_airport'].value_counts()
dst_counts = routes['destination_airport'].value_counts()
total_counts = src_counts.add(dst_counts, fill_value=0)

airports['route_count'] = airports['iata'].map(total_counts).fillna(0)

# Build GeoDataFrame for airports
airports_gdf = gpd.GeoDataFrame(
    airports, geometry=gpd.points_from_xy(airports['long'], airports['lat']),
    crs='EPSG:4326'
)
source_airports = airports[['name', 'iata', 'icao', 'lat', 'long', 'region']]
destination_airports = source_airports.copy()
source_airports.columns = [str(col) + '_source' for col in source_airports.columns]
destination_airports.columns = [str(col) + '_destination' for col in destination_airports.columns]
routes = routes[['source_airport', 'destination_airport']]
routes = pd.merge(routes, source_airports, left_on='source_airport', right_on='iata_source')
routes = pd.merge(routes, destination_airports, left_on='destination_airport', 
                  right_on='iata_destination')
# Create route geometries
geometry = [LineString([[routes.iloc[i]['long_source'], routes.iloc[i]['lat_source']], 
                        [routes.iloc[i]['long_destination'], 
                         routes.iloc[i]['lat_destination']]]) for i in range(routes.shape[0])]
routes = gpd.GeoDataFrame(routes, geometry=geometry, crs='EPSG:4326')
print(routes)
- **Curvature Corrected Map**: Curved lines accounting for the Earth's spherical shape using Cartopy's Geodetic transform.  
- **Region-Colored Routes**: Routes are colored by the continent of the source airport and airports sized by route count to highlight regional hubs and international connectivity.  
- **Final Visualization**: Combines region-colored routes with airports as dots, providing a clear, eye-catching map of global airways.
fig, ax = plt.subplots ( figsize = (16, 8), facecolor = "black")
ax.patch.set_facecolor('black')
routes.plot(ax=ax, color='white', linewidth=0.1)

plt.setp(ax.spines.values(), color='black')
plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='black')

# Save figure
plt.savefig("Global_Airport_Routes_flat.png", dpi=300, bbox_inches='tight', facecolor='black')

plt.show()
```
![1](https://raw.githubusercontent.com/imtiajiqbalmahfuj/imtiajiqbal-portfolio/refs/heads/main/Projects/25016%20Global%20Flight%20Routes%20%26%20Airport%20Network%20Map/Global_Airport_Routes_flat.png)
- **Curvature Corrected Map**: Curved lines accounting for the Earth's spherical shape using Cartopy's Geodetic transform.
``` python
fig, ax = plt.subplots(figsize=(16, 8),
                       subplot_kw={'projection': ccrs.Robinson()},
                       facecolor='black')
ax.set_facecolor('black')

# Plot 'Other' first (ash + low alpha)
subset = routes[routes['region_source'] == 'Other']
subset.plot(ax=ax, transform=ccrs.Geodetic(),
            color=region_colors['Other'], linewidth=0.1, alpha=0.05)

# Plot routes colored by source region
for region, color in region_colors.items():
    if region == 'Other':
        continue
    subset = routes[routes['region_source'] == region]
    subset.plot(ax=ax, transform=ccrs.Geodetic(),
                color=color, linewidth=0.1, alpha=0.3)

ax.set_global()
ax.axis('off')

# Save figure
plt.savefig("Global_Airport_Routes.png", dpi=300, bbox_inches='tight', facecolor='black')

plt.show()
```
![1](https://raw.githubusercontent.com/imtiajiqbalmahfuj/imtiajiqbal-portfolio/refs/heads/main/Projects/25016%20Global%20Flight%20Routes%20%26%20Airport%20Network%20Map/Global_Airport_Routes_nocolor.png)
- **Region-Colored Routes**: Routes are colored by the continent of the source airport and airports sized by route count to highlight regional hubs and international connectivity.
``` python
fig, ax = plt.subplots(figsize=(16, 8),
                       subplot_kw={'projection': ccrs.Robinson()},
                       facecolor='black')
ax.set_facecolor('black')

# Plot 'Other' routes first (ash faint)
other = routes[routes['region_source'] == 'Other']
other.plot(ax=ax, transform=ccrs.Geodetic(),
           color=region_colors['Other'], linewidth=0.1, alpha=0.05)

# Plot colored routes on top
for region, color in region_colors.items():
    if region == 'Other':
        continue
    subset = routes[routes['region_source'] == region]
    subset.plot(ax=ax, transform=ccrs.Geodetic(),
                color=color, linewidth=0.1, alpha=0.3)

# Plot airports as dots (size by route_count, color by region)
for region, color in region_colors.items():
    a = airports_gdf[airports_gdf['region'] == region]
    a.plot(ax=ax, transform=ccrs.PlateCarree(),
           markersize=a['route_count']*0.05 + 1,  # scaling factor
           color=color, alpha=0.7, edgecolor='none')

ax.set_global()
ax.axis('off')

# Save figure
plt.savefig("Global_Airport_Routes_dots.png", dpi=300, bbox_inches='tight', facecolor='black')

plt.show()
```
![1](https://raw.githubusercontent.com/imtiajiqbalmahfuj/imtiajiqbal-portfolio/refs/heads/main/Projects/25016%20Global%20Flight%20Routes%20%26%20Airport%20Network%20Map/Global_Airport_Routes.png)
![1](https://raw.githubusercontent.com/imtiajiqbalmahfuj/imtiajiqbal-portfolio/refs/heads/main/Projects/25016%20Global%20Flight%20Routes%20%26%20Airport%20Network%20Map/Global_Airport_Routes_dots.png)
- **Final Visualization**: Combines region-colored routes with airports as dots, providing a clear, eye-catching map of global airways.

``` python
fig, ax = plt.subplots(figsize=(16, 8),
                       subplot_kw={'projection': ccrs.Robinson()},
                       facecolor='black')
ax.set_facecolor('black')

# Plot 'Other' routes first (ash faint)
other = routes[routes['region_source'] == 'Other']
other.plot(ax=ax, transform=ccrs.Geodetic(),
           color=region_colors['Other'], linewidth=0.1, alpha=0.05)

# Plot colored routes on top
for region, color in region_colors.items():
    if region == 'Other':
        continue
    subset = routes[routes['region_source'] == region]
    subset.plot(ax=ax, transform=ccrs.Geodetic(),
                color=color, linewidth=0.1, alpha=0.3)

# Plot airports as dots (size by route_count, color by region)
for region, color in region_colors.items():
    a = airports_gdf[airports_gdf['region'] == region]
    a.plot(ax=ax, transform=ccrs.PlateCarree(),
           markersize=a['route_count']*0.05 + 1,  # scaling factor
           color=color, alpha=0.7, edgecolor='none')

# Create legend handles
handles = []
for region, color in region_colors.items():
    handles.append(mpatches.Patch(color=color, label=region))

ax.set_global()
ax.axis('off')

# ----- Add legend for regions -----
handles = [mpatches.Patch(color=color, label=region) 
           for region, color in region_colors.items()]

# Create legend
legend = ax.legend(handles=handles, loc='lower left', 
                   fontsize=12, facecolor='black', framealpha=0.8, edgecolor='white',
                   title='Region', title_fontsize=14)

# Set colors directly after creation
plt.setp(legend.get_title(), color='white')  # legend title color
for text in legend.get_texts():
    text.set_color('white')  # legend labels
    
# Add title at top
ax.set_title("Global Airport Routes by Region", fontsize=20, color='white', pad=20)

# Add name at bottom
fig.text(0.5, 0.01, "Imtiaj Iqbal Mahfuj", fontsize=14, color='white', ha='center')

# Save figure
plt.savefig("Global_Airport_Routes_Final.png", dpi=300, bbox_inches='tight', facecolor='black')

plt.show()
```
![1](https://raw.githubusercontent.com/imtiajiqbalmahfuj/imtiajiqbal-portfolio/refs/heads/main/Projects/25016%20Global%20Flight%20Routes%20%26%20Airport%20Network%20Map/Global_Airport_Routes_Final2.png)
![1](https://raw.githubusercontent.com/imtiajiqbalmahfuj/imtiajiqbal-portfolio/refs/heads/main/Projects/25016%20Global%20Flight%20Routes%20%26%20Airport%20Network%20Map/Global_Airport_Routes_Final.png)

---

## 📂 Data Source
- **OpenFlights Database**: [https://openflights.org/data.html](https://openflights.org/data.html)  
- License: Database Contents License – free to use with acknowledgment of the source.

---

## 📎 Links
- 🔗 [See more](https://www.linkedin.com/posts/imtiajiqbalmahfuj_globalairportroutes-activity-7374846086340870144-TQzD?utm_source=share&utm_medium=member_desktop&rcm=ACoAAETCC3UBjMNBwycvXEm57I2FBEXCxvdKcM0)  

---

## 🙏 Shout-Out
This project was inspired by **Adam Symington**, whose work on air traffic visualizations motivated me to explore global flight routes with Python.

---

## 🔖 Tags
`GIS` `Remote Sensing` `Geospatial Python` `Python` `Cartopy` `Air Traffic Visualization` `Data Visualization` `Open Source Data` `Spatial Data Science`  

