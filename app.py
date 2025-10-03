# app.py — Country Routes Explorer (Streamlit version using your OpenFlights .dat files)

# --- Configuration: data file paths (as you sent) ---
AIRPORTS_PATH = "Data/airports.dat"
AIRLINES_PATH = "Data/airlines.dat"
ROUTES_PATH   = "Data/routes.dat"
CC_MAP_PATH   = "Data/country_continent.dat"  # continent mapping (your file)

# --- Imports ---
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from pyproj import Geod

# =============================================================================
#                       COUNTRY↔CONTINENT MAP (from your file)
# =============================================================================
def ensure_country_continent_map(path: str):
    p = Path(path)
    if not p.exists():
        # Keep behavior strict: require your file to be present
        raise FileNotFoundError(
            f"Missing continent map: {path}\n"
            "Please place your 'country_continent.dat' with columns: country, continent"
        )
    # Accept .dat/.csv
    return pd.read_csv(p)

cc_map = ensure_country_continent_map(CC_MAP_PATH)
# Expecting columns: country, continent
COUNTRY_TO_CONTINENT = cc_map.set_index("country")["continent"].to_dict()

# Name harmonization between OpenFlights and mapping (as in your code)
COUNTRY_ALIASES = {
    "Ivory Coast": "Cote d'Ivoire",
    "Congo (Kinshasa)": "DR Congo",
    "Congo (Brazzaville)": "Congo Republic",
    "Republic of the Congo": "Congo Republic",
    "Macau": "Macao",
    "Cape Verde": "Cabo Verde",
    "North Macedonia": "Macedonia",
}

def standardize_country(s: pd.Series) -> pd.Series:
    return s.replace(COUNTRY_ALIASES)

def continent_lookup(s: pd.Series) -> pd.Series:
    return s.map(COUNTRY_TO_CONTINENT).fillna("Other")

# =============================================================================
#                                  LOAD DATA
# =============================================================================
airports_cols = [
    'airport_id','name','city','country','iata','icao','lat','lon','altitude',
    'timezone','dst','tz','type','source'
]
airlines_cols = [
    'airline_id','name','alias','iata','icao','callsign','country','active'
]
routes_cols = [
    'airline','airline_id','source_airport','source_airport_id',
    'destination_airport','destination_airport_id','codeshare','stops','equipment'
]

def load_openflights_data():
    airports = pd.read_csv(AIRPORTS_PATH, header=None, names=airports_cols, na_values="\\N", dtype=str)
    airlines = pd.read_csv(AIRLINES_PATH, header=None, names=airlines_cols, na_values="\\N", dtype=str)
    routes   = pd.read_csv(ROUTES_PATH,   header=None, names=routes_cols,   na_values="\\N", dtype=str)

    # Clean types/casing
    for col in ["lat","lon"]:
        airports[col] = pd.to_numeric(airports[col], errors="coerce")

    for c in ["iata","icao","country","city","name"]:
        airports[c] = airports[c].astype(str).str.strip()

    for c in ["airline","airline_id","source_airport","destination_airport"]:
        routes[c] = routes[c].astype(str).str.strip()

    for c in ["country","active","name","airline_id","iata","icao"]:
        airlines[c] = airlines[c].astype(str).str.strip()

    return airports, airlines, routes

# Build code → (country, lat, lon) maps (IATA & ICAO)
def build_code_maps(airports: pd.DataFrame):
    air_lite = airports[["iata","icao","country","lat","lon"]].copy()
    code_to_country, code_to_lat, code_to_lon = {}, {}, {}
    for _, row in air_lite.dropna(subset=["country"]).iterrows():
        if row["iata"] and row["iata"] != "nan":
            code = row["iata"]
            code_to_country[code] = row["country"]
            code_to_lat[code] = row["lat"]
            code_to_lon[code] = row["lon"]
        if row["icao"] and row["icao"] != "nan":
            code = row["icao"]
            code_to_country[code] = row["country"]
            code_to_lat[code] = row["lat"]
            code_to_lon[code] = row["lon"]
    return code_to_country, code_to_lat, code_to_lon

def map_country(code, code_to_country):
    if pd.isna(code): return None
    return code_to_country.get(str(code).strip())

def map_lat(code, code_to_lat):
    if pd.isna(code): return np.nan
    return code_to_lat.get(str(code).strip())

def map_lon(code, code_to_lon):
    if pd.isna(code): return np.nan
    return code_to_lon.get(str(code).strip())

@st.cache_data
def prepare_data():
    airports, airlines, routes = load_openflights_data()
    c2country, c2lat, c2lon = build_code_maps(airports)

    # Enrich routes (countries + coords + continents), as in your notebook
    r_enriched = routes[["airline","airline_id","source_airport","destination_airport"]].copy()
    r_enriched["src_country"] = r_enriched["source_airport"].apply(lambda x: map_country(x, c2country))
    r_enriched["dst_country"] = r_enriched["destination_airport"].apply(lambda x: map_country(x, c2country))

    r_enriched["src_lat"] = r_enriched["source_airport"].apply(lambda x: map_lat(x, c2lat))
    r_enriched["src_lon"] = r_enriched["source_airport"].apply(lambda x: map_lon(x, c2lon))
    r_enriched["dst_lat"] = r_enriched["destination_airport"].apply(lambda x: map_lat(x, c2lat))
    r_enriched["dst_lon"] = r_enriched["destination_airport"].apply(lambda x: map_lon(x, c2lon))

    # Drop routes missing country or coords
    r_enriched = r_enriched.dropna(subset=["src_country","dst_country","src_lat","src_lon","dst_lat","dst_lon"])

    # Standardize country names and attach continents
    r_enriched["src_country_std"] = standardize_country(r_enriched["src_country"])
    r_enriched["dst_country_std"] = standardize_country(r_enriched["dst_country"])
    r_enriched["src_continent"]   = continent_lookup(r_enriched["src_country_std"])
    r_enriched["dst_continent"]   = continent_lookup(r_enriched["dst_country_std"])

    return airports, airlines, r_enriched

# =============================================================================
#                                 STATS
# =============================================================================
def compute_country_stats(r_df: pd.DataFrame, airlines_df: pd.DataFrame, country_name: str) -> dict:
    """
    - In/Out routes: unique airport→airport pairs (directional).
    - Partner countries: distinct src (inbound) / dst (outbound) country sets + union.
    - Servicing airlines: distinct carriers operating any route to/from the country.
    - Airlines based: from airlines.dat 'country' (total & active==Y).
    """
    c_std = COUNTRY_ALIASES.get(country_name, country_name)

    # Unique airport-pairs (directional)
    ru = r_df[['source_airport','destination_airport','src_country_std','dst_country_std','airline','airline_id']].drop_duplicates()
    inbound   = ru[ru['dst_country_std'].eq(c_std)]
    outbound  = ru[ru['src_country_std'].eq(c_std)]

    partners_in  = inbound['src_country_std'].dropna().unique()
    partners_out = outbound['dst_country_std'].dropna().unique()
    partners_all = set(partners_in) | set(partners_out)

    # Deduplicate airline identifiers
    def airline_key(sid, scode):
        return sid if (isinstance(sid, str) and sid.strip()) else scode

    serv = ru.loc[ru['src_country_std'].eq(c_std) | ru['dst_country_std'].eq(c_std), ['airline_id','airline']]
    servicing = {
        airline_key(sid, sc) for sid, sc in zip(serv['airline_id'], serv['airline'])
        if isinstance(airline_key(sid, sc), str) and airline_key(sid, sc).strip()
    }

    al = airlines_df[['airline_id','name','country','active']].copy()
    al['country'] = standardize_country(al['country'].astype(str).str.strip())
    based = al[al['country'].eq(c_std)]
    based_active = based[based['active'].astype(str).str.upper().eq('Y')]

    return {
        "Number of inbound routes":            int(len(inbound)),
        "Number of outbound routes":           int(len(outbound)),
        "Countries with direct flights (in)":  int(len(partners_in)),
        "Countries with direct flights (out)": int(len(partners_out)),
        "Countries with direct flights (any)": int(len(partners_all)),
        "Servicing airlines (to/from)":        int(len(servicing)),
        "Airlines based in-country (total)":   int(len(based)),
        "Airlines based in-country (active)":  int(len(based_active)),
    }

# =============================================================================
#                                   PLOT
# =============================================================================
COLORS = {
    'North America': '#00e5ff',
    'South America': '#ffaf00',
    'Europe':        '#ff69b4',
    'Africa':        '#ffd800',
    'Asia':          '#ff6b6b',
    'Oceania':       '#7c4dff',
    'Antarctica':    '#8bc34a',
    'Other':         '#999999'
}
geod = Geod(ellps="WGS84")

def plot_country_routes_static(
    r_df, airports_df, country_name: str, direction: str = "both",
    coastline_res: str = "10m",  # "10m" best, "50m" faster
    projection=None,
    sel_dot_size: float = 8,     # airports inside selected country
    other_dot_size: float = 12   # airports at the other end of routes
):
    if projection is None:
        projection = ccrs.Robinson()

    c_std = COUNTRY_ALIASES.get(country_name, country_name)
    mask_out = r_df['src_country_std'].eq(c_std)
    mask_in  = r_df['dst_country_std'].eq(c_std)
    if direction == 'outbound':
        df = r_df[mask_out]
    elif direction == 'inbound':
        df = r_df[mask_in]
    else:
        df = r_df[mask_out | mask_in]

    fig = plt.figure(figsize=(16, 9), facecolor='black')
    ax = plt.axes(projection=projection, facecolor='black')
    ax.set_global()
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.set_autoscale_on(False)

    # Base (no country borders; just continents coastlines)
    ax.add_feature(cfeature.LAND.with_scale(coastline_res),  facecolor='#1b1b1b', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale(coastline_res), facecolor='#0b0b0b', zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale(coastline_res), edgecolor='#aaaaaa', linewidth=0.6, zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale(coastline_res), facecolor='#0b0b0b', edgecolor='#0b0b0b', linewidth=0, zorder=1)
    ax.gridlines(color='#333333', linewidth=0.25, alpha=0.3, draw_labels=False)

    def draw_geodesic(lon1, lat1, lon2, lat2, color, lw=0.35, npts=20, alpha=0.9):
        pts = geod.npts(lon1, lat1, lon2, lat2, npts)
        lons = [lon1] + [p[0] for p in pts] + [lon2]
        lats = [lat1] + [p[1] for p in pts] + [lat2]
        ax.plot(lons, lats, transform=ccrs.Geodetic(), color=color, linewidth=lw, alpha=alpha, zorder=2)

    # Outbound: color by DESTINATION continent
    if direction in ('outbound','both'):
        out = df[df['src_country_std'].eq(c_std)]
        for cont, sub in out.groupby('dst_continent'):
            col = COLORS.get(cont, COLORS['Other'])
            for _, row in sub.iterrows():
                draw_geodesic(row['src_lon'], row['src_lat'], row['dst_lon'], row['dst_lat'], col)

    # Inbound: color by ORIGIN continent
    if direction in ('inbound','both'):
        inn = df[df['dst_country_std'].eq(c_std)]
        for cont, sub in inn.groupby('src_continent'):
            col = COLORS.get(cont, COLORS['Other'])
            for _, row in sub.iterrows():
                draw_geodesic(row['src_lon'], row['src_lat'], row['dst_lon'], row['dst_lat'], col)

    # Airports in selected country
    apts = airports_df.copy()
    apts['country_std'] = standardize_country(apts['country'])
    apts_sel = apts[apts['country_std'].eq(c_std)]

    if not apts_sel.empty:
        ax.scatter(
            apts_sel['lon'], apts_sel['lat'],
            transform=ccrs.PlateCarree(),
            s=sel_dot_size, color='white', alpha=0.95,
            linewidths=0.25, edgecolors='#111', zorder=3
        )

    # Other-end airports (deduped & excluding domestic duplicates)
    other_chunks = []
    if direction in ('outbound','both'):
        sub = df[df['src_country_std'].eq(c_std)][['dst_lon','dst_lat']].dropna()
        other_chunks.append(sub.rename(columns={'dst_lon':'lon','dst_lat':'lat'}))
    if direction in ('inbound','both'):
        sub = df[df['dst_country_std'].eq(c_std)][['src_lon','src_lat']].dropna()
        other_chunks.append(sub.rename(columns={'src_lon':'lon','src_lat':'lat'}))

    if other_chunks:
        other_airports = pd.concat(other_chunks, ignore_index=True)
        other_airports['lon_r'] = other_airports['lon'].round(6)
        other_airports['lat_r'] = other_airports['lat'].round(6)
        other_airports = other_airports.drop_duplicates(['lon_r','lat_r'])

        if not apts_sel.empty:
            sel_xy = set(zip(apts_sel['lon'].round(6), apts_sel['lat'].round(6)))
            other_airports = other_airports[~other_airports[['lon_r','lat_r']].apply(tuple, axis=1).isin(sel_xy)]

        if not other_airports.empty:
            ax.scatter(
                other_airports['lon'], other_airports['lat'],
                transform=ccrs.PlateCarree(),
                s=other_dot_size, color='#cfcfcf', alpha=0.85,
                linewidths=0.25, edgecolors='#111', zorder=3
            )

    # Legend
    handles = []
    if direction in ('outbound','both'):
        for cont in sorted(df[df['src_country_std'].eq(c_std)]['dst_continent'].dropna().unique()):
            handles.append(Line2D([0],[0], color=COLORS.get(cont, COLORS['Other']), lw=2, label=f'Outbound – {cont}'))
    if direction in ('inbound','both'):
        for cont in sorted(df[df['dst_country_std'].eq(c_std)]['src_continent'].dropna().unique()):
            handles.append(Line2D([0],[0], color=COLORS.get(cont, COLORS['Other']), lw=2, label=f'Inbound – {cont}'))

    if handles:
        leg = ax.legend(handles=handles, facecolor='black', edgecolor='#444', labelcolor='white', loc='lower left')
        for t in leg.get_texts():
            t.set_color('white')

    ax.set_title(f"Flights: {country_name} ({direction.capitalize()})", color='white', fontsize=14)
    plt.tight_layout()
    return fig

# =============================================================================
#                              STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="Country Routes Explorer", layout="wide")
st.title("Country Routes Explorer")

# Load & prepare
airports, airlines, r_enriched = prepare_data()

# UI controls (match your notebook + dot sizes)
country_opts   = sorted(airports['country'].dropna().unique().tolist())
direction_opts = ['both','outbound','inbound']
detail_opts    = ['10m','50m','110m']

with st.sidebar:
    st.header("Controls")
    country = st.selectbox("Country", country_opts, index=0)
    direction = st.radio("Direction", direction_opts, index=0, horizontal=True)
    detail = st.selectbox("Coastline detail", detail_opts, index=0)
    sel_dot_size = st.slider("Selected-country airport size", 2, 30, 8)
    other_dot_size = st.slider("Other-end airport size", 2, 40, 12)

# Map first
fig = plot_country_routes_static(
    r_df=r_enriched,
    airports_df=airports,
    country_name=country,
    direction=direction,
    coastline_res=detail,
    sel_dot_size=sel_dot_size,
    other_dot_size=other_dot_size
)
st.pyplot(fig, use_container_width=True)

# Stats table below
stats = compute_country_stats(r_enriched, airlines, country)
st.subheader("Country Stats")
st.dataframe(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))
