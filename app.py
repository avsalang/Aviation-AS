# app.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pyproj import Geod

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Country Routes Explorer", layout="wide")
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

# Replace with your own alias map if you already have one
COUNTRY_ALIASES = {}
def standardize_country(s):
    if isinstance(s, pd.Series):
        return s.fillna("").map(lambda x: COUNTRY_ALIASES.get(x, x)).str.strip()
    return COUNTRY_ALIASES.get(s, s)

# ---- Replace these loaders with your actual file paths or logic
@st.cache_data
def load_data():
    # Expecting the same columns you used in the notebook
    # r_enriched: src_country_std, dst_country_std, src_continent, dst_continent,
    #             src_lon, src_lat, dst_lon, dst_lat, (plus airline cols if needed)
    # airports:   country, lon, lat, (name / iata optional)
    # airlines:   (whatever you use for stats)
    r_enriched = pd.read_parquet("data/routes_enriched.parquet")   # or .csv
    airports = pd.read_parquet("data/airports.parquet")           # or .csv
    airlines = pd.read_parquet("data/airlines.parquet")           # optional
    return r_enriched, airports, airlines

# Dummy “stats” — plug in your real one if you already have it
def compute_country_stats(r_df, airlines_df, country):
    c_std = COUNTRY_ALIASES.get(country, country)
    mask_out = r_df['src_country_std'].eq(c_std)
    mask_in  = r_df['dst_country_std'].eq(c_std)
    outbound = r_df[mask_out]
    inbound  = r_df[mask_in]
    # examples – adapt to your real logic:
    inbound_routes  = len(inbound)
    outbound_routes = len(outbound)
    partners = pd.unique(
        pd.concat([outbound['dst_country_std'], inbound['src_country_std']]).dropna()
    )
    stats = {
        "Inbound routes":  inbound_routes,
        "Outbound routes": outbound_routes,
        "Countries with direct flights": len([p for p in partners if p != c_std]),
        # Add any airline-based stats you compute elsewhere:
        # "Servicing airlines": ...,
        # "Airlines stationed in country": ...,
    }
    return stats

def geodesic_lons_lats(lon1, lat1, lon2, lat2, npts=20):
    pts = geod.npts(lon1, lat1, lon2, lat2, npts)
    lons = [lon1] + [p[0] for p in pts] + [lon2]
    lats = [lat1] + [p[1] for p in pts] + [lat2]
    return lons, lats

# ----------------- UI -----------------
r_enriched, airports, airlines = load_data()
airports['country_std'] = standardize_country(airports['country'])
countries = sorted(airports['country_std'].dropna().unique())

st.sidebar.title("Controls")
country = st.sidebar.selectbox("Country", countries, index= countries.index("India") if "India" in countries else 0)
direction = st.sidebar.radio("Direction", ["both","outbound","inbound"], index=0)
sel_dot_size = st.sidebar.slider("Selected-country airport size", 3, 30, 8)
oth_dot_size = st.sidebar.slider("Other-end airport size", 3, 40, 12)

# ----------------- FILTER -----------------
c_std = COUNTRY_ALIASES.get(country, country)
mask_out = r_enriched['src_country_std'].eq(c_std)
mask_in  = r_enriched['dst_country_std'].eq(c_std)
if direction == "outbound":
    df = r_enriched[mask_out]
elif direction == "inbound":
    df = r_enriched[mask_in]
else:
    df = r_enriched[mask_out | mask_in]

# ----------------- BUILD FIG -----------------
fig = go.Figure()

# geodesic arcs, colored by continent (same rule as your notebook)
if direction in ("outbound", "both"):
    out = df[df['src_country_std'].eq(c_std)]
    for cont, sub in out.groupby('dst_continent'):
        col = COLORS.get(cont, COLORS['Other'])
        for _, row in sub.iterrows():
            lons, lats = geodesic_lons_lats(row['src_lon'], row['src_lat'], row['dst_lon'], row['dst_lat'])
            fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats, mode="lines",
                line=dict(width=0.6, color=col), opacity=0.9, hoverinfo="skip",
                showlegend=False
            ))

if direction in ("inbound", "both"):
    inn = df[df['dst_country_std'].eq(c_std)]
    for cont, sub in inn.groupby('src_continent'):
        col = COLORS.get(cont, COLORS['Other'])
        for _, row in sub.iterrows():
            lons, lats = geodesic_lons_lats(row['src_lon'], row['src_lat'], row['dst_lon'], row['dst_lat'])
            fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats, mode="lines",
                line=dict(width=0.6, color=col), opacity=0.9, hoverinfo="skip",
                showlegend=False
            ))

# dots: selected-country airports (smaller if you like)
apts_sel = airports[airports['country_std'].eq(c_std)]
if not apts_sel.empty:
    fig.add_trace(go.Scattergeo(
        lon=apts_sel['lon'], lat=apts_sel['lat'],
        mode="markers", marker=dict(size=sel_dot_size, color="white", line=dict(width=0.4, color="#111")),
        name=f"Airports in {country}", hovertext=apts_sel.get('name', None), hoverinfo="text"
    ))

# dots: other-end airports (deduped)
others = []
if direction in ("outbound","both"):
    sub = df[df['src_country_std'].eq(c_std)][['dst_lon','dst_lat']].dropna().rename(columns={'dst_lon':'lon','dst_lat':'lat'})
    others.append(sub)
if direction in ("inbound","both"):
    sub = df[df['dst_country_std'].eq(c_std)][['src_lon','src_lat']].dropna().rename(columns={'src_lon':'lon','src_lat':'lat'})
    others.append(sub)
if others:
    other_airports = pd.concat(others, ignore_index=True)
    other_airports['lon_r'] = other_airports['lon'].round(6)
    other_airports['lat_r'] = other_airports['lat'].round(6)
    other_airports = other_airports.drop_duplicates(['lon_r','lat_r'])
    # remove domestic duplicates
    if not apts_sel.empty:
        sel_xy = set(zip(apts_sel['lon'].round(6), apts_sel['lat'].round(6)))
        other_airports = other_airports[~other_airports[['lon_r','lat_r']].apply(tuple, axis=1).isin(sel_xy)]
    if not other_airports.empty:
        fig.add_trace(go.Scattergeo(
            lon=other_airports['lon'], lat=other_airports['lat'],
            mode="markers", marker=dict(size=oth_dot_size, color="#cfcfcf", line=dict(width=0.4, color="#111")),
            name="Connected airports (other end)", hoverinfo="skip"
        ))

fig.update_layout(
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor="black", plot_bgcolor="black",
    geo=dict(
        projection_type="natural earth",  # Robinson is not native; Natural Earth looks nice
        showland=True, landcolor="#1b1b1b",
        showocean=True, oceancolor="#0b0b0b",
        showcountries=False, showcoastlines=True, coastlinecolor="#aaaaaa",
        showlakes=True, lakecolor="#0b0b0b",
    ),
    font=dict(color="white"),
    legend=dict(bgcolor="rgba(0,0,0,0.6)")
)
fig.update_layout(title=f"Flights: {country} ({direction.capitalize()})")

# ----------------- LAYOUT: Map then Stats -----------------
st.plotly_chart(fig, use_container_width=True)

stats = compute_country_stats(r_enriched, airlines, country)
st.subheader("Country Stats")
st.dataframe(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))
