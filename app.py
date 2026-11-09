"""Road accident hotspot & severity dashboard (Streamlit).

Reads only the compact files in data/dashboard/ produced by
notebooks/Road_Accident_Spatiotemporal_Severity.ipynb, so it runs without Spark.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="GB Road Accident Hotspots & Severity", page_icon="🚦", layout="wide")

DASH = Path(__file__).resolve().parent / "data" / "dashboard"
NAVY, TEAL, ORANGE, CREAM = "#0f1b2d", "#2bb3a3", "#e0612a", "#f5f3ee"
FATAL = "#7a0f2a"
SEV_COLORS = {"Fatal": FATAL, "Serious": ORANGE, "Slight": TEAL}
SEV_ORDER = ["Fatal", "Serious", "Slight"]
DOW = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
GB_VIEW = pdk.ViewState(latitude=54.3, longitude=-2.6, zoom=5.1, pitch=0)
HEAT_COLORS = np.array([[255, 255, 204], [254, 217, 118], [254, 178, 76], [253, 141, 60], [227, 26, 28], [128, 0, 38]])
GI_99, GI_95 = [84, 39, 143], [158, 154, 200]     # purple, so Gi* never looks like the yellow→red heatmap


def rgb(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"


def swatch(kind, color, edge=None):
    base = "display:inline-block;vertical-align:middle;margin-right:8px;flex:none;"
    if kind == "square":
        return f"<span style='{base}width:14px;height:14px;border-radius:2px;background:{color}'></span>"
    if kind == "dot":
        return f"<span style='{base}width:10px;height:10px;margin:0 10px 0 2px;border-radius:50%;background:{color}'></span>"
    if kind == "circle":
        return f"<span style='{base}width:14px;height:14px;border-radius:50%;background:{color};border:2px solid {edge}'></span>"
    return (f"<span style='{base}width:14px;height:14px;border-radius:3px;border:2px solid {color};"
            f"background:{color}55'></span>")   # "shape": outlined, translucent fill


def legend_row(sw, text):
    return f"<div style='display:flex;align-items:center;margin:4px 0;font-size:.86rem'>{sw}<span>{text}</span></div>"


def gradient_row(text):
    stops = ", ".join(rgb(c) for c in HEAT_COLORS)
    return (f"<div style='margin:4px 0 8px;font-size:.86rem'>{text}"
            f"<div style='height:10px;border-radius:3px;margin-top:4px;background:linear-gradient(to right, {stops})'></div>"
            "<div style='display:flex;justify-content:space-between;font-size:.75rem;color:#5b6472'>"
            "<span>fewer</span><span>more</span></div></div>")

st.markdown(f"""
<style>
.block-container {{padding-top: 1.2rem; padding-bottom: 2rem;}}
[data-testid="stMetric"] {{background: #ffffff; border: 1px solid #e3e0d8; border-radius: 10px; padding: 10px 14px;}}
[data-testid="stMetricLabel"] p {{font-size: 0.82rem; color: #5b6472;}}
.eyebrow {{color: {TEAL}; letter-spacing: .12em; font-size: .78rem; font-weight: 600; margin-bottom: -.4rem;}}
.note {{color: #5b6472; font-size: .85rem;}}
.risk-badge {{display:inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 700; color: white;}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading collision data…")
def load_points():
    return pd.read_parquet(DASH / "collisions_points.parquet")


@st.cache_resource
def load_static():
    d = {name: pd.read_parquet(DASH / f"{name}.parquet") for name in
         ["district_stats", "grid_cells", "h3_risk", "kmeans_clusters", "dbscan_fatal_clusters",
          "dbscan_ksi_clusters", "association_rules"]}
    d["districts_geo"] = json.loads((DASH / "districts.geojson").read_text())
    d["metrics"] = json.loads((DASH / "model_metrics.json").read_text())
    d["summary"] = json.loads((DASH / "project_summary.json").read_text())
    d["meta"] = json.loads((DASH / "predictor_metadata.json").read_text())
    return d


@st.cache_resource(show_spinner="Loading severity model…")
def load_model():
    return joblib.load(DASH / "severity_predictor.joblib")


@st.cache_resource
def h3_parents():
    """Map each resolution-7 H3 cell to its resolution-5 parent (~9 km hexagons) for the 3D column layer."""
    import h3
    return {c: h3.cell_to_parent(c, 5) for c in load_points().h3_cell.cat.categories}


if not (DASH / "collisions_points.parquet").exists():
    st.error("data/dashboard/ is missing. Run the notebook first (section 11 writes the dashboard files).")
    st.stop()

pts = load_points()
S = load_static()
summary, metrics, meta = S["summary"], S["metrics"], S["meta"]


def opts(col):
    vals = [v for v in pts[col].cat.categories if v != "Unknown"]
    return sorted(vals) + (["Unknown"] if "Unknown" in pts[col].cat.categories else [])


# --------------------------------------------------------------------------- sidebar
st.sidebar.markdown(f"<p class='eyebrow'>FILTERS</p>", unsafe_allow_html=True)
st.sidebar.subheader("Time controls")
yr = st.sidebar.slider("Years", 2021, 2025, (2021, 2025))
hr = st.sidebar.slider("Hour of day", 0, 23, (0, 23))
days = st.sidebar.multiselect("Day of week", list(DOW.values()), placeholder="All days")
months = st.sidebar.multiselect("Month", MONTHS, placeholder="All months")
seasons = st.sidebar.multiselect("Season", SEASONS, placeholder="All seasons")

st.sidebar.subheader("Condition filters")
sev = st.sidebar.multiselect("Severity", SEV_ORDER, placeholder="All severities")
weather = st.sidebar.multiselect("Weather", opts("weather"), placeholder="Any weather")
road_type = st.sidebar.multiselect("Road type", opts("road_type"), placeholder="Any road type")
junction = st.sidebar.multiselect("Junction", opts("junction"), placeholder="Any junction")
with st.sidebar.expander("More conditions"):
    light = st.multiselect("Light", opts("light"), placeholder="Any light")
    surface = st.multiselect("Road surface", opts("surface"), placeholder="Any surface")
    road_class = st.multiselect("Road class", opts("road_class"), placeholder="Any class")
    area_type = st.multiselect("Urban / rural", opts("area_type"), placeholder="Both")
    speeds = st.multiselect("Speed limit (mph)", sorted(int(s) for s in pts.speed_limit.unique() if s > 0), placeholder="Any")
district = st.sidebar.selectbox("District", ["All Great Britain"] + sorted(pts.lad_name.cat.categories))


def apply_filters(df, ignore_district=False):
    m = df.year.between(*yr) & df.hour.between(*hr)
    if days: m &= df.day_of_week.isin([k for k, v in DOW.items() if v in days])
    if months: m &= df.month.isin([MONTHS.index(x) + 1 for x in months])
    if seasons: m &= df.season.isin(seasons)
    for col, sel in [("severity", sev), ("weather", weather), ("road_type", road_type), ("junction", junction),
                     ("light", light), ("surface", surface), ("road_class", road_class), ("area_type", area_type)]:
        if sel: m &= df[col].isin(sel)
    if speeds: m &= df.speed_limit.isin(speeds)
    if district != "All Great Britain" and not ignore_district: m &= df.lad_name == district
    return df[m]


fdf = apply_filters(pts)
n = len(fdf)
active = sum(bool(x) for x in [days, months, seasons, sev, weather, road_type, junction, light, surface,
                               road_class, area_type, speeds]) + (yr != (2021, 2025)) + (hr != (0, 23)) + (district != "All Great Britain")
st.sidebar.caption(f"**{n:,}** of {len(pts):,} collisions match · {active} filter(s) active")

# --------------------------------------------------------------------------- header
st.markdown("<p class='eyebrow'>SPATIAL BIG DATA · APACHE SEDONA + SPARK MLLIB</p>", unsafe_allow_html=True)
st.title("Road Accident Hotspots & Severity – Great Britain")
st.caption(f"UK DfT STATS19 collisions {summary['years']} · {summary['study_records']:,} geo-located collisions in "
           f"{summary['districts']} local authority districts")


def kpis(df):
    c = df.severity.value_counts()
    f, s = int(c.get("Fatal", 0)), int(c.get("Serious", 0))
    cols = st.columns(5)
    cols[0].metric("Collisions", f"{len(df):,}")
    cols[1].metric("Fatal", f"{f:,}")
    cols[2].metric("Serious", f"{s:,}")
    cols[3].metric("KSI share", f"{100 * (f + s) / max(len(df), 1):.1f}%", help="Killed or seriously injured (fatal + serious)")
    cols[4].metric("Fatal per 1,000", f"{1000 * f / max(len(df), 1):.1f}")


def sev_colored(fig):
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color=NAVY), legend_title_text="")
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="#eeeeee")
    return fig


def linked_charts(df, key):
    a, b, c = st.columns(3)
    with a:
        h = df.groupby(["hour", "severity"], observed=True).size().reset_index(name="collisions")
        fig = px.bar(h, x="hour", y="collisions", color="severity", color_discrete_map=SEV_COLORS,
                     category_orders={"severity": SEV_ORDER}, title="By hour of day")
        st.plotly_chart(sev_colored(fig), key=f"{key}_h")
    with b:
        y = df.groupby(["year", "severity"], observed=True).size().reset_index(name="collisions")
        fig = px.bar(y, x="year", y="collisions", color="severity", color_discrete_map=SEV_COLORS,
                     category_orders={"severity": SEV_ORDER}, title="By year")
        st.plotly_chart(sev_colored(fig), key=f"{key}_y")
    with c:
        t = df.groupby("lad_name", observed=True).agg(collisions=("severity", "size"),
                                                      fatal=("severity_code", lambda s: int((s == 1).sum())))
        t = t.nlargest(10, "collisions").sort_values("collisions")
        fig = go.Figure([go.Bar(y=t.index, x=t.collisions, orientation="h", marker_color=NAVY, name="collisions",
                                hovertemplate="%{y}: %{x:,} collisions<extra></extra>")])
        fig.update_layout(title="Top districts in selection")
        st.plotly_chart(sev_colored(fig), key=f"{key}_d")


tabs = st.tabs(["🗺️ Hotspot map", "⚠️ Fatal & risk areas", "🔥 Hotspots & clusters", "🎯 Severity predictor",
                "🔗 Rule explorer", "⏱️ Time patterns", "📊 Model performance", "ℹ️ About"])

# =========================================================================== 1 MAP
with tabs[0]:
    kpis(fdf)
    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("**Map layers**")
        heat_mode = st.radio("Heatmap shows", ["All collisions", "Fatal & serious only", "Fatal only", "Off"],
                             help="Colour = relative density of the chosen collisions (square-root scale, so cities "
                                  "outside London stay visible).")
        show_hex = st.toggle("3D hexagon columns (H3)", False,
                             help="Column height and colour = collisions in each H3 hexagon (~9 km across). "
                                  "The map tilts automatically; right-drag or Ctrl+drag to rotate.")
        show_ksi = st.toggle("Fatal & serious points", False)
        show_gi = st.toggle("Gi* significant hot spots", True)
        show_km = st.toggle(f"K-Means centres (k={summary['kmeans_k']})", False)
        show_dbf = st.toggle("DBSCAN fatal clusters", True)
        show_dbk = st.toggle("DBSCAN KSI junction clusters", False)
        st.markdown("<p class='note'>Hover for details, click a cluster or cell for a popup panel below the map.</p>",
                    unsafe_allow_html=True)

    layers = []
    if heat_mode != "Off":
        hsub = {"All collisions": fdf, "Fatal & serious only": fdf[fdf.severity_code <= 2],
                "Fatal only": fdf[fdf.severity_code == 1]}[heat_mode]
        agg = (hsub.assign(lon_r=(hsub.lon / 0.01).round() * 0.01, lat_r=(hsub.lat / 0.01).round() * 0.01)
               .groupby(["lon_r", "lat_r"]).size().reset_index(name="n"))
        agg["w"] = np.sqrt(agg.n)   # compress the range so London does not swamp the colour scale
        layers.append(pdk.Layer("HeatmapLayer", agg, get_position=["lon_r", "lat_r"], get_weight="w",
                                radius_pixels=22, intensity=1.5, threshold=0.03, aggregation="SUM",
                                color_range=HEAT_COLORS.tolist()))
    if show_hex and len(fdf):
        hx = (fdf.assign(hex=fdf.h3_cell.map(h3_parents()).astype(str), fatal=(fdf.severity_code == 1).astype(int),
                         ksi=(fdf.severity_code <= 2).astype(int))
              .groupby("hex").agg(n=("fatal", "size"), fatal=("fatal", "sum"), ksi=("ksi", "sum")).reset_index())
        t = np.sqrt(hx.n / hx.n.max())
        ci = np.minimum((t * len(HEAT_COLORS)).astype(int), len(HEAT_COLORS) - 1)
        hx["r"], hx["g"], hx["b"] = HEAT_COLORS[ci, 0], HEAT_COLORS[ci, 1], HEAT_COLORS[ci, 2]
        hx["elev"] = t * 80_000     # tallest column 80 km, square-root scale
        hx["popup"] = ("H3 hexagon · " + hx.n.map("{:,}".format) + " collisions · " + hx.ksi.map("{:,}".format)
                       + " fatal/serious · " + hx.fatal.astype(str) + " fatal")
        layers.append(pdk.Layer("H3HexagonLayer", hx, id="hex", get_hexagon="hex", get_fill_color=["r", "g", "b", 225],
                                extruded=True, get_elevation="elev", coverage=0.92, pickable=True))
    if show_ksi:
        k = fdf[fdf.severity_code <= 2]
        k = k.sample(min(len(k), 40_000), random_state=0)
        k = k.assign(r=np.where(k.severity_code == 1, 122, 224), g=np.where(k.severity_code == 1, 15, 97),
                     b=np.where(k.severity_code == 1, 42, 42), label=k.severity.astype(str),
                     where=k.lad_name.astype(str), when=k.year.astype(str) + " · " + k.hour.astype(str) + ":00",
                     cond=k.weather.astype(str) + " · " + k.light.astype(str) + " · " + k.road_type.astype(str))
        layers.append(pdk.Layer("ScatterplotLayer", k[["lon", "lat", "r", "g", "b", "label", "where", "when", "cond"]],
                                id="points", get_position=["lon", "lat"], get_fill_color=["r", "g", "b", 190],
                                radius_min_pixels=2, radius_max_pixels=6, get_radius=60, pickable=True))
    if show_gi:
        gi = S["grid_cells"]
        gi = gi[gi.gi_class.str.startswith("Hot")].copy()
        gi["polygon_c"] = gi.polygon.apply(json.loads)
        gi["fill"] = gi.gi_class.map({"Hot spot (99%)": GI_99 + [185], "Hot spot (95%)": GI_95 + [175]})
        gi["popup"] = ("Gi* " + gi.gi_class + " · z = " + gi.Z.round(1).astype(str) + " · KSI " + gi.ksi.astype(int).astype(str)
                       + " · " + gi.temporal_category)
        layers.append(pdk.Layer("PolygonLayer", gi[["grid_id", "polygon_c", "fill", "popup", "ksi", "collisions", "Z", "gi_class",
                                                    "temporal_category"]],
                                id="gi", get_polygon="polygon_c", get_fill_color="fill", get_line_color=[255, 255, 255, 80],
                                line_width_min_pixels=0.5, pickable=True))
    if show_km:
        km = S["kmeans_clusters"].copy()
        km["popup"] = ("K-Means cluster " + km.kmeans_cluster.astype(str) + " · " + km.ksi_collisions.astype(str)
                       + " KSI · " + km.fatal.astype(str) + " fatal · " + km.main_district)
        layers.append(pdk.Layer("ScatterplotLayer", km, id="kmeans", get_position=["lon", "lat"],
                                get_radius="ksi_collisions", radius_scale=4, get_fill_color=[43, 179, 163, 140],
                                get_line_color=[15, 27, 45], line_width_min_pixels=1.5, stroked=True, pickable=True))
    for flag, key, color, label in [(show_dbf, "dbscan_fatal_clusters", [122, 15, 42], "fatal"),
                                    (show_dbk, "dbscan_ksi_clusters", [224, 97, 42], "KSI")]:
        if flag:
            cl = S[key].copy()
            cl["poly"] = cl.hull_geojson.apply(lambda g: json.loads(g)["coordinates"][0] if json.loads(g)["type"] == "Polygon" else [])
            cl["popup"] = (f"DBSCAN {label} cluster " + cl.cluster.astype(str) + " · " + cl.collisions.astype(str)
                           + " collisions (" + cl.fatal.astype(str) + " fatal) · " + cl.district.astype(str)
                           + " · " + cl.first_year.astype(str) + "–" + cl.last_year.astype(str))
            layers.append(pdk.Layer("PolygonLayer", cl[["poly", "popup", "cluster", "collisions", "fatal", "district", "lon", "lat"]],
                                    id=key, get_polygon="poly", get_fill_color=color + [110], get_line_color=color + [255],
                                    line_width_min_pixels=2, pickable=True))
            layers.append(pdk.Layer("ScatterplotLayer", cl[["lon", "lat", "popup", "cluster", "collisions", "fatal", "district"]],
                                    id=key + "_pt", get_position=["lon", "lat"], get_radius=350, radius_min_pixels=4,
                                    get_fill_color=color + [235], stroked=True, get_line_color=[255, 255, 255],
                                    line_width_min_pixels=1.2, pickable=True))

    tooltip = {"html": "{popup}{label} {where} {when} {cond}{elevationValue}",
               "style": {"backgroundColor": NAVY, "color": "white", "fontSize": "12px"}}
    view = pdk.ViewState(latitude=53.0, longitude=-2.0, zoom=5.5, pitch=50, bearing=-12) if show_hex else GB_VIEW
    deck = pdk.Deck(layers=layers, initial_view_state=view, map_style=pdk.map_styles.CARTO_LIGHT, tooltip=tooltip)
    with c1:
        event = st.pydeck_chart(deck, height=620, on_select="rerun", selection_mode="single-object", key="map")

    # ---- legend: one row per active layer
    rows = []
    what = {"All collisions": "collisions", "Fatal & serious only": "fatal & serious collisions",
            "Fatal only": "fatal collisions"}
    if heat_mode != "Off":
        rows.append(gradient_row(f"Heatmap – density of {what[heat_mode]}"))
    if show_hex:
        rows.append(gradient_row("3D columns – collisions per hexagon (taller = more)"))
    if show_ksi:
        rows += [legend_row(swatch("dot", "#7a0f2a"), "Fatal collision"),
                 legend_row(swatch("dot", ORANGE), "Serious collision")]
    if show_gi:
        rows += [legend_row(swatch("square", rgb(GI_99)), "Gi* hot spot – 99% confidence"),
                 legend_row(swatch("square", rgb(GI_95)), "Gi* hot spot – 95% confidence")]
    if show_km:
        rows.append(legend_row(swatch("circle", TEAL, NAVY), "K-Means region centre (size = fatal & serious crashes)"))
    if show_dbf:
        rows.append(legend_row(swatch("shape", "#7a0f2a"), "DBSCAN fatal cluster – ≥ 4 fatal crashes within 1 km"))
    if show_dbk:
        rows.append(legend_row(swatch("shape", ORANGE), "DBSCAN junction cluster – ≥ 12 fatal/serious within 150 m"))
    with c2:
        st.markdown("**Legend**")
        st.markdown("".join(rows) if rows else "<p class='note'>Switch on a layer to see its legend.</p>",
                    unsafe_allow_html=True)

    sel = (event.selection.get("objects") or {}) if event is not None and hasattr(event, "selection") else {}
    picked = next(((lid, objs[0]) for lid, objs in sel.items() if objs), None)
    if picked:
        lid, obj = picked
        with st.container(border=True):
            st.markdown(f"**📍 Selected:** {obj.get('popup') or obj.get('label', '')}")
            if lid in ("dbscan_fatal_clusters", "dbscan_fatal_clusters_pt"):
                members = fdf[fdf.dbscan_fatal_cluster == obj["cluster"]]
            elif lid in ("dbscan_ksi_clusters", "dbscan_ksi_clusters_pt"):
                members = fdf[fdf.dbscan_ksi_cluster == obj["cluster"]]
            elif lid == "kmeans":
                members = fdf[fdf.kmeans_cluster == obj["kmeans_cluster"]]
            elif lid == "gi":
                members = fdf[fdf.grid_id == obj["grid_id"]]
            else:
                members = None
            if members is not None and len(members):
                st.caption(f"{len(members):,} collisions in this selection match the current filters – charts below show them.")
                linked_charts(members, "sel")
    st.markdown("##### Linked charts – current filters")
    linked_charts(fdf, "main")

# =========================================================================== 2 FATAL & RISK
with tabs[1]:
    st.subheader("Which areas are fatal, and which have high predicted severity?")
    dfx = apply_filters(pts, ignore_district=True)
    agg = (dfx.assign(fatal=(dfx.severity_code == 1).astype("int64"), ksi=(dfx.severity_code <= 2).astype("int64"))
           .groupby(["lad_code", "lad_name"], observed=True)
           .agg(collisions=("fatal", "size"), fatal=("fatal", "sum"), ksi=("ksi", "sum")).reset_index())
    agg[["collisions", "fatal", "ksi"]] = agg[["collisions", "fatal", "ksi"]].astype("int64")
    ds = S["district_stats"][["lad_code", "area_km2", "risk_index", "mean_p_ksi", "observed_ksi_rate_2025", "test_collisions"]]
    agg = agg.merge(ds, on="lad_code", how="left")
    agg["fatal_per_1000"] = 1000 * agg.fatal / agg.collisions
    agg["ksi_pct"] = 100 * agg.ksi / agg.collisions
    agg["fatal_per_100km2"] = 100 * agg.fatal / agg.area_km2
    METRICS = {"Fatal collisions (count)": ("fatal", "Reds"), "Fatal per 1,000 collisions (rate)": ("fatal_per_1000", "Reds"),
               "KSI share %": ("ksi_pct", "OrRd"), "Fatal per 100 km²": ("fatal_per_100km2", "Reds"),
               "Predicted severity risk index (RF, 2025)": ("risk_index", "RdBu_r")}
    mcol, ncol = st.columns([2, 1])
    choice = mcol.radio("Map metric", list(METRICS), horizontal=True)
    min_n = ncol.slider("Hide rates for districts with fewer collisions than", 0, 1000, 200, 50)
    col, scale = METRICS[choice]
    plot = agg.copy()
    if col in ("fatal_per_1000", "ksi_pct"):
        plot.loc[plot.collisions < min_n, col] = np.nan
    left, right = st.columns([3, 2])
    with left:
        kw = dict(color_continuous_midpoint=1.0) if col == "risk_index" else {}
        fig = px.choropleth_map(plot, geojson=S["districts_geo"], locations="lad_code", featureidkey="properties.lad_code",
                                color=col, color_continuous_scale=scale, hover_name="lad_name",
                                hover_data={"lad_code": False, "collisions": ":,", "fatal": ":,", "fatal_per_1000": ":.2f",
                                            "ksi_pct": ":.1f", "risk_index": ":.3f"},
                                map_style="carto-positron", center={"lat": 54.4, "lon": -2.8}, zoom=4.6, opacity=0.85, **kw)
        fig.update_layout(height=640, margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title=""))
        st.plotly_chart(fig, key="choro")
        if col == "risk_index":
            st.caption("Risk index = mean predicted P(fatal or serious) of the district's 2025 collisions ÷ national mean "
                       f"(> 1 = more severe than average). Spearman ρ with the observed 2025 KSI share across districts: "
                       f"{metrics['lad_risk_spearman']}. Not affected by the sidebar filters.")
    with right:
        st.markdown(f"**Top 15 districts – {choice}**")
        top = plot.dropna(subset=[col]).nlargest(15, col)
        st.dataframe(top[["lad_name", "collisions", "fatal", "fatal_per_1000", "ksi_pct", "risk_index"]],
                     hide_index=True, height=560,
                     column_config={"lad_name": "District", "collisions": st.column_config.NumberColumn("Collisions", format="%d"),
                                    "fatal": "Fatal", "fatal_per_1000": st.column_config.NumberColumn("Fatal / 1,000", format="%.2f"),
                                    "ksi_pct": st.column_config.NumberColumn("KSI %", format="%.1f"),
                                    "risk_index": st.column_config.NumberColumn("Pred. risk", format="%.3f")})
    st.markdown("##### Predicted vs observed severity on H3 hexagons (2025 test year)")
    h3 = S["h3_risk"]
    h3 = h3[h3.test_collisions >= 10].copy()
    h3_metric = st.radio("Hexagon colour", ["Predicted risk index", "Observed KSI share 2025"], horizontal=True)
    v = h3.risk_index if h3_metric.startswith("Pred") else h3.observed_ksi_rate_2025 / h3.observed_ksi_rate_2025.mean()
    t = ((v - 0.6) / 0.8).clip(0, 1)
    h3["r"] = (43 + t * (122 - 43)).astype(int); h3["g"] = (179 - t * 164).astype(int); h3["b"] = (163 - t * 121).astype(int)
    h3["popup"] = ("H3 " + h3.h3_cell + " · " + h3.test_collisions.astype(str) + " collisions (2025) · predicted risk "
                   + h3.risk_index.round(2).astype(str) + " · observed KSI " + (100 * h3.observed_ksi_rate_2025).round(1).astype(str) + "%")
    st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("H3HexagonLayer", h3, get_hexagon="h3_cell", get_fill_color=["r", "g", "b", 170],
                                               stroked=False, pickable=True)],
                             initial_view_state=GB_VIEW, map_style=pdk.map_styles.CARTO_LIGHT,
                             tooltip={"html": "{popup}", "style": {"backgroundColor": NAVY, "color": "white"}}), height=520)
    st.markdown(
        "<div style='max-width:520px;font-size:.86rem'>"
        f"<b>Legend</b> – {h3_metric.lower()} relative to the GB average"
        f"<div style='height:10px;border-radius:3px;margin-top:4px;background:linear-gradient(to right, {TEAL}, rgb(83,97,103), {FATAL})'></div>"
        "<div style='display:flex;justify-content:space-between;font-size:.75rem;color:#5b6472'>"
        "<span>0.6× (lower than average)</span><span>1.0× average</span><span>1.4× (higher than average)</span></div></div>",
        unsafe_allow_html=True)
    st.caption("Each hexagon is ~5 km² (H3 resolution 7). Hexagons with fewer than 10 collisions in 2025 are hidden.")

# =========================================================================== 3 HOTSPOTS
with tabs[2]:
    st.subheader("Four hotspot methods")
    g = S["grid_cells"]
    a, b, c, d = st.columns(4)
    a.metric("Occupied 5 km cells", f"{summary['occupied_cells']:,}")
    b.metric("Persistent high-count cells", f"{summary['persistent_cells']:,}",
             help=f"≥ {summary['grid_threshold']:.0f} collisions (95th percentile) in every year 2021–2025")
    c.metric("Gi* hot spots (99% / 95%)", f"{summary['gi_hot_99']} / {summary['gi_hot_95']}", help="FDR-corrected")
    d.metric("DBSCAN fatal / KSI clusters", f"{summary['dbscan_fatal_clusters']} / {summary['dbscan_ksi_clusters']}")
    l, r = st.columns(2)
    with l:
        tc = g.temporal_category.value_counts().reindex(["Persistent", "Emerging", "Declining", "Variable"]).reset_index()
        fig = px.bar(tc, x="temporal_category", y="count", title="Grid screening – high-count cells by temporal pattern",
                     color_discrete_sequence=[NAVY], text="count")
        st.plotly_chart(sev_colored(fig), key="tc")
        st.markdown("<p class='note'>Grid screening flags 5 km cells in the top 5% of collision counts each year. "
                    "<b>Persistent</b> = flagged every year, <b>Emerging</b> = not in 2021 but in 2025, "
                    "<b>Declining</b> = the reverse.</p>", unsafe_allow_html=True)
    with r:
        gc = g.gi_class.value_counts().reset_index()
        fig = px.bar(gc, x="gi_class", y="count", title="Getis-Ord Gi* classes (KSI counts)", text="count",
                     color="gi_class", color_discrete_map={"Hot spot (99%)": rgb(GI_99), "Hot spot (95%)": rgb(GI_95),
                                                           "Not significant": "#c9c6bd", "Cold spot (95%)": "#4575b4"})
        st.plotly_chart(sev_colored(fig).update_layout(showlegend=False), key="gc")
        st.markdown("<p class='note'>Gi* tests whether a cell <i>and its neighbours</i> have more KSI collisions "
                    "than expected by chance (z-score, Benjamini–Hochberg corrected). Unlike raw counts, it is a "
                    "statistical hotspot test.</p>", unsafe_allow_html=True)
    st.markdown("##### K-Means regions of KSI collisions (Spark MLlib)")
    km = S["kmeans_clusters"]
    scan = pd.DataFrame(metrics["kmeans_scan"])
    l, r = st.columns([1, 2])
    with l:
        fig = px.line(scan, x="k", y="silhouette", markers=True, title=f"Silhouette by k (chosen k = {summary['kmeans_k']} by elbow)")
        fig.add_vline(x=summary["kmeans_k"], line_dash="dash", line_color=ORANGE)
        st.plotly_chart(sev_colored(fig).update_traces(line_color=TEAL), key="km")
    with r:
        st.dataframe(km[["kmeans_cluster", "main_district", "ksi_collisions", "fatal", "fatal_share_pct"]], hide_index=True, height=330,
                     column_config={"kmeans_cluster": "Cluster", "main_district": "Main district", "ksi_collisions": "KSI collisions",
                                    "fatal": "Fatal", "fatal_share_pct": st.column_config.NumberColumn("Fatal share %", format="%.2f")})
    st.markdown("##### DBSCAN clusters (Apache Sedona)")
    l, r = st.columns(2)
    p = summary["dbscan_params"]
    with l:
        st.markdown(f"**Fatal clusters** · eps {p['fatal']['epsilon']:.0f} m · min {p['fatal']['min_pts']} fatal collisions")
        st.dataframe(S["dbscan_fatal_clusters"][["cluster", "district", "collisions", "first_year", "last_year"]],
                     hide_index=True, height=320, column_config={"collisions": "Fatal collisions"})
    with r:
        st.markdown(f"**KSI junction clusters** · eps {p['ksi']['epsilon']:.0f} m · min {p['ksi']['min_pts']} KSI collisions")
        st.dataframe(S["dbscan_ksi_clusters"][["cluster", "district", "collisions", "fatal"]], hide_index=True, height=320,
                     column_config={"collisions": "KSI collisions"})
    st.info("High counts reflect traffic volume as well as danger: STATS19 has no exposure data (vehicle-km), so "
            "hotspots show where severe crashes concentrate, not crash risk per journey.")

# =========================================================================== 4 PREDICTOR
with tabs[3]:
    st.subheader("Severity predictor – enter conditions, get predicted severity and risk")
    model = load_model()
    o = meta["options"]
    dflt = meta["defaults"]

    def pick(label, field, col):
        choices = list(o[field].keys())
        labels = o[field]
        idx = choices.index(str(dflt[field])) if str(dflt[field]) in choices else 0
        return int(col.selectbox(label, choices, index=idx, format_func=lambda c: labels[c], key=f"p_{field}"))

    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**When**")
        hour = c1.slider("Hour", 0, 23, 17)
        dow_name = c1.selectbox("Day of week", list(DOW.values()), index=5)
        month = c1.selectbox("Month", MONTHS, index=10)
        c2.markdown("**Road**")
        rc = pick("Road class", "first_road_class", c2)
        rt = pick("Road type", "road_type", c2)
        speed = int(c2.selectbox("Speed limit (mph)", [s for s in o["speed_limit"] if int(s) > 0],
                                 index=[s for s in o["speed_limit"] if int(s) > 0].index("30")))
        area = pick("Urban / rural", "urban_or_rural_area", c2)
        trunk = pick("Trunk road", "trunk_road_flag", c2)
        c3.markdown("**Junction & conditions**")
        jd = pick("Junction detail", "junction_detail", c3)
        jc = pick("Junction control", "junction_control", c3)
        pc = pick("Pedestrian crossing", "pedestrian_crossing", c3)
        lt = pick("Light", "light_conditions", c3)
        wt = pick("Weather", "weather_conditions", c3)
        sf = pick("Road surface", "road_surface_conditions", c3)
        hz = pick("Carriageway hazards", "carriageway_hazards", c3)
        pf = pick("Police force area", "police_force", c3)
        go_btn = st.form_submit_button("Predict severity", type="primary")

    dow_code = [k for k, v in DOW.items() if v == dow_name][0]
    row = {"hour": hour, "day_of_week": dow_code, "month": MONTHS.index(month) + 1, "is_weekend": int(dow_code in (1, 7)),
           "first_road_class": rc, "road_type": rt, "speed_limit": speed, "junction_detail": jd, "junction_control": jc,
           "pedestrian_crossing": pc, "light_conditions": lt, "weather_conditions": wt, "road_surface_conditions": sf,
           "carriageway_hazards": hz, "urban_or_rural_area": area, "trunk_road_flag": trunk, "police_force": pf}
    X = pd.DataFrame([row])[meta["features"]]
    for c_ in meta["categorical"]:
        X[c_] = pd.Categorical(X[c_].astype(int), categories=[int(v) for v in meta["categories"][c_]])
    proba = model.predict_proba(X)[0]
    classes = [meta["classes"][str(c_)] for c_ in model.classes_]
    pred = classes[int(np.argmax(proba))]
    p_ksi = float(proba[classes.index("Fatal")] + proba[classes.index("Serious")])
    q = meta["p_ksi_test_quantiles"]
    pct = max([int(k) for k, v in q.items() if v <= p_ksi] or [0])
    level, color = (("Very high", FATAL) if pct >= 95 else ("High", ORANGE) if pct >= 80 else
                    ("Elevated", "#c9a227") if pct >= 50 else ("Lower", TEAL))
    l, r = st.columns([1, 1.4])
    with l:
        st.markdown(f"Most likely class: **{pred}**")
        st.markdown(f"<span class='risk-badge' style='background:{color}'>Severity risk: {level}</span>", unsafe_allow_html=True)
        st.markdown(f"Predicted P(fatal or serious) is higher than for **{pct}%** of the collisions that happened in "
                    f"{meta['test_year']}.")
        st.caption("The model was trained with class weights to detect rare fatal crashes, so its raw probabilities "
                   "over-state severity. The percentile against real 2025 collisions is the meaningful number.")
    with r:
        pr = pd.DataFrame({"severity": classes, "probability": proba})
        fig = px.bar(pr, x="probability", y="severity", orientation="h", color="severity", color_discrete_map=SEV_COLORS,
                     category_orders={"severity": SEV_ORDER}, text=pr.probability.map(lambda v: f"{v:.0%}"),
                     title="Class probabilities (class-weighted model)")
        fig.update_layout(showlegend=False, xaxis_range=[0, 1], height=260)
        st.plotly_chart(sev_colored(fig), key="proba")
    mm = meta["metrics"]
    st.caption(f"Model: scikit-learn HistGradientBoosting trained on 2021–2024, tested on {meta['test_year']} · "
               f"macro F1 {mm['Macro F1']:.3f} · fatal recall {mm['Fatal recall']:.2f} · serious recall {mm['Serious recall']:.2f}. "
               "Research tool: not for operational decisions.")

# =========================================================================== 5 RULES
with tabs[4]:
    st.subheader("Association rule explorer (Spark MLlib FP-Growth)")
    rules = S["association_rules"]
    all_items = sorted({i for a in rules.antecedent for i in a.split(" AND ")})
    QUICK = {"🌧️ Rain": "Weather=Raining", "🌙 Darkness, unlit": "Light=Darkness - unlit or no lighting",
             "🛣️ Motorway": "Road class=Motorway", "🌾 Rural": "Area=Rural", "🕛 Night": "Time=Night (00–05)",
             "❄️ Wet road": "Surface=Wet or damp", "🏎️ 60 mph": "Speed=60 mph", "📅 Weekend": "Day=Weekend"}
    quick = st.pills("Quick filters (rule must contain all selected)", list(QUICK), selection_mode="multi")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    extra = c1.multiselect("Antecedent must contain", all_items, placeholder="Any condition")
    target = c2.selectbox("Consequent", ["Fatal", "Serious", "Slight"], index=0)
    min_lift = c3.number_input("Min lift", 0.0, 10.0, 1.2, 0.1)
    max_len = c4.selectbox("Max conditions", [1, 2, 3], index=2)
    must = set(extra) | {QUICK[q_] for q_ in (quick or [])}
    rr = rules[(rules.severity == target) & (rules.lift >= min_lift) & (rules.antecedent_size <= max_len)]
    if must:
        rr = rr[rr.antecedent.apply(lambda a: must.issubset(set(a.split(" AND "))))]
    rr = rr.sort_values("lift", ascending=False)
    base = summary[target.lower()] / summary["study_records"]
    st.caption(f"{len(rr):,} rules · base rate of {target}: {100 * base:.2f}% · minimum support "
               f"{100 * summary['fp_min_support']:.1f}% of collisions (≈ {summary['fp_min_support'] * summary['study_records']:,.0f})")
    l, r = st.columns([1.6, 1])
    with l:
        st.dataframe(rr[["antecedent", "support_count", "confidence", "lift"]].head(300), hide_index=True, height=480,
                     column_config={"antecedent": st.column_config.TextColumn("IF conditions…", width="large"),
                                    "support_count": st.column_config.NumberColumn("Collisions", format="%d"),
                                    "confidence": st.column_config.ProgressColumn(f"P({target} | conditions)", format="%.3f",
                                                                                  min_value=0, max_value=float(max(rr.confidence.max(), 0.01)) if len(rr) else 1),
                                    "lift": st.column_config.NumberColumn("Lift", format="%.2f×")})
    with r:
        if len(rr):
            fig = px.scatter(rr.head(2000), x="support_count", y="lift", color_discrete_sequence=[SEV_COLORS[target]],
                             hover_data={"antecedent": True}, log_x=True, title="Support vs lift",
                             labels={"support_count": "collisions matching rule (log)"})
            fig.add_hline(y=1, line_dash="dot", line_color="grey")
            st.plotly_chart(sev_colored(fig), key="rules")
    st.markdown("<p class='note'><b>Lift</b> = how many times more likely the outcome is under these conditions than "
                "on average. Rules show co-occurrence in police records, not causes.</p>", unsafe_allow_html=True)

# =========================================================================== 6 TIME
with tabs[5]:
    st.subheader("When do collisions happen? (updates with all filters)")
    kpis(fdf)
    l, r = st.columns(2)
    with l:
        hd = (fdf.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
              .reindex(index=range(1, 8), columns=range(24), fill_value=0))
        fig = px.imshow(hd.values, x=list(range(24)), y=[DOW[i] for i in hd.index], color_continuous_scale="Blues",
                        labels=dict(x="hour", y="", color="collisions"), aspect="auto", title="Collisions: day × hour")
        st.plotly_chart(fig, key="hd")
    with r:
        kd = (fdf.assign(k=fdf.severity_code <= 2).groupby(["day_of_week", "hour"]).k.mean().unstack()
              .reindex(index=range(1, 8), columns=range(24)) * 100)
        fig = px.imshow(kd.values, x=list(range(24)), y=[DOW[i] for i in kd.index], color_continuous_scale="OrRd",
                        labels=dict(x="hour", y="", color="KSI %"), aspect="auto", title="Severity: KSI % by day × hour")
        st.plotly_chart(fig, key="kd")
    l, r = st.columns(2)
    with l:
        mo = fdf.groupby(["year", "month"]).size().reset_index(name="collisions")
        mo["year"] = mo.year.astype(str)
        fig = px.line(mo, x="month", y="collisions", color="year", markers=True, title="Monthly collisions by year",
                      color_discrete_sequence=["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", NAVY])
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=MONTHS)
        st.plotly_chart(sev_colored(fig), key="mo")
    with r:
        se = fdf.groupby("season", observed=True).agg(collisions=("severity_code", "size"),
                                                      ksi=("severity_code", lambda s: 100 * (s <= 2).mean())).reindex(SEASONS).reset_index()
        fig = px.bar(se, x="season", y="ksi", text=se.ksi.round(1), title="KSI % by season", color_discrete_sequence=[TEAL])
        st.plotly_chart(sev_colored(fig), key="se")

# =========================================================================== 7 MODELS
with tabs[6]:
    st.subheader(f"Severity classification – chronological test on {metrics['test_year']}")
    mc = pd.DataFrame(metrics["multiclass"])
    st.dataframe(mc, hide_index=True, column_config={c_: st.column_config.NumberColumn(format="%.3f")
                                                     for c_ in mc.columns if c_ not in ("Model", "Train seconds")})
    st.caption("Main model: class-weighted Spark MLlib Random Forest. Macro F1 treats the three classes equally; "
               "accuracy is misleading here because predicting 'Slight' for everything already scores about 77%.")
    l, m_, r = st.columns(3)
    names = list(metrics["confusion_matrices"])
    with l:
        which = st.selectbox("Confusion matrix", names, index=names.index("Class-Weighted Random Forest"))
        cm = np.array(metrics["confusion_matrices"][which]); cmn = cm / cm.sum(axis=1, keepdims=True)
        fig = px.imshow(cmn, x=SEV_ORDER, y=SEV_ORDER, color_continuous_scale="Blues", zmin=0, zmax=1,
                        labels=dict(x="predicted", y="actual", color="share"), text_auto=".2f")
        st.plotly_chart(fig, key="cm")
    with m_:
        fi = pd.Series(metrics["rf_feature_importance"]).sort_values().reset_index()
        fi.columns = ["feature", "importance"]
        fig = px.bar(fi, x="importance", y="feature", orientation="h", title="Random Forest feature importance",
                     color_discrete_sequence=[TEAL])
        st.plotly_chart(sev_colored(fig).update_layout(height=480), key="fi")
    with r:
        st.markdown("**Binary KSI vs Slight**")
        st.dataframe(pd.DataFrame(metrics["binary_ksi"]).drop(columns=["Train seconds"]), hide_index=True)
        bm = pd.DataFrame(metrics["benchmark"])
        fig = px.line(bm, x="records", y="median_seconds", color="method", markers=True,
                      title="Spark vs Sedona: point-in-polygon join", color_discrete_sequence=[NAVY, TEAL, ORANGE])
        fig.update_layout(legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(sev_colored(fig), key="bm")

# =========================================================================== 8 ABOUT
with tabs[7]:
    st.subheader("Project pipeline")
    steps = [("1. Clean", "Duplicates, coordinates, missing codes"), ("2. Store", "Sedona KDB-tree partitioning, GeoParquet, PostGIS"),
             ("3. Wrangle", "Sedona spatial join to 363 districts, 5 km grid, H3, temporal features"),
             ("4. Model", "Random Forest / GBT severity, K-Means & DBSCAN & Gi* hotspots, FP-Growth rules"),
             ("5. Deliver", "This dashboard")]
    cols = st.columns(5)
    for c_, (t_, d_) in zip(cols, steps):
        c_.markdown(f"<div style='background:{NAVY};color:white;border-radius:10px;padding:14px;height:130px'>"
                    f"<b style='color:{TEAL}'>{t_}</b><br><span style='font-size:.85rem'>{d_}</span></div>", unsafe_allow_html=True)
    st.markdown(f"""
**Data.** UK Department for Transport STATS19 road collision data {summary['years']} ({summary['raw_records']:,} records;
{summary['missing_coordinates']} without coordinates; {summary['outside_gb']} outside Great Britain). District boundaries:
ONS Local Authority Districts (Dec 2021). Both under the Open Government Licence v3.0.

**Engine.** Apache Spark {summary['spark_version']} with Apache Sedona {summary['sedona_version']}.
PostGIS: {summary['postgis'].get('status')} {('– ' + str(summary['postgis'].get('rows', '')) + ' rows, ' + str(summary['postgis'].get('table_size', ''))) if summary['postgis'].get('status') == 'loaded' else ''}.

**Limitations.**
* No traffic-exposure data: counts and hotspots reflect traffic volume as well as danger.
* Severity is police-recorded and some conditions are recorded after the crash, so the model is not a validated pre-crash predictor.
* Fatal crashes are rare (~1.5%); fatal precision is low even when recall is reasonable.
* Hotspot results depend on grid size, Gi* neighbourhood and DBSCAN parameters (documented in the notebook).
* Association rules describe co-occurrence, not causation.
""")
    st.caption("Contains public sector information licensed under the Open Government Licence v3.0. "
               "Source: Department for Transport (STATS19); Office for National Statistics (boundaries).")
