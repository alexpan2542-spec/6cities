#!/usr/bin/env python3
"""
Figure 1 - Study area locator map.  [rewritten 2026-09-11]

Panel (a): the six study cities in the middle-lower Yangtze. The six provinces
           that contain them are shaded and named; the Yangtze main stem is
           drawn. Each city's analysis AOI is its FAO GAUL 2015 ADM2
           (prefecture) polygon, with the 150 BAMS boundary points inside.
Panel (b): China locator (Albers). The same six provinces (+ Shanghai) are
           highlighted and the panel-(a) frame is boxed; one parallel / meridian
           is drawn for reference. Tops of (a) and (b) are aligned.

Panel (a) is geographic (WGS84) with the vertical exaggeration corrected to the
mean latitude; the inset uses Albers Equal Area for China (SP 25N/47N, CM 105E).

NOTE (open): the China locator does not yet use an approved national base map
(South China Sea islands etc.). Revisit before submission if the institutional
map-review rules require it.

Inputs  : data/boundaries/ne_admin1/ne_10m_admin_1_states_provinces.shp   (inset context)
          data/boundaries/gaul_l1_yreb6.geojson    (6 provinces + Shanghai, GAUL L1)
          data/boundaries/gaul_l2_cities6.geojson  (6 prefecture AOIs, GAUL L2)
          data/boundaries/ne_rivers/ne_10m_rivers_lake_centerlines.shp
          data/shared_reference/cross_city_second_ref_dw_esri/all_cities_BAMS150_second_ref.csv
Output  : submission/figures/Figure1_study_area.png  (+ .pdf)

The GAUL geojson + NE rivers were exported once with
scripts_bk/... / a one-off ee export; regenerate only if the AOIs change.

Run with the `gee` conda env:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 scripts/make_study_area_map.py
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from shapely.geometry import Polygon, box, Point

# ----------------------------------------------------------------------------- config
WGS84 = "EPSG:4326"
ALBERS = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"

CITIES = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha", "Hangzhou"]

# municipal seat (downtown) - anchors the star + label leader
SEAT = {
    "Wuhan":    (114.305, 30.593),
    "Hefei":    (117.283, 31.861),
    "Nanchang": (115.858, 28.683),
    "Nanjing":  (118.797, 32.060),
    "Changsha": (112.939, 28.228),
    "Hangzhou": (120.155, 30.274),
}
# label anchor in data deg relative to the seat (dlon, dlat) + alignment
LABEL_KW = {
    "Wuhan":    dict(dlon=-1.35, dlat=+0.15, ha="right",  va="center"),
    "Hefei":    dict(dlon=+0.20, dlat=+1.25, ha="left",   va="bottom"),
    "Nanjing":  dict(dlon=+1.25, dlat=+0.15, ha="left",   va="center"),
    "Nanchang": dict(dlon=-0.10, dlat=-1.30, ha="center", va="top"),
    "Changsha": dict(dlon=-1.45, dlat=-0.25, ha="right",  va="center"),
    "Hangzhou": dict(dlon=+1.30, dlat=-0.55, ha="left",   va="center"),
}

# province short name -> label anchor (data deg) + alignment
PROV_LABEL = {
    "Hubei Sheng":    (110.15, 31.75, "center", "center"),
    "Hunan Sheng":    (110.70, 26.60, "center", "center"),
    "Jiangxi Sheng":  (115.55, 26.60, "center", "center"),
    "Anhui Sheng":    (116.10, 33.35, "center", "center"),
    "Jiangsu Sheng":  (119.35, 33.95, "center", "center"),
    "Zhejiang Sheng": (119.75, 28.55, "center", "center"),
    "Shanghai Shi":   (121.35, 31.55, "right",  "center"),
}
PROV_SHORT = {k: k.split()[0] for k in PROV_LABEL}

C_YREB_FILL = "#e3ede5"
C_YREB_EDGE = "#5f8f73"
C_CN_FILL = "#f2f2ef"
C_CN_EDGE = "#c4c4be"
C_BOX = "#c2410c"
C_AOI_FILL = "#f2b48c"
C_PT = "#1d4e74"
C_RIVER = "#4a86c5"
C_EXTENT = "#c2410c"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.linewidth": 0.7, "savefig.dpi": 400,
})

# ----------------------------------------------------------------------------- data
prov = gpd.read_file("data/boundaries/gaul_l1_yreb6.geojson").set_crs(WGS84)
aoi = gpd.read_file("data/boundaries/gaul_l2_cities6.geojson").set_crs(WGS84)
aoi = aoi.set_index("City").loc[CITIES]

ne = gpd.read_file("data/boundaries/ne_admin1/ne_10m_admin_1_states_provinces.shp")
cn = ne[ne["admin"] == "China"].to_crs(WGS84)

riv = gpd.read_file("data/boundaries/ne_rivers/ne_10m_rivers_lake_centerlines.shp")
yangtze = riv[riv["name_en"].astype(str).eq("Yangtze")].to_crs(WGS84)

# locator base: all countries in the region, each with its own national border
# (China included - no special treatment, just one country among its neighbours)
countries_ea = gpd.read_file(
    "data/boundaries/ne_countries_eastasia.geojson").set_crs(WGS84)

pts = pd.read_csv("data/shared_reference/cross_city_second_ref_dw_esri/"
                  "all_cities_BAMS150_second_ref.csv")

# ----------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.3, 5.0))
AXM_POS = [0.060, 0.080, 0.925, 0.870]
axm = fig.add_axes(AXM_POS)

# ---- panel (a): geographic ----
minx, miny, maxx, maxy = prov.total_bounds
padx = (maxx - minx) * 0.015
pady = (maxy - miny) * 0.02
x0 = minx - padx
x1 = maxx + (maxx - minx) * 0.42          # extra East China Sea for the inset
y0, y1 = miny - pady, maxy + pady
mean_lat = 0.5 * (y0 + y1)

axm.set_facecolor("#dCE7EF")             # sea

# embedded China locator over the sea, top-right, with an equal top/right gap.
# size the axes box to the locator window's own aspect so it does not letterbox.
LOC_WIN = (73, 9.5, 135, 54)            # lon0, lat0, lon1, lat1
_wb = gpd.GeoSeries([box(*LOC_WIN)], crs=WGS84).to_crs(ALBERS).total_bounds
_axw_in, _axh_in = AXM_POS[2] * 7.3, AXM_POS[3] * 5.0
_gap_in, _ins_w_in = 0.15, 1.95
_ins_h_in = _ins_w_in * (_wb[3] - _wb[1]) / (_wb[2] - _wb[0])
axi = axm.inset_axes([
    1 - (_gap_in + _ins_w_in) / _axw_in, 1 - (_gap_in + _ins_h_in) / _axh_in,
    _ins_w_in / _axw_in, _ins_h_in / _axh_in])
axi.patch.set_facecolor("#dCE7EF")      # sea (same as panel a)

cn.plot(ax=axm, facecolor=C_CN_FILL, edgecolor=C_CN_EDGE, lw=0.4, zorder=1)
prov.plot(ax=axm, facecolor=C_YREB_FILL, edgecolor=C_YREB_EDGE, lw=0.7, zorder=2)
yangtze.plot(ax=axm, color=C_RIVER, lw=1.5, zorder=3, alpha=0.9)

for name, (px, py, pha, pva) in PROV_LABEL.items():
    axm.text(px, py, PROV_SHORT[name], ha=pha, va=pva, fontsize=8,
             style="italic", color="#3f6b52", zorder=4,
             path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

for c in CITIES:
    gpd.GeoSeries([aoi.loc[c].geometry], crs=WGS84).plot(
        ax=axm, facecolor=C_AOI_FILL, edgecolor=C_BOX, lw=1.0, alpha=0.55, zorder=5)
    d = pts[pts.City == c]
    axm.scatter(d.lon, d.lat, s=1.4, c=C_PT, alpha=0.65, linewidths=0, zorder=6)
    sx, sy = SEAT[c]
    axm.scatter([sx], [sy], s=46, marker="*", c="black",
                edgecolors="white", linewidths=0.5, zorder=8)
    k = LABEL_KW[c]
    lx, ly = sx + k["dlon"], sy + k["dlat"]
    axm.annotate("", xy=(sx, sy), xytext=(lx, ly),
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="0.35"), zorder=7)
    axm.text(lx, ly, c, ha=k["ha"], va=k["va"], fontsize=8.5, fontweight="bold",
             zorder=9,
             path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

# Yangtze label, angled roughly along the reach between Wuhan and Nanjing
axm.text(115.9, 29.55, "Yangtze River", rotation=20, rotation_mode="anchor",
         ha="center", va="center", fontsize=7.5, style="italic", color=C_RIVER,
         zorder=4, path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

axm.set_xlim(x0, x1)
axm.set_ylim(y0, y1)
axm.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
xt = np.arange(108, 127, 3)
yt = np.arange(26, 36, 2)
axm.set_xticks(xt); axm.set_yticks(yt)
axm.set_xticklabels([f"{v:g}°E" for v in xt], fontsize=7)
axm.set_yticklabels([f"{v:g}°N" for v in yt], fontsize=7)
axm.grid(True, color="0.75", lw=0.3, ls=(0, (2, 3)), zorder=0)
axm.tick_params(length=2.5)

# scale bar (km -> deg lon at mean lat); 4 x 100 km segments
km_per_deg = 111.320 * np.cos(np.deg2rad(mean_lat))
seg_km, n_seg = 100, 4
seg_deg = seg_km / km_per_deg
sbx = x0 + 0.055 * (x1 - x0)
sby = y0 + 0.050 * (y1 - y0)
sbh = 0.012 * (y1 - y0)
for i in range(n_seg):
    axm.add_patch(Rectangle((sbx + i * seg_deg, sby), seg_deg, sbh,
                            facecolor="black" if i % 2 == 0 else "white",
                            edgecolor="black", lw=0.6, zorder=10))
for frac in (0, n_seg):
    axm.text(sbx + frac * seg_deg, sby - sbh * 0.6,
             "0" if frac == 0 else f"{n_seg*seg_km:g} km",
             ha="center", va="top", fontsize=6.3, zorder=10)

# north arrow (top-left, clear of provinces)
nax = x0 + 0.05 * (x1 - x0)
nay = y1 - 0.135 * (y1 - y0)
ndy = 0.070 * (y1 - y0)
axm.add_patch(FancyArrow(nax, nay, 0, ndy, width=0, head_width=ndy * 0.42,
                         head_length=ndy * 0.5, length_includes_head=True,
                         facecolor="black", edgecolor="black", zorder=10))
axm.text(nax, nay + ndy * 1.12, "N", ha="center", va="bottom",
         fontsize=9, fontweight="bold", zorder=10)

axm.set_title("Six study cities in the middle–lower Yangtze",
              fontsize=9, loc="left", pad=5)

leg = [
    Line2D([0], [0], marker="*", color="none", markerfacecolor="black",
           markeredgecolor="white", markersize=10, label="City centre"),
    Patch(facecolor=C_AOI_FILL, edgecolor=C_BOX, alpha=0.55,
          label="City AOI (ADM2 prefecture)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_PT,
           markeredgecolor="none", markersize=4, label="150 BAMS boundary points"),
    Line2D([0], [0], color=C_RIVER, lw=1.5, label="Yangtze River"),
    Patch(facecolor=C_YREB_FILL, edgecolor=C_YREB_EDGE,
          label="Provinces of the six cities"),
]
axm.legend(handles=leg, loc="lower right", bbox_to_anchor=(0.978, 0.015),
           bbox_transform=axm.transAxes, borderaxespad=0,
           fontsize=6.6, frameon=True, framealpha=0.92, borderpad=0.6,
           handletextpad=0.5, edgecolor="0.7").set_zorder(11)

# ---- embedded locator: every country shown, each with its own border -------
countries_ea.to_crs(ALBERS).plot(ax=axi, facecolor=C_CN_FILL,
                                 edgecolor=C_CN_EDGE, lw=0.4, zorder=2)
prov.to_crs(ALBERS).dissolve().plot(ax=axi, facecolor=C_YREB_FILL,
                                    edgecolor=C_YREB_EDGE, lw=0.7,
                                    zorder=3)               # study provinces

# "China" label - light grey, unobtrusive; placed over empty NW territory so it
# does not sit on the study-area box or crowd the coastline/neighbours.
_cn_lbl = gpd.GeoSeries([Point(97.0, 37.5)], crs=WGS84).to_crs(ALBERS).iloc[0]
axi.text(_cn_lbl.x, _cn_lbl.y, "China", ha="center", va="center", fontsize=6.5,
          style="italic", color="#8a8a84", zorder=5,
          path_effects=[pe.withStroke(linewidth=1.4, foreground="#f7f7f4")])

# study-area box (province bounds), densified so it curves under reprojection
n = 60
bx0, by0, bx1, by1 = minx, miny, maxx, maxy
ex = np.concatenate([np.linspace(bx0, bx1, n), np.full(n, bx1),
                     np.linspace(bx1, bx0, n), np.full(n, bx0)])
ey = np.concatenate([np.full(n, by0), np.linspace(by0, by1, n),
                     np.full(n, by1), np.linspace(by1, by0, n)])
gpd.GeoSeries([Polygon(zip(ex, ey))], crs=WGS84).to_crs(ALBERS).boundary.plot(
    ax=axi, color="black", lw=0.9, zorder=4)

axi.set_xlim(_wb[0], _wb[2]); axi.set_ylim(_wb[1], _wb[3])
axi.set_aspect("equal")
axi.set_xticks([]); axi.set_yticks([])
for s in axi.spines.values():
    s.set_linewidth(0.8); s.set_edgecolor("0.35")

fig.savefig("submission/figures/Figure1_study_area.png", bbox_inches="tight")
fig.savefig("submission/figures/Figure1_study_area.pdf", bbox_inches="tight")
print("wrote submission/figures/Figure1_study_area.png / .pdf")
