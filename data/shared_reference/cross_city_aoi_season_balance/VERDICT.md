# City AOI: WC class balance + Sentinel-2 season counts

Geometry = bounding box of each city's 15k sample points (+0.02° pad).  
WC = ESA WorldCover v200 remapped to 3-class.  
S2 = COPERNICUS/S2_SR_HARMONIZED, cloudlt20 = CLOUDY_PIXEL_PERCENTAGE < 20.

## Important caveat

Training/eval **15k pools are stratified 5000/5000/5000** in every city — identical by design.  
AOI percentages below describe the **map footprint**, not the experiment class prior.

## Results

| City | Built% | Nonbuilt% | Water% | Built/(B+NB) | S2 2021 (<20% cloud) | MAM | JJA | SON | DJF |
|------|-------:|----------:|-------:|-------------:|---------------------:|----:|----:|----:|----:|
| Wuhan | 9.7 | 79.0 | 11.3 | 10.9 | 256 | 28 | 28 | 96 | 81 |
| Changsha | 7.2 | 90.9 | 1.9 | 7.4 | 256 | 21 | 36 | 108 | 77 |
| Nanchang | 6.9 | 69.2 | 23.9 | 9.0 | 207 | 25 | 20 | 86 | 66 |
| Hefei | 8.2 | 83.6 | 8.1 | 9.0 | 284 | 54 | 24 | 106 | 77 |
| **Nanjing** | **14.7** | 77.2 | 8.2 | **16.0** | 249 | 51 | 34 | 81 | 60 |

## Reading vs Nanjing OA failure

1. **Class balance (experiment):** identical → cannot explain PE gap.  
2. **Class balance (AOI):** Nanjing has **more** built, not less — opposite of “建成太少”.  
3. **Image count:** Nanjing mid-pack (249); not data-poor vs Wuhan/Changsha (256).  
4. **Season:** all cities thin in summer (cloud); Nanjing spring actually relatively rich (MAM=51). No unique seasonal desert.

→ Season / scene count / built–nonbuilt ratio are **unlikely root causes** of Nanjing’s WC-OA ceiling.
