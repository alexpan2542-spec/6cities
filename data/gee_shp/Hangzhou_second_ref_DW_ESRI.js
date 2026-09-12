// ============================================================================
// Hangzhou BAMS150 -- second / third reference sampling (Dynamic World + ESRI)
// ----------------------------------------------------------------------------
// Fills the one gap in
//   data/shared_reference/cross_city_second_ref_dw_esri/
// The other 5 cities have DW + ESRI class values at their 150 boundary points;
// Hangzhou does not. This samples, at Hangzhou's 150 BAMS points:
//   - Dynamic World  (2021 annual per-pixel majority label)
//   - ESRI Land Cover (Impact Observatory) 2021
//   - ESA WorldCover v200 (2021)               <- re-extracted, as a cross-check
// and exports the RAW product class codes. The 3-class (built/non-built/water)
// remap is done afterwards by scripts/add_hangzhou_second_ref.py, using the
// exact same mapping the 5-city file already encodes.
//
// PREREQUISITE -- upload the point table as an Earth Engine asset:
//   data/cities/Hangzhou/03_top150/Hangzhou_BAMS150.csv
//     Assets > New > CSV file (table upload)
//     x property = lon, y property = lat   (columns are also kept as properties)
//   then point POINTS_ASSET below at the resulting asset id.
//
// OUTPUT -- Export.table.toDrive -> Hangzhou_BAMS150_dwesri_raw.csv
//   columns: PointID, Original_ID, lon, lat, DW_raw, ESRI_raw, WC_ee_raw
// Move that file to
//   data/shared_reference/cross_city_second_ref_dw_esri/Hangzhou_BAMS150_dwesri_raw.csv
// and run scripts/add_hangzhou_second_ref.py.
// ============================================================================

var POINTS_ASSET = 'projects/healthy-area-463312-i4/assets/Hangzhou_BAMS150';
var YEAR_START = '2021-01-01';
var YEAR_END   = '2022-01-01';   // exclusive upper bound

// ---- points: rebuild geometry from lon/lat props (same idiom as the *_inspect.js)
var raw = ee.FeatureCollection(POINTS_ASSET);
var points = raw.map(function (f) {
  return ee.Feature(
    ee.Geometry.Point([ee.Number(f.get('lon')), ee.Number(f.get('lat'))]),
    f.toDictionary()
  );
});
print('n points (expect 150):', points.size());

// ---- Dynamic World: 2021 annual majority label ----------------------------
// DW is per-scene and already per-pixel cloud-masked; take the mode of the
// 'label' band over all 2021 scenes. No extra cloud filter (DW V1 does not
// reliably carry CLOUDY_PIXEL_PERCENTAGE; masking is already per pixel).
// label: 0 water, 1 trees, 2 grass, 3 flooded_veg, 4 crops, 5 shrub,
//        6 built, 7 bare, 8 snow/ice
var dwRaw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(points)
  .filterDate(YEAR_START, YEAR_END)
  .select('label')
  .mode()
  .rename('DW_raw');
print('DW scenes over Hangzhou 2021:',
      ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        .filterBounds(points).filterDate(YEAR_START, YEAR_END).size());

// ---- ESRI / Impact Observatory Land Cover, 2021 -------------------------
// sat-io time-series mirror; one image per year, single band 'b1', values 1..11:
// 1 water, 2 trees, 4 flooded_veg, 5 crops, 7 built, 8 bare,
// 9 snow/ice, 10 clouds, 11 rangeland
var esriRaw = ee.ImageCollection(
    'projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS')
  .filterDate(YEAR_START, YEAR_END)
  .mosaic()
  .rename('ESRI_raw');

// ---- ESA WorldCover v200 (2021) --------------------------------------
// values 10..100 (10 tree,20 shrub,30 grass,40 crop,50 built,60 bare/sparse,
// 70 snow,80 water,90 herb.wetland,95 mangrove,100 moss)
var wcRaw = ee.ImageCollection('ESA/WorldCover/v200').first()
  .select('Map')
  .rename('WC_ee_raw');

// ---- stack + sample at 10 m ------------------------------------------
var stack = dwRaw.toInt16()
  .addBands(esriRaw.toInt16())
  .addBands(wcRaw.toInt16());

var sampled = stack.reduceRegions({
  collection: points,
  reducer: ee.Reducer.first(),
  scale: 10,
  tileScale: 4
});

// ---- sanity preview ------------------------------------------------
print('first 5 sampled:', sampled.limit(5));
Map.setOptions('HYBRID');
Map.centerObject(points, 11);
Map.addLayer(points, {color: 'red'}, 'Hangzhou BAMS150');
Map.addLayer(dwRaw, {min: 0, max: 8}, 'DW label 2021 mode', false);

// ---- export -------------------------------------------------------
Export.table.toDrive({
  collection: sampled,
  description: 'Hangzhou_BAMS150_dwesri_raw',
  fileNamePrefix: 'Hangzhou_BAMS150_dwesri_raw',
  fileFormat: 'CSV',
  selectors: ['PointID', 'Original_ID', 'lon', 'lat',
              'DW_raw', 'ESRI_raw', 'WC_ee_raw']
});
