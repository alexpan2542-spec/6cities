// ============================================================
// Nanjing BAMS150 — export true 3x3 (center + 8-neighbor) info
// ============================================================
//
 // Purpose
//   Only for the 150 BAMS points (NOT all 15000).
 //   Export ESA WorldCover class in a 3x3 window at 10 m,
//   remapped to your 3-class scheme, to help re-label uncertain points.
//
// Steps
//   1) Upload data/gee_shp/Nanjing_BAMS150.zip to Assets
 //      (or use your existing Nanjing_BAMS150 asset)
//   2) Set ASSET / EXPORT_FOLDER below
//   3) Run. Check Tasks tab → wait for CSV on Google Drive
//
// Runtime
//   150 points is small; usually a few minutes, not hours.
// ============================================================

// ---- edit these ----
var ASSET = 'projects/healthy-area-463312-i4/assets/Nanjing_BAMS150';
var EXPORT_FOLDER = 'gee_exports';           // Google Drive folder
var EXPORT_PREFIX = 'Nanjing_BAMS150_3x3nb'; // output name prefix
var YEAR = 2021;                             // S2 year for RGB context layers

// ============================================================
// 1) Load BAMS points (rebuild geometry from lon/lat)
// ============================================================
var table = ee.FeatureCollection(ASSET);

var points = table.map(function(f) {
  return ee.Feature(
    ee.Geometry.Point([
      ee.Number(f.get('lon')),
      ee.Number(f.get('lat'))
    ]),
    f.toDictionary()
  );
});

print('BAMS points:', points.size()); // should be 150

// ============================================================
 // 2) WorldCover 10 m → your 3-class map
//    Class1 = built-up (50)
//    Class3 = water / wetland (80, 90, 95)
//    Class2 = everything else
 // ============================================================
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');

var wc3 = wc.remap(
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
  [ 2,  2,  2,  2,  1,  2,  2,  3,  3,  3,   2]
).rename('C3').toInt();

// Keep raw WC too (optional diagnostics)
var wcRaw = wc.rename('WC');

// ============================================================
// 3) Build 3x3 neighborhood bands (center + 8 neighbors)
 //    Kernel radius=1 pixel → 3x3 window on 10 m grid
// ============================================================
var kernel = ee.Kernel.square({radius: 1, units: 'pixels', normalize: false});

// neighborhoodToBands names look like: C3_-1_-1, C3_0_0, C3_1_1, ...
var nb3 = wc3.neighborhoodToBands(kernel);
var nbWC = wcRaw.neighborhoodToBands(kernel);

// Helpful summaries from the 9 neighborhood class values
var bandList = nb3.bandNames();

function countEq(img, val, outName) {
  return img.eq(val).reduce(ee.Reducer.sum()).rename(outName);
}

var nC1 = countEq(nb3, 1, 'n_C1');
var nC2 = countEq(nb3, 2, 'n_C2');
var nC3 = countEq(nb3, 3, 'n_C3');
var hasC1 = nC1.gt(0).rename('has_C1');
var hasC3 = nC3.gt(0).rename('has_C3');

// Center pixel only (offset 0,0)
var centerC3 = nb3.select('C3_0_0').rename('center_C3');
var centerWC = nbWC.select('WC_0_0').rename('center_WC');

// Optional: majority of 8 neighbors EXCLUDING center
// (approx: total counts minus center)
var nC1_nb8 = nC1.subtract(centerC3.eq(1)).rename('n_C1_nb8');
var nC2_nb8 = nC2.subtract(centerC3.eq(2)).rename('n_C2_nb8');
var nC3_nb8 = nC3.subtract(centerC3.eq(3)).rename('n_C3_nb8');
var hasC1_nb8 = nC1_nb8.gt(0).rename('has_C1_nb8');

var stack = nb3
  .addBands(nbWC)
  .addBands([
    centerC3, centerWC,
    nC1, nC2, nC3, hasC1, hasC3,
    nC1_nb8, nC2_nb8, nC3_nb8, hasC1_nb8
  ]);

// ============================================================
// 4) Sample at BAMS points (scale=10 m)
// ============================================================
var sampled = stack.reduceRegions({
  collection: points,
  reducer: ee.Reducer.first(),
  scale: 10,
  tileScale: 2
});

// Suggested decision helpers (for your re-label protocol)
var sampled2 = sampled.map(function(f) {
  var hasBuildNb = ee.Number(f.get('has_C1_nb8'));
  var suggest = ee.Algorithms.If(
    hasBuildNb.eq(1),
    1,   // any of 8 neighbors is built-up → lean Class1
    2    // otherwise lean Class2 (for uncertain C1/C2 cases)
  );
  return f.set({
    suggest_C_if_uncertain: suggest,
    note: 'Use suggest_C_if_uncertain ONLY when center is visually ambiguous between C1/C2; always confirm on imagery.'
  });
});

print('Sample feature (first):', sampled2.first());
print('Property names:', sampled2.first().propertyNames());

// ============================================================
 // 5) Map preview (optional)
// ============================================================
Map.setOptions('HYBRID');
Map.centerObject(points, 10);
Map.addLayer(wc3, {min: 1, max: 3, palette: ['red', 'green', 'blue']}, 'WC 3-class', false);
Map.addLayer(points, {color: 'yellow'}, 'BAMS150');

// ============================================================
// 6) Export to Drive (CSV)
 // ============================================================
Export.table.toDrive({
  collection: sampled2,
  description: EXPORT_PREFIX + '_wide',
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_PREFIX + '_wide',
  fileFormat: 'CSV',
  selectors: null  // export all properties
});

print('>>> Go to Tasks panel, run: ' + EXPORT_PREFIX + '_wide');
print('>>> Drive folder:', EXPORT_FOLDER);
print('>>> Expected rows: ~150 (one per BAMS point)');
print('>>> Key fields:');
print('    center_C3, C3_-1_-1 ... C3_1_1  (3x3 remapped classes)');
print('    n_C1_nb8 / has_C1_nb8          (8-neighbor built-up evidence)');
print('    suggest_C_if_uncertain         (1 if any nb is C1 else 2)');
