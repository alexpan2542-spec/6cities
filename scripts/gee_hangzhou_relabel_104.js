/**** Hangzhou standardisation -> label the 104 new two-stage BAMS150 points *****
 *
 * Paste into the GEE Code Editor (code.earthengine.google.com).
 *
 * WHY: Hangzhou's 150 boundary points were drawn with the early weighted-blend
 * BAMS variant. Standardising Hangzhou onto the frozen two-stage rule (1000
 * lowest-margin -> 150 largest BoundaryScore, used by the other five cities)
 * keeps only 46 of the 150; these 104 points are NEW and need an expert label
 * under the same protocol as the other cities. The 46 carry-over points keep
 * their existing labels (data/analysis_outputs/hangzhou_standardise/
 * keep_labelled.csv) and are NOT re-done here.
 *
 * PROTOCOL (identical to the 900-point labelling, manuscript Section 3.1):
 *   - primary reference = the 2021 Sentinel-2 composite itself, viewed as
 *       true colour  B4/B3/B2  stretch 0-3000
 *       false colour B8/B4/B3  stretch 0-4000   (veg red, water black)
 *   - ESA WorldCover v200 shown for context only -- do NOT anchor on it,
 *     the shown WC class is exactly the value BAMS was selecting against
 *   - Google HYBRID sub-metre basemap = ancillary spatial context only
 *     (form / edge localisation when the S2 view is insufficient)
 *   - judge the DOMINANT cover inside the 10 m pixel (red box)
 *   - assign: 1 built-up | 2 non-built | 3 water
 *             confidence: h / m / l
 *             scene code (+ optional free-text note)
 *
 * WORKFLOW: Prev/Next to step. Click a CLASS button, a CONFIDENCE button, and
 * a SCENE button; the row records automatically once class + confidence are
 * set. Add an optional note. At the end press "Dump log" and copy the console
 * CSV block. It has the same columns as {City}_BAMS150_Manual.csv, so it
 * concatenates directly with keep_labelled.csv into the new 150-row file.
 ******************************************************************************/

var YEAR = 2021;
var ZOOM = 18;                                  // ~1:2000; wheel-zoom further as needed

// ----------------------------------------------------------------------------
// 104 points  (data/analysis_outputs/hangzhou_standardise/to_label.csv)
//   wc = ESA WorldCover v200 remapped class currently at that pixel
//        (1 built / 2 non-built / 3 water) -- context only, do NOT anchor on it
// ----------------------------------------------------------------------------
var PTS = [
  {pid:  1, oid:   61, lon:119.03698684, lat:29.32175093, wc:1, margin:0.537, bscore:1596},
  {pid:  2, oid:  172, lon:119.20641039, lat:29.42775268, wc:1, margin:0.650, bscore:1461},
  {pid:  3, oid:  224, lon:118.93978736, lat:29.46395166, wc:1, margin:0.693, bscore:1009},
  {pid:  4, oid:  400, lon:119.03438273, lat:29.60489939, wc:1, margin:0.487, bscore:1333},
  {pid:  5, oid:  404, lon:119.02962041, lat:29.60624604, wc:1, margin:0.527, bscore:1222},
  {pid:  6, oid:  442, lon:119.53509964, lat:29.62942444, wc:1, margin:0.500, bscore:1107},
  {pid:  7, oid:  467, lon:119.21179698, lat:29.65089056, wc:1, margin:0.617, bscore:1320},
  {pid:  8, oid:  489, lon:119.10903254, lat:29.68466825, wc:1, margin:0.440, bscore:2111},
  {pid:  9, oid:  530, lon:119.18556855, lat:29.74368891, wc:1, margin:0.680, bscore:1327},
  {pid: 10, oid:  583, lon:119.20425218, lat:29.78267483, wc:1, margin:0.407, bscore:1013},
  {pid: 11, oid:  601, lon:119.67002758, lat:29.79049163, wc:1, margin:0.527, bscore:1131},
  {pid: 12, oid:  622, lon:119.7258154, lat:29.80171965, wc:1, margin:0.680, bscore:1120},
  {pid: 13, oid: 1658, lon:120.25025027, lat:30.120532, wc:1, margin:0.673, bscore:1250},
  {pid: 14, oid: 1874, lon:120.24971072, lat:30.1550276, wc:1, margin:0.690, bscore:1714},
  {pid: 15, oid: 1978, lon:119.21071788, lat:30.16706272, wc:1, margin:0.633, bscore:1011},
  {pid: 16, oid: 2003, lon:120.13158467, lat:30.16877055, wc:1, margin:0.493, bscore:1198},
  {pid: 17, oid: 2071, lon:119.75788525, lat:30.17559744, wc:1, margin:0.620, bscore:2011},
  {pid: 18, oid: 2088, lon:120.30882056, lat:30.17613699, wc:1, margin:0.660, bscore:1671},
  {pid: 19, oid: 2333, lon:119.67981977, lat:30.1930236, wc:1, margin:0.457, bscore:1113},
  {pid: 20, oid: 2482, lon:120.23516067, lat:30.20470199, wc:1, margin:0.567, bscore:1619},
  {pid: 21, oid: 2532, lon:120.51740839, lat:30.20838521, wc:1, margin:0.613, bscore:1133},
  {pid: 22, oid: 2657, lon:119.70434482, lat:30.22024641, wc:1, margin:0.677, bscore:1123},
  {pid: 23, oid: 2795, lon:120.52477483, lat:30.23264271, wc:1, margin:0.563, bscore:1121},
  {pid: 24, oid: 2972, lon:120.19284826, lat:30.24629648, wc:1, margin:0.693, bscore:1172},
  {pid: 25, oid: 3348, lon:119.88796169, lat:30.27926262, wc:1, margin:0.447, bscore:1172},
  {pid: 26, oid: 3506, lon:120.47842247, lat:30.29165892, wc:1, margin:0.667, bscore:1310},
  {pid: 27, oid: 3619, lon:120.10265849, lat:30.30172311, wc:1, margin:0.687, bscore:1899},
  {pid: 28, oid: 3669, lon:120.33145942, lat:30.30594588, wc:1, margin:0.633, bscore:1096},
  {pid: 29, oid: 4069, lon:120.29947875, lat:30.34654599, wc:1, margin:0.527, bscore:1058},
  {pid: 30, oid: 4525, lon:119.84726793, lat:30.41005697, wc:1, margin:0.453, bscore:1032},
  {pid: 31, oid: 4961, lon:120.29920674, lat:30.51390497, wc:1, margin:0.680, bscore:1058},
  {pid: 32, oid: 4967, lon:120.24369093, lat:30.51650909, wc:1, margin:0.710, bscore:1772},
  {pid: 33, oid: 5759, lon:118.88247453, lat:29.48596178, wc:2, margin:0.680, bscore:1151},
  {pid: 34, oid: 5841, lon:118.76165073, lat:29.50518497, wc:2, margin:0.367, bscore:1059},
  {pid: 35, oid: 6244, lon:118.94005936, lat:29.60804305, wc:2, margin:0.390, bscore:1188},
  {pid: 36, oid: 6800, lon:119.6796414, lat:29.744496, wc:2, margin:0.423, bscore:1041},
  {pid: 37, oid: 6825, lon:119.10336948, lat:29.74961505, wc:2, margin:0.603, bscore:1383},
  {pid: 38, oid: 6836, lon:119.0879187, lat:29.75285236, wc:2, margin:0.663, bscore:1142},
  {pid: 39, oid: 7416, lon:119.5224358, lat:29.87412207, wc:2, margin:0.633, bscore:1199},
  {pid: 40, oid: 7510, lon:119.60696698, lat:29.89352808, wc:2, margin:0.593, bscore:1106},
  {pid: 41, oid: 7518, lon:118.99871884, lat:29.89523592, wc:2, margin:0.627, bscore:1321},
  {pid: 42, oid: 7736, lon:119.3047737, lat:29.94607859, wc:2, margin:0.540, bscore:1009},
  {pid: 43, oid: 7934, lon:119.88149153, lat:29.98452496, wc:2, margin:0.680, bscore:1049},
  {pid: 44, oid: 8078, lon:119.41580531, lat:30.01165413, wc:2, margin:0.520, bscore:1753},
  {pid: 45, oid: 8092, lon:119.74512776, lat:30.01479779, wc:2, margin:0.573, bscore:1100},
  {pid: 46, oid: 8228, lon:119.34466036, lat:30.04049113, wc:2, margin:0.633, bscore:1172},
  {pid: 47, oid: 8262, lon:119.15439497, lat:30.04830794, wc:2, margin:0.380, bscore:1114},
  {pid: 48, oid: 8581, lon:119.42056763, lat:30.12098237, wc:2, margin:0.433, bscore:1226},
  {pid: 49, oid: 8591, lon:119.6497253, lat:30.12313611, wc:2, margin:0.447, bscore:1406},
  {pid: 50, oid: 8627, lon:119.82435707, lat:30.13113128, wc:2, margin:0.587, bscore:1021},
  {pid: 51, oid: 8675, lon:119.97338915, lat:30.14181974, wc:2, margin:0.533, bscore:1334},
  {pid: 52, oid: 8967, lon:119.5836102, lat:30.19895421, wc:2, margin:0.420, bscore:1193},
  {pid: 53, oid: 9062, lon:119.62259613, lat:30.22042478, wc:2, margin:0.527, bscore:1095},
  {pid: 54, oid: 9226, lon:120.5575626, lat:30.25797487, wc:2, margin:0.470, bscore:1391},
  {pid: 55, oid: 9329, lon:119.74216246, lat:30.28088127, wc:2, margin:0.677, bscore:1045},
  {pid: 56, oid: 9361, lon:119.73138481, lat:30.2890548, wc:2, margin:0.613, bscore:1047},
  {pid: 57, oid: 9613, lon:119.30261995, lat:30.35031839, wc:2, margin:0.467, bscore:1071},
  {pid: 58, oid: 9633, lon:119.87502584, lat:30.35633818, wc:2, margin:0.503, bscore:1132},
  {pid: 59, oid: 9650, lon:119.91410094, lat:30.36065013, wc:2, margin:0.423, bscore:1052},
  {pid: 60, oid: 9694, lon:119.76165765, lat:30.37313561, wc:2, margin:0.423, bscore:1313},
  {pid: 61, oid: 9702, lon:120.16805566, lat:30.37520463, wc:2, margin:0.457, bscore:1215},
  {pid: 62, oid: 9791, lon:119.82301042, lat:30.40385882, wc:2, margin:0.513, bscore:1090},
  {pid: 63, oid: 9796, lon:120.17937286, lat:30.40610621, wc:2, margin:0.593, bscore:1081},
  {pid: 64, oid: 9910, lon:120.08963345, lat:30.45254774, wc:2, margin:0.413, bscore:1137},
  {pid: 65, oid: 9946, lon:120.12493615, lat:30.47608288, wc:2, margin:0.480, bscore:1504},
  {pid: 66, oid:10026, lon:119.35418054, lat:29.30405276, wc:3, margin:0.470, bscore:1117},
  {pid: 67, oid:10077, lon:119.51713393, lat:29.3812175, wc:3, margin:0.410, bscore:1101},
  {pid: 68, oid:10151, lon:118.63732653, lat:29.41643102, wc:3, margin:0.383, bscore:1036},
  {pid: 69, oid:10173, lon:118.67756992, lat:29.42020342, wc:3, margin:0.673, bscore:1273},
  {pid: 70, oid:10220, lon:118.73748686, lat:29.42748067, wc:3, margin:0.547, bscore:1510},
  {pid: 71, oid:10357, lon:118.69823339, lat:29.46215464, wc:3, margin:0.473, bscore:1031},
  {pid: 72, oid:10379, lon:118.761022, lat:29.46494158, wc:3, margin:0.633, bscore:1131},
  {pid: 73, oid:10416, lon:118.68915912, lat:29.46835279, wc:3, margin:0.457, bscore:1081},
  {pid: 74, oid:10418, lon:118.78851235, lat:29.46835279, wc:3, margin:0.703, bscore:1178},
  {pid: 75, oid:10559, lon:118.83055276, lat:29.48093192, wc:3, margin:0.640, bscore:1058},
  {pid: 76, oid:10599, lon:118.88013796, lat:29.48470432, wc:3, margin:0.497, bscore:1212},
  {pid: 77, oid:10605, lon:119.21206899, lat:29.48488268, wc:3, margin:0.520, bscore:1445},
  {pid: 78, oid:11079, lon:119.04138798, lat:29.51030402, wc:3, margin:0.677, bscore:1159},
  {pid: 79, oid:11620, lon:119.58226356, lat:29.53554699, wc:3, margin:0.420, bscore:1246},
  {pid: 80, oid:11702, lon:119.15224123, lat:29.53806192, wc:3, margin:0.433, bscore:1433},
  {pid: 81, oid:11801, lon:118.8761872, lat:29.54102722, wc:3, margin:0.627, bscore:1190},
  {pid: 82, oid:11873, lon:119.15861774, lat:29.54462126, wc:3, margin:0.463, bscore:2036},
  {pid: 83, oid:12086, lon:119.06384401, lat:29.55522054, wc:3, margin:0.653, bscore:1201},
  {pid: 84, oid:12221, lon:118.98883747, lat:29.56303735, wc:3, margin:0.393, bscore:1546},
  {pid: 85, oid:12510, lon:118.91733133, lat:29.58854786, wc:3, margin:0.560, bscore:1617},
  {pid: 86, oid:12573, lon:119.12861245, lat:29.59681504, wc:3, margin:0.460, bscore:1376},
  {pid: 87, oid:12634, lon:119.11989937, lat:29.60238445, wc:3, margin:0.570, bscore:1105},
  {pid: 88, oid:12864, lon:119.00518453, lat:29.62771661, wc:3, margin:0.637, bscore:1245},
  {pid: 89, oid:12936, lon:119.00150132, lat:29.63696924, wc:3, margin:0.570, bscore:1319},
  {pid: 90, oid:12986, lon:118.92442575, lat:29.64244948, wc:3, margin:0.403, bscore:1036},
  {pid: 91, oid:13138, lon:119.02351144, lat:29.66517751, wc:3, margin:0.430, bscore:1274},
  {pid: 92, oid:13403, lon:119.65448762, lat:29.72185715, wc:3, margin:0.563, bscore:1028},
  {pid: 93, oid:13483, lon:119.63741818, lat:29.74269899, wc:3, margin:0.397, bscore:1524},
  {pid: 94, oid:13515, lon:119.64631408, lat:29.76003597, wc:3, margin:0.380, bscore:1175},
  {pid: 95, oid:13644, lon:118.84052331, lat:29.84969066, wc:3, margin:0.710, bscore:1153},
  {pid: 96, oid:13689, lon:119.56878815, lat:29.87996349, wc:3, margin:0.653, bscore:1425},
  {pid: 97, oid:13800, lon:119.43682552, lat:29.93988044, wc:3, margin:0.443, bscore:1343},
  {pid: 98, oid:13801, lon:119.46449424, lat:29.94059835, wc:3, margin:0.557, bscore:1262},
  {pid: 99, oid:14265, lon:120.21171472, lat:30.12107155, wc:3, margin:0.640, bscore:1463},
  {pid:100, oid:14525, lon:119.68763657, lat:30.23542519, wc:3, margin:0.467, bscore:1425},
  {pid:101, oid:14544, lon:120.2268935, lat:30.23973714, wc:3, margin:0.490, bscore:1049},
  {pid:102, oid:14549, lon:119.71206799, lat:30.24081625, wc:3, margin:0.460, bscore:1011},
  {pid:103, oid:14630, lon:119.99971122, lat:30.25554912, wc:3, margin:0.490, bscore:1247},
  {pid:104, oid:14714, lon:120.16823403, lat:30.28662905, wc:3, margin:0.667, bscore:1072}
];

var CLASS_NAME = {1: 'built', 2: 'non-built', 3: 'water'};

// scene codes -> aligned with the pooled scene groups (manuscript 3.2d) so
// Hangzhou can optionally enter the scene stratification. Same short codes as
// Wuhan / Changsha / Nanchang.
var SCENE_CLASS = {
  be: 1, br: 1,                    // building edge / settlement fabric
  fe: 2, veg: 2, bl: 2, re: 2, ct: 2,  // forest edge / green / bare / road / cropland
  we: 3, wr: 3, pf: 3, tw: 3,      // shoreline / open water / paddy-as-water / turbid pond
  mx: 0, ot: 0                     // mixed / other -> class from the buttons
};
var SCENE_HINT =
  'be building edge  br settlement fabric | fe forest/veg edge  veg green/grass  ' +
  'bl bare/construction  re road  ct cropland/paddy | we shoreline  wr open water  ' +
  'pf paddy-as-water  tw turbid pond | mx mixed  ot other';

var fc = ee.FeatureCollection(PTS.map(function (p, k) {
  return ee.Feature(ee.Geometry.Point([p.lon, p.lat]), {k: k + 1, pid: p.pid, oid: p.oid});
}));

// ----------------------------------------------------------------------------
// Imagery -- exactly the BAMS / labelling recipe
// ----------------------------------------------------------------------------
function s2composite(geom) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom)
    .filterDate(YEAR + '-01-01', YEAR + '-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .median();
}
var s2 = s2composite(fc.geometry());
var s2True  = {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000};
var s2False = {bands: ['B8', 'B4', 'B3'], min: 0, max: 4000};   // veg red, water black
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');

Map.setOptions('HYBRID');                       // ancillary context only
Map.addLayer(s2, s2True,  'S2 ' + YEAR + ' true colour (PRIMARY)', true);
Map.addLayer(s2, s2False, 'S2 ' + YEAR + ' false colour NIR (veg red, water black)', false);
Map.addLayer(wc, {min: 10, max: 100}, 'ESA WorldCover v200 (context only)', false, 0.6);
Map.addLayer(fc.style({color: 'ff0', fillColor: '00000000', pointSize: 8, width: 2}),
             {}, 'all 104', false);

// ----------------------------------------------------------------------------
// Stepper UI
// ----------------------------------------------------------------------------
var idx = 0;
var log = [];
var cur = {cls: null, conf: null, scene: null};

var title   = ui.Label('', {fontWeight: 'bold', fontSize: '15px'});
var meta    = ui.Label('', {whiteSpace: 'pre', fontSize: '13px'});
var counter = ui.Label('', {fontSize: '12px', color: '#666'});
var chosen  = ui.Label('(not set)', {fontSize: '13px', fontWeight: 'bold', color: '#0a0'});
var marker  = ui.Map.Layer(ee.Geometry.Point([0, 0]), {color: 'red'}, 'current');
var boxLyr  = ui.Map.Layer(ee.Geometry.Point([0, 0]), {color: 'red'}, '10 m pixel');
var noteBox = ui.Textbox({placeholder: 'note (what you see / why)', style: {width: '280px'}});

function commit() {
  var p = PTS[idx];
  var tag = 'class=' + (cur.cls || '_') + '  conf=' + (cur.conf || '_') +
            '  scene=' + (cur.scene || '_');
  if (cur.cls === null || cur.conf === null) {
    chosen.setValue(tag + '   (need class + confidence)');
    return;
  }
  var warn = (cur.scene && SCENE_CLASS[cur.scene] && SCENE_CLASS[cur.scene] !== cur.cls)
             ? '  [!] scene/class mismatch' : '';
  log[idx] = {k: idx + 1, pid: p.pid, oid: p.oid, wc: p.wc,
              cls: cur.cls, conf: cur.conf, scene: cur.scene || '',
              note: noteBox.getValue()};
  chosen.setValue('recorded: ' + tag + warn);
  counter.setValue((idx + 1) + ' / ' + PTS.length + '   |   recorded: ' +
                   log.filter(function (x) { return x; }).length);
}

var clsPanel = ui.Panel([1, 2, 3].map(function (c) {
  return ui.Button(c + ' ' + CLASS_NAME[c], function () { cur.cls = c; commit(); });
}), ui.Panel.Layout.flow('horizontal'));

var confPanel = ui.Panel(['h', 'm', 'l'].map(function (c) {
  return ui.Button(c, function () { cur.conf = c; commit(); });
}), ui.Panel.Layout.flow('horizontal'));

function sceneRow(codes) {
  return ui.Panel(codes.map(function (s) {
    return ui.Button(s, function () { cur.scene = s; commit(); });
  }), ui.Panel.Layout.flow('horizontal'));
}

function show(i) {
  idx = ((i % PTS.length) + PTS.length) % PTS.length;
  var p = PTS[idx];
  cur = {cls: null, conf: null, scene: null};
  title.setValue('#' + (idx + 1) + '   PointID=' + p.pid + '   (orig ' + p.oid + ')');
  meta.setValue(
    'WC context : ' + p.wc + ' (' + CLASS_NAME[p.wc] + ')   <- do NOT anchor on this\n' +
    'margin     : ' + p.margin.toFixed(3) + '    BoundaryScore : ' + p.bscore + '\n' +
    'lon,lat    : ' + p.lon.toFixed(6) + ', ' + p.lat.toFixed(6));
  counter.setValue((idx + 1) + ' / ' + PTS.length + '   |   recorded: ' +
                   log.filter(function (x) { return x; }).length);
  var g = ee.Geometry.Point([p.lon, p.lat]);
  marker.setEeObject(g);
  marker.setVisParams({color: 'red'});
  boxLyr.setEeObject(g.buffer(5).bounds());     // ~10 m pixel footprint
  boxLyr.setVisParams({color: 'red'});
  Map.centerObject(g, ZOOM);
  var pre = log[idx];
  if (pre) {
    cur = {cls: pre.cls, conf: pre.conf, scene: pre.scene || null};
    chosen.setValue('recorded: class=' + pre.cls + '  conf=' + pre.conf +
                    '  scene=' + (pre.scene || '_'));
    noteBox.setValue(pre.note, false);
  } else {
    chosen.setValue('(not set)');
    noteBox.setValue('', false);
  }
}

function dump() {
  var done = log.filter(function (x) { return x; });
  print('==== Hangzhou 104 new two-stage points (' + done.length + ' / ' +
        PTS.length + ' labelled) ==== copy the block below ====');
  print('PointID,Original_ID,Human_Class,Scene,Confidence,Notes');
  done.forEach(function (r) {
    var notes = 'WC_Class=' + r.wc + (r.note ? '; ' + r.note.replace(/[",]/g, ' ') : '');
    print([r.pid, r.oid, r.cls, r.scene, r.conf, notes].join(','));
  });
  var byc = {1: 0, 2: 0, 3: 0};
  var flips = 0;
  done.forEach(function (r) { byc[r.cls]++; if (r.cls !== r.wc) flips++; });
  print('class counts: built=' + byc[1] + ' non-built=' + byc[2] + ' water=' + byc[3]);
  print('differ from WC: ' + flips + ' / ' + done.length +
        '   (still to label: ' + (PTS.length - done.length) + ')');
  var miss = [];
  for (var j = 0; j < PTS.length; j++) { if (!log[j]) miss.push(PTS[j].pid); }
  if (miss.length) print('missing PointIDs: ' + miss.join(', '));
}

Map.layers().add(boxLyr);
Map.layers().add(marker);

var panel = ui.Panel({style: {position: 'top-left', width: '320px', padding: '8px'}});
panel.add(ui.Label('Hangzhou -> two-stage: label 104 new points', {fontWeight: 'bold'}));
panel.add(counter);
panel.add(title);
panel.add(meta);
panel.add(ui.Panel(
  [ui.Button('◀ Prev', function () { show(idx - 1); }),
   ui.Button('Next ▶', function () { show(idx + 1); })],
  ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Label('1. class', {fontSize: '12px', margin: '6px 0 0 8px'}));
panel.add(clsPanel);
panel.add(ui.Label('2. confidence', {fontSize: '12px', margin: '6px 0 0 8px'}));
panel.add(confPanel);
panel.add(ui.Label('3. scene code', {fontSize: '12px', margin: '6px 0 0 8px'}));
panel.add(sceneRow(['be', 'br']));
panel.add(sceneRow(['fe', 'veg', 'bl', 're', 'ct']));
panel.add(sceneRow(['we', 'wr', 'pf', 'tw']));
panel.add(sceneRow(['mx', 'ot']));
panel.add(chosen);
panel.add(noteBox);
panel.add(ui.Button('Dump log', dump));
panel.add(ui.Label(SCENE_HINT, {fontSize: '11px', color: '#888'}));
panel.add(ui.Label(
  'Primary judge = S2 true-colour (0-3000). Toggle false-colour / WorldCover ' +
  'in the layer list. HYBRID basemap is ancillary context only. Records once ' +
  'class + confidence are set.',
  {fontSize: '11px', color: '#888'}));
ui.root.insert(0, panel);

show(0);
