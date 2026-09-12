/**** Hangzhou vegetation-edge points -> urban green space vs forest edge ********
 *
 * Paste into the GEE Code Editor (code.earthengine.google.com).
 *
 * WHY: Hangzhou's scene notes (original + the 104-point re-label) use one code
 * `veg` for all vegetation edges and do NOT separate "urban green space" (the
 * other five cities' `ub` code). To fold Hangzhou into the six-city physical-
 * scene table (Section 4.4 / Table T4) we split its 43 vegetation-edge points
 * (Scene in {veg, fe}, all expert-class = non-built) into:
 *
 *   ug  URBAN GREEN SPACE  -- parks, street trees, campus / residential /
 *       roadside greenery embedded in the built matrix; grass strips, sports
 *       fields, cemeteries inside the urban footprint
 *   fe  VEGETATION / FOREST EDGE -- forest / tree-canopy edge, hillside scrub,
 *       shelterbelts, rural/natural vegetation not inside a built matrix
 *   ot  OTHER -- on inspection the pixel is really bare / water / built /
 *       cropland, not a vegetation edge at all (record what it is in the note)
 *
 * Judge from context: is the vegetation *inside / against* an urban fabric
 * (roads, buildings, lots on 2-3 sides) -> ug; or is it a natural / rural
 * vegetation boundary -> fe.
 *
 * Imagery: S2_SR_HARMONIZED 2021, CLOUDY_PIXEL_PERCENTAGE < 20, median (the
 * BAMS / labelling recipe). True colour B4/B3/B2 stretch 0-3000; false colour
 * B8/B4/B3 stretch 0-4000. ESA WorldCover v200 overlay for context only.
 * Google HYBRID basemap = ancillary spatial context. Red box = the 10 m pixel.
 *
 * WORKFLOW: Prev/Next to step; click ug / fe / ot; optional note; it records
 * automatically. "Dump log" -> copy the console CSV block into
 *   data/analysis_outputs/hangzhou_standardise/veg_subclass.csv
 * then re-run the scene-table build (I'll wire that up).
 ******************************************************************************/

var YEAR = 2021;
var ZOOM = 18;   // ~1:2000; wheel-zoom further as needed

// 43 Hangzhou vegetation-edge points (Scene in {veg, fe}, expert = non-built)
//   hum = expert class (all 2 = non-built here); wc = ESA WorldCover class
//   (1 built / 2 non-built / 3 water) at that pixel -- context only.
var PTS = [
  {pid: 1, oid:    4, lon:119.07372538, lat:29.2498836, hum:2, wc:1, scene:'veg'},
  {pid: 2, oid:   61, lon:119.03698684, lat:29.32175093, hum:2, wc:1, scene:'fe'},
  {pid: 3, oid:  442, lon:119.53509964, lat:29.62942444, hum:2, wc:1, scene:'veg'},
  {pid: 4, oid:  467, lon:119.21179698, lat:29.65089056, hum:2, wc:1, scene:'veg'},
  {pid: 5, oid:  489, lon:119.10903254, lat:29.68466825, hum:2, wc:1, scene:'veg'},
  {pid: 6, oid:  498, lon:119.12529042, lat:29.70011903, hum:2, wc:1, scene:'veg'},
  {pid: 7, oid:  530, lon:119.18556855, lat:29.74368891, hum:2, wc:1, scene:'veg'},
  {pid: 8, oid: 1658, lon:120.25025027, lat:30.120532, hum:2, wc:1, scene:'veg'},
  {pid: 9, oid: 1874, lon:120.24971072, lat:30.1550276, hum:2, wc:1, scene:'veg'},
  {pid:10, oid: 1978, lon:119.21071788, lat:30.16706272, hum:2, wc:1, scene:'veg'},
  {pid:11, oid: 2003, lon:120.13158467, lat:30.16877055, hum:2, wc:1, scene:'veg'},
  {pid:12, oid: 2071, lon:119.75788525, lat:30.17559744, hum:2, wc:1, scene:'veg'},
  {pid:13, oid: 2088, lon:120.30882056, lat:30.17613699, hum:2, wc:1, scene:'veg'},
  {pid:14, oid: 2333, lon:119.67981977, lat:30.1930236, hum:2, wc:1, scene:'veg'},
  {pid:15, oid: 2972, lon:120.19284826, lat:30.24629648, hum:2, wc:1, scene:'veg'},
  {pid:16, oid: 4069, lon:120.29947875, lat:30.34654599, hum:2, wc:1, scene:'veg'},
  {pid:17, oid: 4961, lon:120.29920674, lat:30.51390497, hum:2, wc:1, scene:'veg'},
  {pid:18, oid: 6000, lon:119.19158388, lat:29.54462126, hum:2, wc:2, scene:'veg'},
  {pid:19, oid: 6800, lon:119.6796414, lat:29.744496, hum:2, wc:2, scene:'veg'},
  {pid:20, oid: 6929, lon:119.70784968, lat:29.77656586, hum:2, wc:2, scene:'veg'},
  {pid:21, oid: 7416, lon:119.5224358, lat:29.87412207, hum:2, wc:2, scene:'veg'},
  {pid:22, oid: 7518, lon:118.99871884, lat:29.89523592, hum:2, wc:2, scene:'veg'},
  {pid:23, oid: 7736, lon:119.3047737, lat:29.94607859, hum:2, wc:2, scene:'fe'},
  {pid:24, oid: 8078, lon:119.41580531, lat:30.01165413, hum:2, wc:2, scene:'veg'},
  {pid:25, oid: 8368, lon:119.43655797, lat:30.07579384, hum:2, wc:2, scene:'veg'},
  {pid:26, oid: 8581, lon:119.42056763, lat:30.12098237, hum:2, wc:2, scene:'fe'},
  {pid:27, oid: 8591, lon:119.6497253, lat:30.12313611, hum:2, wc:2, scene:'fe'},
  {pid:28, oid: 8627, lon:119.82435707, lat:30.13113128, hum:2, wc:2, scene:'veg'},
  {pid:29, oid: 8717, lon:119.76309348, lat:30.14927536, hum:2, wc:2, scene:'veg'},
  {pid:30, oid: 8967, lon:119.5836102, lat:30.19895421, hum:2, wc:2, scene:'fe'},
  {pid:31, oid: 8998, lon:120.2298588, lat:30.20452362, hum:2, wc:2, scene:'veg'},
  {pid:32, oid: 9189, lon:119.86783776, lat:30.2480935, hum:2, wc:2, scene:'veg'},
  {pid:33, oid: 9329, lon:119.74216246, lat:30.28088127, hum:2, wc:2, scene:'veg'},
  {pid:34, oid: 9484, lon:119.6338286, lat:30.31780263, hum:2, wc:2, scene:'veg'},
  {pid:35, oid: 9694, lon:119.76165765, lat:30.37313561, hum:2, wc:2, scene:'veg'},
  {pid:36, oid: 9796, lon:120.17937286, lat:30.40610621, hum:2, wc:2, scene:'veg'},
  {pid:37, oid: 9877, lon:120.29121157, lat:30.43674023, hum:2, wc:2, scene:'veg'},
  {pid:38, oid:10077, lon:119.51713393, lat:29.3812175, hum:2, wc:3, scene:'fe'},
  {pid:39, oid:10220, lon:118.73748686, lat:29.42748067, hum:2, wc:3, scene:'fe'},
  {pid:40, oid:10416, lon:118.68915912, lat:29.46835279, hum:2, wc:3, scene:'veg'},
  {pid:41, oid:11079, lon:119.04138798, lat:29.51030402, hum:2, wc:3, scene:'veg'},
  {pid:42, oid:11620, lon:119.58226356, lat:29.53554699, hum:2, wc:3, scene:'veg'},
  {pid:43, oid:14714, lon:120.16823403, lat:30.28662905, hum:2, wc:3, scene:'veg'}
];

var WC_NAME = {1: 'built', 2: 'non-built', 3: 'water'};
var CHOICES = ['ug', 'fe', 'ot'];
var CHOICE_LABEL = {
  ug: 'ug  urban green space',
  fe: 'fe  vegetation / forest edge',
  ot: 'ot  other (not a veg edge)'
};

var fc = ee.FeatureCollection(PTS.map(function (p, k) {
  return ee.Feature(ee.Geometry.Point([p.lon, p.lat]), {k: k + 1, pid: p.pid, oid: p.oid});
}));

// ---------------------------------------------------------------------------
// Imagery -- the BAMS / labelling recipe
// ---------------------------------------------------------------------------
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

Map.setOptions('HYBRID');
Map.addLayer(s2, s2True,  'S2 ' + YEAR + ' true colour (PRIMARY)', true);
Map.addLayer(s2, s2False, 'S2 ' + YEAR + ' false colour NIR (veg red, water black)', false);
Map.addLayer(wc, {min: 10, max: 100}, 'ESA WorldCover v200 (context only)', false, 0.6);
Map.addLayer(fc.style({color: 'ff0', fillColor: '00000000', pointSize: 8, width: 2}),
             {}, 'all 43', false);

// ---------------------------------------------------------------------------
// Stepper UI
// ---------------------------------------------------------------------------
var idx = 0;
var log = [];
var cur = {sub: null};

var title   = ui.Label('', {fontWeight: 'bold', fontSize: '15px'});
var meta    = ui.Label('', {whiteSpace: 'pre', fontSize: '13px'});
var counter = ui.Label('', {fontSize: '12px', color: '#666'});
var chosen  = ui.Label('(not set)', {fontSize: '13px', fontWeight: 'bold', color: '#0a0'});
var marker  = ui.Map.Layer(ee.Geometry.Point([0, 0]), {color: 'red'}, 'current');
var boxLyr  = ui.Map.Layer(ee.Geometry.Point([0, 0]), {color: 'red'}, '10 m pixel');
var noteBox = ui.Textbox({placeholder: 'note (what you see / why)', style: {width: '280px'}});

function commit() {
  if (cur.sub === null) { chosen.setValue('(pick ug / fe / ot)'); return; }
  var p = PTS[idx];
  log[idx] = {k: idx + 1, pid: p.pid, oid: p.oid, wc: p.wc, scene0: p.scene,
              sub: cur.sub, note: noteBox.getValue()};
  chosen.setValue('recorded: ' + CHOICE_LABEL[cur.sub]);
  counter.setValue((idx + 1) + ' / ' + PTS.length + '   |   recorded: ' +
                   log.filter(function (x) { return x; }).length);
}

var choicePanel = ui.Panel(CHOICES.map(function (c) {
  return ui.Button(c, function () { cur.sub = c; commit(); });
}), ui.Panel.Layout.flow('horizontal'));

function show(i) {
  idx = ((i % PTS.length) + PTS.length) % PTS.length;
  var p = PTS[idx];
  cur = {sub: null};
  title.setValue('#' + (idx + 1) + '   PointID=' + p.pid + '   (orig ' + p.oid + ')');
  meta.setValue(
    'current scene code : ' + p.scene + '   (expert class = non-built)\n' +
    'WC context         : ' + p.wc + ' (' + WC_NAME[p.wc] + ')\n' +
    'lon,lat            : ' + p.lon.toFixed(6) + ', ' + p.lat.toFixed(6));
  counter.setValue((idx + 1) + ' / ' + PTS.length + '   |   recorded: ' +
                   log.filter(function (x) { return x; }).length);
  var g = ee.Geometry.Point([p.lon, p.lat]);
  marker.setEeObject(g); marker.setVisParams({color: 'red'});
  boxLyr.setEeObject(g.buffer(5).bounds()); boxLyr.setVisParams({color: 'red'});
  Map.centerObject(g, ZOOM);
  var pre = log[idx];
  if (pre) {
    cur = {sub: pre.sub};
    chosen.setValue('recorded: ' + CHOICE_LABEL[pre.sub]);
    noteBox.setValue(pre.note, false);
  } else {
    chosen.setValue('(not set)');
    noteBox.setValue('', false);
  }
}

function dump() {
  var done = log.filter(function (x) { return x; });
  print('==== Hangzhou veg-edge sub-classification (' + done.length + ' / ' +
        PTS.length + ') ==== copy the block below ====');
  print('PointID,Original_ID,WC_Class,scene_old,subclass,note');
  done.forEach(function (r) {
    print([r.pid, r.oid, r.wc, r.scene0, r.sub,
           '"' + (r.note || '').replace(/[",]/g, ' ') + '"'].join(','));
  });
  var c = {ug: 0, fe: 0, ot: 0};
  done.forEach(function (r) { c[r.sub]++; });
  print('subclass counts: ug=' + c.ug + ' fe=' + c.fe + ' ot=' + c.ot);
  var miss = [];
  for (var j = 0; j < PTS.length; j++) { if (!log[j]) miss.push(PTS[j].pid); }
  if (miss.length) print('missing PointIDs: ' + miss.join(', '));
}

Map.layers().add(boxLyr);
Map.layers().add(marker);

var panel = ui.Panel({style: {position: 'top-left', width: '320px', padding: '8px'}});
panel.add(ui.Label('Hangzhou veg edge: ug vs fe (43 pts)', {fontWeight: 'bold'}));
panel.add(counter);
panel.add(title);
panel.add(meta);
panel.add(ui.Panel(
  [ui.Button('◀ Prev', function () { show(idx - 1); }),
   ui.Button('Next ▶', function () { show(idx + 1); })],
  ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Label('sub-class', {fontSize: '12px', margin: '6px 0 0 8px'}));
panel.add(choicePanel);
panel.add(chosen);
panel.add(noteBox);
panel.add(ui.Button('Dump log', dump));
panel.add(ui.Label(
  'ug = greenery inside the built matrix (park / street trees / campus / ' +
  'residential / roadside / sports field). fe = forest edge, hillside scrub, ' +
  'shelterbelt, rural/natural veg. ot = on inspection not a veg edge.',
  {fontSize: '11px', color: '#888'}));
panel.add(ui.Label(
  'Primary judge = S2 true colour (0-3000); toggle false colour / WorldCover ' +
  'in the layer list; HYBRID basemap is ancillary. Records on click.',
  {fontSize: '11px', color: '#888'}));
ui.root.insert(0, panel);

show(0);
