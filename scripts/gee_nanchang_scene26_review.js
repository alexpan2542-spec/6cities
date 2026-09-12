/**** Nanchang 26 blank-scene points -> full re-annotation (class + scene) ******
 *
 * Paste into the GEE Code Editor (code.earthengine.google.com).
 *
 * Why 26: in data/cities/Nanchang/04_manual/Nanchang_BAMS150_Manual.csv exactly
 * 26 rows have an empty Scene field AND Human_Class == WC.  These were never
 * truly annotated -- the WC value was placeholder-copied into Human_Class and the
 * scene note was skipped.  This tool re-annotates all 26 from imagery so the
 * Nanchang anchor set is consistent.  13 were already done in round 1
 * (gee_nanchang_scene13_review.js); they are marked [done r1] here for a second
 * pass / reconciliation.  The other 13 (all currently Human_Class = 2, copied
 * from WC = 2) have never been checked.
 *
 * For each point decide, from the sub-metre HYBRID basemap:
 *   1. the CLASS   : 1 built | 2 non-built | 3 water   (this is the label that
 *                    feeds T1/T2/T3/T5 -- the WC value shown is only the bad
 *                    placeholder currently in the file, do NOT anchor on it)
 *   2. the SCENE   : the fine boundary-scene code
 *        built     : be  single building roof / footprint edge
 *                    br  settlement fabric edge (houses + roads + lots, not one building)
 *        non-built : fe  forest / tree-canopy edge
 *                    veg low vegetation / grass / urban green
 *                    bl  bare land / construction ground
 *                    re  road / road edge
 *                    ct  cropland / paddy transition (dry field margin)
 *        water     : we  shoreline (land/water boundary)
 *                    wr  open water body interior
 *                    pf  paddy / cropland mistaken for water (bunds, rect. plots)
 *                    tw  turbid / green / sediment-laden pond
 *
 * Imagery : S2_SR_HARMONIZED 2021, CLOUDY_PIXEL_PERCENTAGE < 20, median
 *           (same recipe as the BAMS pipeline).  ESA/WorldCover/v200 overlay.
 * Basemap : Google "HYBRID" sub-metre -> primary judge.
 *
 * Workflow: Prev/Next to step, click a CLASS button then a SCENE button, add an
 * optional note, it records automatically once both are set.  At the end press
 * "Dump log" and copy the console CSV block into the review sheet.
 ******************************************************************************/

var YEAR = 2021;
var ZOOM = 18;

// ----------------------------------------------------------------------------
// 26 points.  wc = the placeholder value currently sitting in Human_Class.
// done = re-annotated in round 1.
// ----------------------------------------------------------------------------
var PTS = [
  {pid: 100, oid: 14010, wc: 3, lat: 28.29949164832467, lon: 116.29369580966716, done: true},
  {pid: 106, oid:  4508, wc: 1, lat: 28.53745536708793, lon: 115.82369725301582, done: true},
  {pid: 107, oid:  6333, wc: 2, lat: 29.00296234731866, lon: 115.96437342650894, done: false},
  {pid: 108, oid:  7170, wc: 2, lat: 28.53251463302527, lon: 115.81228864890753, done: false},
  {pid: 109, oid: 10890, wc: 3, lat: 28.67058569219444, lon: 116.42898209145557, done: true},
  {pid: 110, oid: 10319, wc: 3, lat: 28.59898996405012, lon: 116.08277138095588, done: true},
  {pid: 116, oid:  6847, wc: 2, lat: 28.20175494541246, lon: 116.22901710921056, done: false},
  {pid: 117, oid:  6178, wc: 2, lat: 28.48957516244436, lon: 116.02114695246530, done: false},
  {pid: 118, oid:  6430, wc: 2, lat: 28.58758135994180, lon: 115.92170345051326, done: false},
  {pid: 119, oid:  4063, wc: 1, lat: 28.63662937445472, lon: 115.80321566453790, done: true},
  {pid: 120, oid:  1708, wc: 1, lat: 28.56341667879898, lon: 115.79629863685018, done: true},
  {pid: 126, oid: 10584, wc: 3, lat: 28.36731445227569, lon: 115.67736169323275, done: true},
  {pid: 127, oid:  2157, wc: 1, lat: 28.85105723277405, lon: 115.61169484596360, done: true},
  {pid: 128, oid: 13035, wc: 3, lat: 28.78035981991385, lon: 115.91442709671188, done: true},
  {pid: 129, oid:  6127, wc: 2, lat: 28.33767004789975, lon: 116.03812511133516, done: false},
  {pid: 130, oid:  1041, wc: 1, lat: 28.74325939867971, lon: 115.92574586929180, done: true},
  {pid: 136, oid: 14949, wc: 3, lat: 28.81117203415915, lon: 116.16200278901523, done: true},
  {pid: 137, oid:  3553, wc: 1, lat: 28.51266186524623, lon: 116.39610375205680, done: true},
  {pid: 138, oid:  8184, wc: 2, lat: 28.64237859227309, lon: 115.76431861273552, done: false},
  {pid: 139, oid:  7364, wc: 2, lat: 28.25250975896522, lon: 116.32621482295228, done: false},
  {pid: 140, oid:  9545, wc: 2, lat: 28.86201667924031, lon: 115.96033100773040, done: false},
  {pid: 146, oid:  7599, wc: 2, lat: 28.67238232276268, lon: 116.11349376367278, done: false},
  {pid: 147, oid:  8946, wc: 2, lat: 28.62324447672134, lon: 115.74132174146206, done: false},
  {pid: 148, oid:  5656, wc: 2, lat: 28.65845843585883, lon: 115.83950760201633, done: false},
  {pid: 149, oid: 13160, wc: 3, lat: 28.86471162509267, lon: 115.98763979236764, done: true},
  {pid: 150, oid:  7619, wc: 2, lat: 28.65872793044407, lon: 116.21653052676128, done: false}
];

var CLASS_NAME = {1: 'built', 2: 'non-built', 3: 'water'};
var SCENE_CLASS = {                       // which class each scene code implies
  be: 1, br: 1,
  fe: 2, veg: 2, bl: 2, re: 2, ct: 2,
  we: 3, wr: 3, pf: 3, tw: 3
};

var fc = ee.FeatureCollection(PTS.map(function (p, k) {
  return ee.Feature(ee.Geometry.Point([p.lon, p.lat]), {k: k + 1, pid: p.pid, oid: p.oid});
}));

// ----------------------------------------------------------------------------
// Imagery
// ----------------------------------------------------------------------------
function s2composite(geom) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom)
    .filterDate(YEAR + '-01-01', YEAR + '-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .median();
}
var s2 = s2composite(fc.geometry());
var s2True  = {bands: ['B4', 'B3', 'B2'], min: 200, max: 2500};
var s2False = {bands: ['B8', 'B4', 'B3'], min: 200, max: 4000};   // veg red, water black
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');

Map.setOptions('HYBRID');
Map.addLayer(s2, s2True,  'S2 ' + YEAR + ' true colour', false);
Map.addLayer(s2, s2False, 'S2 ' + YEAR + ' false colour (NIR): veg red, water black', false);
Map.addLayer(wc, {}, 'ESA WorldCover v200', false, 0.5);
Map.addLayer(fc.style({color: 'ff0', fillColor: '00000000', pointSize: 10, width: 2}), {}, 'all 26');

// ----------------------------------------------------------------------------
// Stepper UI
// ----------------------------------------------------------------------------
var idx = 0;
var log = [];
var cur = {cls: null, scene: null};

var title   = ui.Label('', {fontWeight: 'bold', fontSize: '15px'});
var meta    = ui.Label('', {whiteSpace: 'pre', fontSize: '13px'});
var counter = ui.Label('', {fontSize: '12px', color: '#666'});
var chosen  = ui.Label('', {fontSize: '13px', fontWeight: 'bold', color: '#0a0'});
var marker  = ui.Map.Layer(ee.Geometry.Point([0, 0]), {color: 'red'}, 'current');
var noteBox = ui.Textbox({placeholder: 'note (what you see / why)', style: {width: '270px'}});

function commit() {
  if (cur.cls === null || cur.scene === null) {
    chosen.setValue('class=' + (cur.cls || '_') + '  scene=' + (cur.scene || '_') +
                    '  (pick both)');
    return;
  }
  var p = PTS[idx];
  var warn = (SCENE_CLASS[cur.scene] !== cur.cls) ? '  [!] scene/class mismatch' : '';
  log[idx] = {k: idx + 1, pid: p.pid, oid: p.oid, wc: p.wc,
              cls: cur.cls, scene: cur.scene, note: noteBox.getValue()};
  chosen.setValue('recorded: class=' + cur.cls + ' (' + CLASS_NAME[cur.cls] + ')  scene=' +
                  cur.scene + warn);
  counter.setValue((idx + 1) + ' / ' + PTS.length + '   |   recorded: ' +
                   log.filter(function (x) { return x; }).length);
}

var clsPanel = ui.Panel(
  [1, 2, 3].map(function (c) {
    return ui.Button(c + ' ' + CLASS_NAME[c], function () { cur.cls = c; commit(); });
  }), ui.Panel.Layout.flow('horizontal'));

function sceneRow(codes) {
  return ui.Panel(codes.map(function (s) {
    return ui.Button(s, function () { cur.scene = s; commit(); });
  }), ui.Panel.Layout.flow('horizontal'));
}

function show(i) {
  idx = ((i % PTS.length) + PTS.length) % PTS.length;
  var p = PTS[idx];
  cur = {cls: null, scene: null};
  title.setValue('#' + (idx + 1) + '   PointID=' + p.pid + '   (orig ' + p.oid + ')' +
                 (p.done ? '   [done r1]' : ''));
  meta.setValue(
    'placeholder in file : Human_Class = ' + p.wc + ' (' + CLASS_NAME[p.wc] + ')  <- BAD, ignore\n' +
    'lon,lat             : ' + p.lon.toFixed(6) + ', ' + p.lat.toFixed(6));
  counter.setValue((idx + 1) + ' / ' + PTS.length + '   |   recorded: ' +
                   log.filter(function (x) { return x; }).length);
  var g = ee.Geometry.Point([p.lon, p.lat]);
  marker.setEeObject(g);
  marker.setVisParams({color: 'red'});
  Map.centerObject(g, ZOOM);
  var pre = log[idx];
  if (pre) {
    cur = {cls: pre.cls, scene: pre.scene};
    chosen.setValue('recorded: class=' + pre.cls + '  scene=' + pre.scene);
    noteBox.setValue(pre.note, false);
  } else {
    chosen.setValue('(not set)');
    noteBox.setValue('', false);
  }
}

function dump() {
  var done = log.filter(function (x) { return x; });
  print('==== Nanchang 26 blank-scene re-annotation (' + done.length + ' rows) ==== copy below ====');
  print('PointID,Original_ID,old_placeholder,Class_final,Scene_final,note');
  done.forEach(function (r) {
    print([r.pid, r.oid, r.wc, r.cls, r.scene,
           '"' + (r.note || '').replace(/"/g, "'") + '"'].join(','));
  });
  var flips = done.filter(function (r) { return r.cls !== r.wc; }).length;
  var byc = {1: 0, 2: 0, 3: 0};
  done.forEach(function (r) { byc[r.cls]++; });
  print('class_final counts: built=' + byc[1] + ' non-built=' + byc[2] + ' water=' + byc[3]);
  print('changed vs placeholder: ' + flips + ' / ' + done.length +
        '   (still missing: ' + (PTS.length - done.length) + ')');
}

Map.layers().add(marker);

var panel = ui.Panel({style: {position: 'top-left', width: '312px', padding: '8px'}});
panel.add(ui.Label('Nanchang 26 blank-scene re-annotation', {fontWeight: 'bold'}));
panel.add(counter);
panel.add(title);
panel.add(meta);
panel.add(ui.Panel(
  [ui.Button('◀ Prev', function () { show(idx - 1); }),
   ui.Button('Next ▶', function () { show(idx + 1); })],
  ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Label('1. class', {fontSize: '12px', margin: '6px 0 0 8px'}));
panel.add(clsPanel);
panel.add(ui.Label('2. scene code', {fontSize: '12px', margin: '6px 0 0 8px'}));
panel.add(sceneRow(['be', 'br']));
panel.add(sceneRow(['fe', 'veg', 'bl', 're', 'ct']));
panel.add(sceneRow(['we', 'wr', 'pf', 'tw']));
panel.add(chosen);
panel.add(noteBox);
panel.add(ui.Button('Dump log', dump));
panel.add(ui.Label(
  'be/br=built  fe=forest edge  veg=grass/green  bl=bare  re=road  ct=dry-field margin  ' +
  'we=shoreline  wr=open water  pf=paddy-as-water  tw=turbid pond',
  {fontSize: '11px', color: '#888'}));
panel.add(ui.Label('Primary judge = HYBRID basemap. The Human_Class in the file is a bad placeholder (= WC).',
                   {fontSize: '11px', color: '#888'}));
ui.root.insert(0, panel);

show(0);
