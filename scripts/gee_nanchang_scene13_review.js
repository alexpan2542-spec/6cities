/**** Nanchang 13 unlabeled-scene points -> assign a special scene code ********
 *
 * Paste into the GEE Code Editor (code.earthengine.google.com).
 *
 * Source : data/cities/Nanchang/04_manual/Nanchang_UnlabeledScene_ToReview13.csv
 * Task   : for each point, decide which fine scene code it is, from a small
 *          candidate set fixed by its HC_label:
 *            built candidates : be | br
 *              be = building edge      -> marker on a single building roof / footprint edge
 *              br = built-up edge      -> mixed settlement fabric (houses + roads + lots), not one building
 *            water candidates : we | wr | pf | tw
 *              we = water edge         -> on the land/water shoreline, half water half land
 *              wr = open water body    -> marker fully inside a continuous open water surface
 *              pf = paddy / cropland   -> field-bund grid, regular rectangular plots, dirt track / farmhouse nearby
 *              tw = turbid water       -> yellow / green / sediment-laden pond, usually still water
 *
 * Imagery : matches the BAMS pipeline -> S2_SR_HARMONIZED, 2021,
 *           CLOUDY_PIXEL_PERCENTAGE < 20, median.  ESA/WorldCover/v200 overlay.
 * Basemap : Google satellite ("HYBRID") for the sub-metre visual check -> primary.
 *
 * Workflow: step with Prev/Next, judge on the HYBRID basemap what sits under the
 *           marker and in the ~30 m (3x3 S2-pixel) neighbourhood, click the code
 *           button, add an optional note, hit "Record".  At the end press
 *           "Dump log" and copy the console CSV block back into the review sheet
 *           as columns  Scene_final,note .
 ******************************************************************************/

var YEAR = 2021;
var ZOOM = 18;

// ----------------------------------------------------------------------------
// 13 points  (PointID, Original_ID, HC_label, lat, lon)  from the review CSV
// cand = the allowed scene codes for that point
// ----------------------------------------------------------------------------
var BUILT = ['be', 'br'];
var WATER = ['we', 'wr', 'pf', 'tw'];

var PTS = [
  {pid: 106, oid: 4508,  hc: 'built', lat: 28.53745536708793,  lon: 115.82369725301582, cand: BUILT},
  {pid: 119, oid: 4063,  hc: 'built', lat: 28.636629374454724, lon: 115.8032156645379,  cand: BUILT},
  {pid: 120, oid: 1708,  hc: 'built', lat: 28.563416678798983, lon: 115.79629863685018, cand: BUILT},
  {pid: 127, oid: 2157,  hc: 'built', lat: 28.851057232774053, lon: 115.6116948459636,  cand: BUILT},
  {pid: 130, oid: 1041,  hc: 'built', lat: 28.74325939867971,  lon: 115.9257458692918,  cand: BUILT},
  {pid: 137, oid: 3553,  hc: 'built', lat: 28.51266186524623,  lon: 116.3961037520568,  cand: BUILT},
  {pid: 100, oid: 14010, hc: 'water', lat: 28.29949164832467,  lon: 116.29369580966716, cand: WATER},
  {pid: 109, oid: 10890, hc: 'water', lat: 28.670585692194443, lon: 116.42898209145557, cand: WATER},
  {pid: 110, oid: 10319, hc: 'water', lat: 28.598989964050116, lon: 116.08277138095588, cand: WATER},
  {pid: 126, oid: 10584, hc: 'water', lat: 28.367314452275693, lon: 115.67736169323275, cand: WATER},
  {pid: 128, oid: 13035, hc: 'water', lat: 28.780359819913848, lon: 115.91442709671188, cand: WATER},
  {pid: 136, oid: 14949, hc: 'water', lat: 28.811172034159146, lon: 116.16200278901523, cand: WATER},
  {pid: 149, oid: 13160, hc: 'water', lat: 28.86471162509267,  lon: 115.98763979236764, cand: WATER}
];

var fc = ee.FeatureCollection(PTS.map(function (p, k) {
  return ee.Feature(ee.Geometry.Point([p.lon, p.lat]), {
    k: k + 1, pid: p.pid, oid: p.oid, hc: p.hc
  });
}));

// ----------------------------------------------------------------------------
// Imagery (same recipe as the BAMS selection pipeline)
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

Map.setOptions('HYBRID');                       // Google sub-metre basemap -> primary judge
Map.addLayer(s2, s2True,  'S2 ' + YEAR + ' true colour', false);
Map.addLayer(s2, s2False, 'S2 ' + YEAR + ' false colour (NIR): veg red, water black', false);
Map.addLayer(wc, {}, 'ESA WorldCover v200', false, 0.5);
Map.addLayer(fc.style({color: 'ff0', fillColor: '00000000', pointSize: 10, width: 2}), {}, 'all 13');

// ----------------------------------------------------------------------------
// Stepper UI
// ----------------------------------------------------------------------------
var idx = 0;
var log = [];

var title   = ui.Label('', {fontWeight: 'bold', fontSize: '15px'});
var meta    = ui.Label('', {whiteSpace: 'pre', fontSize: '13px'});
var counter = ui.Label('', {fontSize: '12px', color: '#666'});
var chosen  = ui.Label('', {fontSize: '13px', fontWeight: 'bold', color: '#0a0'});
var marker  = ui.Map.Layer(ee.Geometry.Point([0, 0]), {color: 'red'}, 'current');

var noteBox = ui.Textbox({placeholder: 'note (what you see / why)', style: {width: '260px'}});
var codeButtons = ui.Panel([], ui.Panel.Layout.flow('horizontal'));

function setCode(code) {
  var p = PTS[idx];
  log[idx] = {k: idx + 1, pid: p.pid, oid: p.oid, hc: p.hc, lat: p.lat, lon: p.lon,
              scene: code, note: noteBox.getValue()};
  chosen.setValue('Scene_final = ' + code);
  counter.setValue((idx + 1) + ' / ' + PTS.length + '   |   recorded: ' +
                   log.filter(function (x) { return x; }).length);
}

function show(i) {
  idx = ((i % PTS.length) + PTS.length) % PTS.length;
  var p = PTS[idx];
  title.setValue('#' + (idx + 1) + '   PointID=' + p.pid + '   (orig ' + p.oid + ')');
  meta.setValue(
    'HC_label : ' + p.hc + '\n' +
    'options  : ' + p.cand.join('  |  ') + '\n' +
    'lon,lat  : ' + p.lon.toFixed(6) + ', ' + p.lat.toFixed(6));
  counter.setValue((idx + 1) + ' / ' + PTS.length + '   |   recorded: ' +
                   log.filter(function (x) { return x; }).length);

  var g = ee.Geometry.Point([p.lon, p.lat]);
  marker.setEeObject(g);
  marker.setVisParams({color: 'red'});
  Map.centerObject(g, ZOOM);

  codeButtons.clear();
  p.cand.forEach(function (c) {
    codeButtons.add(ui.Button(c, function () { setCode(c); }));
  });

  var pre = log[idx];
  chosen.setValue(pre ? 'Scene_final = ' + pre.scene : '(not set)');
  noteBox.setValue(pre ? pre.note : '', false);
}

function dump() {
  var done = log.filter(function (x) { return x; });
  print('==== Nanchang scene-13 review (' + done.length + ' rows) ==== copy below ====');
  print('PointID,Original_ID,HC_label,lat,lon,Scene_final,note');
  done.forEach(function (r) {
    print([r.pid, r.oid, r.hc, r.lat, r.lon, r.scene,
           '"' + (r.note || '').replace(/"/g, "'") + '"'].join(','));
  });
  var by = {};
  done.forEach(function (r) { by[r.scene] = (by[r.scene] || 0) + 1; });
  print('counts: ' + JSON.stringify(by) + '   (of ' + done.length + '/' + PTS.length + ')');
}

Map.layers().add(marker);

var panel = ui.Panel({style: {position: 'top-left', width: '300px', padding: '8px'}});
panel.add(ui.Label('Nanchang 13 unlabeled-scene review', {fontWeight: 'bold'}));
panel.add(counter);
panel.add(title);
panel.add(meta);
panel.add(ui.Panel(
  [ui.Button('◀ Prev', function () { show(idx - 1); }),
   ui.Button('Next ▶', function () { show(idx + 1); })],
  ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Label('pick scene code:', {fontSize: '12px', margin: '6px 0 0 8px'}));
panel.add(codeButtons);
panel.add(chosen);
panel.add(noteBox);
panel.add(ui.Button('Dump log', dump));
panel.add(ui.Label(
  'be=single building edge  br=settlement fabric  |  ' +
  'we=shoreline  wr=open water  pf=paddy/cropland  tw=turbid pond',
  {fontSize: '11px', color: '#888'}));
panel.add(ui.Label('Primary judge = HYBRID basemap. Toggle S2 false-colour / WorldCover for context.',
                   {fontSize: '11px', color: '#888'}));
ui.root.insert(0, panel);

show(0);
