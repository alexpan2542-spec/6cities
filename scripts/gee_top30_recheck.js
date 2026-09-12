/**** Top-30 high-confidence disagreement re-check (expert vs ESA WorldCover) ****
 *
 * Paste into the GEE Code Editor (code.earthengine.google.com).
 *
 * Subset rule : Conf == 'high'  AND  Human_Class != WC_bams   (293 candidates)
 * Selection   : 5 per city, ranked by BAMS margin descending  -> 30 points
 * Source file : data/shared_reference/cross_city_second_ref_dw_esri/all_cities_BAMS150_second_ref.csv
 * Ontology    : 1 built | 2 non-built | 3 water
 * Imagery     : matches the BAMS pipeline -> S2_SR_HARMONIZED, 2021, CLOUDY_PIXEL_PERCENTAGE < 20, median
 *               ESA/WorldCover/v200
 * Basemap     : Google satellite ("HYBRID") for the sub-metre visual check
 *
 * Workflow: step through with Prev/Next, judge on the satellite basemap whether the
 * EXPERT label still holds, type v (verdict) + optional note, hit "Record".
 * At the end press "Dump log" and copy the console block into your results sheet.
 ******************************************************************************/

var YEAR = 2021;
var ZOOM = 18;

// ----------------------------------------------------------------------------
// 30 points
// ----------------------------------------------------------------------------
var PTS = [
  {id:5225, city:'Changsha', lon:113.30239574507756, lat:28.511494055376875, expert:'water',    wc:'nonbuilt', margin:0.593},
  {id:1346, city:'Changsha', lon:112.94666289256624, lat:28.142196642075340, expert:'water',    wc:'built',    margin:0.587},
  {id:1831, city:'Changsha', lon:112.98555994436860, lat:28.169146100598930, expert:'nonbuilt', wc:'built',    margin:0.583},
  {id:6236, city:'Changsha', lon:112.80104598501046, lat:28.214331359390137, expert:'water',    wc:'nonbuilt', margin:0.583},
  {id:10031,city:'Changsha', lon:112.79691373470352, lat:28.272093032159020, expert:'nonbuilt', wc:'water',    margin:0.577},
  {id:9811, city:'Hangzhou', lon:120.13059475350190, lat:30.410957706974557, expert:'built',    wc:'nonbuilt', margin:0.387},
  {id:9536, city:'Hangzhou', lon:120.25330475539164, lat:30.331723940355870, expert:'built',    wc:'nonbuilt', margin:0.367},
  {id:498,  city:'Hangzhou', lon:119.12529042100252, lat:29.700119030741583, expert:'nonbuilt', wc:'built',    margin:0.360},
  {id:4,    city:'Hangzhou', lon:119.07372537642044, lat:29.249883597375153, expert:'nonbuilt', wc:'built',    margin:0.347},
  {id:6042, city:'Hangzhou', lon:119.44284530252364, lat:29.554591808761327, expert:'built',    wc:'nonbuilt', margin:0.333},
  {id:558,  city:'Hefei',    lon:117.22848269432193, lat:31.762137742491777, expert:'nonbuilt', wc:'built',    margin:0.587},
  {id:4251, city:'Hefei',    lon:117.31391247784170, lat:31.927787080883416, expert:'nonbuilt', wc:'built',    margin:0.580},
  {id:6480, city:'Hefei',    lon:117.01657011879814, lat:31.873528837722596, expert:'water',    wc:'nonbuilt', margin:0.577},
  {id:2773, city:'Hefei',    lon:117.37095549838328, lat:32.458242256155990, expert:'nonbuilt', wc:'built',    margin:0.573},
  {id:3319, city:'Hefei',    lon:117.47956181623334, lat:32.033608621352690, expert:'nonbuilt', wc:'built',    margin:0.567},
  {id:4402, city:'Nanchang', lon:115.89125056238160, lat:28.684779073683530, expert:'nonbuilt', wc:'built',    margin:0.537},
  {id:2863, city:'Nanchang', lon:115.96787685611700, lat:28.725113429940496, expert:'nonbuilt', wc:'built',    margin:0.530},
  {id:4930, city:'Nanchang', lon:115.91954749383136, lat:28.503139723234560, expert:'nonbuilt', wc:'built',    margin:0.527},
  {id:4239, city:'Nanchang', lon:116.00596542416368, lat:28.382316317520488, expert:'nonbuilt', wc:'built',    margin:0.513},
  {id:3382, city:'Nanchang', lon:116.05330663963676, lat:28.742181420338767, expert:'nonbuilt', wc:'built',    margin:0.507},
  {id:4904, city:'Nanjing',  lon:118.61664930287183, lat:32.215337803330070, expert:'nonbuilt', wc:'built',    margin:0.527},
  {id:8323, city:'Nanjing',  lon:118.91938155362013, lat:32.099814457792306, expert:'built',    wc:'nonbuilt', margin:0.507},
  {id:4317, city:'Nanjing',  lon:118.92926302174544, lat:32.375956576130640, expert:'nonbuilt', wc:'built',    margin:0.507},
  {id:3446, city:'Nanjing',  lon:118.56535550014860, lat:31.793399114379135, expert:'nonbuilt', wc:'built',    margin:0.500},
  {id:6074, city:'Nanjing',  lon:118.84239593377106, lat:31.877930582614780, expert:'water',    wc:'nonbuilt', margin:0.493},
  {id:1804, city:'Wuhan',    lon:114.18031927224756, lat:30.548693456703127, expert:'nonbuilt', wc:'built',    margin:0.560},
  {id:1326, city:'Wuhan',    lon:114.15498678123540, lat:30.420953023301330, expert:'nonbuilt', wc:'built',    margin:0.560},
  {id:8894, city:'Wuhan',    lon:114.31470723875184, lat:31.249469209844765, expert:'water',    wc:'nonbuilt', margin:0.553},
  {id:4399, city:'Wuhan',    lon:113.78128762304168, lat:30.319982385366295, expert:'nonbuilt', wc:'built',    margin:0.553},
  {id:865,  city:'Wuhan',    lon:114.15363930830922, lat:30.523810123333018, expert:'nonbuilt', wc:'built',    margin:0.547}
];

var fc = ee.FeatureCollection(PTS.map(function (p, k) {
  return ee.Feature(ee.Geometry.Point([p.lon, p.lat]), {
    k: k + 1, id: p.id, city: p.city, expert: p.expert, wc: p.wc, margin: p.margin
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
var s2True  = {bands: ['B4', 'B3', 'B2'], min: 200,  max: 2500};
var s2False = {bands: ['B8', 'B4', 'B3'], min: 200,  max: 4000};   // veg red, water black

var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');

Map.setOptions('HYBRID');                       // Google sub-metre basemap
Map.addLayer(s2,  s2True,  'S2 ' + YEAR + ' true colour');
Map.addLayer(s2,  s2False, 'S2 ' + YEAR + ' false colour (NIR)', false);
Map.addLayer(wc,  {}, 'ESA WorldCover v200', false, 0.5);
Map.addLayer(fc.style({color: 'ff0', fillColor: '00000000', pointSize: 10, width: 2}), {}, 'all 30');

// ----------------------------------------------------------------------------
// Stepper UI
// ----------------------------------------------------------------------------
var idx = 0;
var log = [];

var title   = ui.Label('', {fontWeight: 'bold', fontSize: '15px'});
var meta    = ui.Label('', {whiteSpace: 'pre', fontSize: '13px'});
var counter = ui.Label('', {fontSize: '12px', color: '#666'});
var marker  = ui.Map.Layer(ee.Geometry.Point([0, 0]), {color: 'red'}, 'current');

var verdictBox = ui.Textbox({placeholder: 'v: 1=expert holds  0=WC right  ?=unsure', style: {width: '260px'}});
var noteBox    = ui.Textbox({placeholder: 'note (scene / why)', style: {width: '260px'}});

function show(i) {
  idx = ((i % PTS.length) + PTS.length) % PTS.length;
  var p = PTS[idx];
  title.setValue('#' + (idx + 1) + '  ' + p.city + '  id=' + p.id);
  meta.setValue(
    'expert : ' + p.expert + '\n' +
    'WC     : ' + p.wc + '\n' +
    'margin : ' + p.margin + '\n' +
    'lon,lat: ' + p.lon.toFixed(6) + ', ' + p.lat.toFixed(6));
  counter.setValue((idx + 1) + ' / ' + PTS.length + '   |   recorded: ' + log.length);
  var g = ee.Geometry.Point([p.lon, p.lat]);
  marker.setEeObject(g);
  marker.setVisParams({color: 'red'});
  Map.centerObject(g, ZOOM);
  var pre = log[idx];
  verdictBox.setValue(pre ? pre.v : '', false);
  noteBox.setValue(pre ? pre.note : '', false);
}

function record() {
  var p = PTS[idx];
  log[idx] = {k: idx + 1, id: p.id, city: p.city, expert: p.expert, wc: p.wc,
              margin: p.margin, lon: p.lon, lat: p.lat,
              v: verdictBox.getValue(), note: noteBox.getValue()};
  counter.setValue((idx + 1) + ' / ' + PTS.length + '   |   recorded: ' +
                   log.filter(function (x) { return x; }).length);
}

function dump() {
  var done = log.filter(function (x) { return x; });
  print('==== re-check log (' + done + ' rows) ==== copy below ====');
  print('k,id,city,expert,wc,margin,lon,lat,verdict,note');
  done.forEach(function (r) {
    print([r.k, r.id, r.city, r.expert, r.wc, r.margin, r.lon, r.lat,
           r.v, '"' + (r.note || '').replace(/"/g, "'") + '"'].join(','));
  });
  var hold = done.filter(function (r) { return r.v === '1'; }).length;
  var wcr  = done.filter(function (r) { return r.v === '0'; }).length;
  var uns  = done.filter(function (r) { return r.v === '?'; }).length;
  print('expert holds: ' + hold + '   WC right: ' + wcr + '   unsure: ' + uns +
        '   (of ' + done.length + ')');
}

Map.layers().add(marker);

var panel = ui.Panel({style: {position: 'top-left', width: '300px', padding: '8px'}});
panel.add(ui.Label('Top-30 disagreement re-check', {fontWeight: 'bold'}));
panel.add(counter);
panel.add(title);
panel.add(meta);
panel.add(ui.Panel(
  [ui.Button('◀ Prev', function () { show(idx - 1); }),
   ui.Button('Next ▶', function () { show(idx + 1); })],
  ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Label('verdict', {fontSize: '12px', margin: '6px 0 0 8px'}));
panel.add(verdictBox);
panel.add(noteBox);
panel.add(ui.Panel(
  [ui.Button('Record', record),
   ui.Button('Dump log', dump)],
  ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Label('Tip: toggle "ESA WorldCover v200" / false-colour layers in the layer list.',
                   {fontSize: '11px', color: '#888'}));
ui.root.insert(0, panel);

show(0);
