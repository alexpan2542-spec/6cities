// =====================================
// Nanjing Phase-2 Manual Inspection
// Select points by AuditID only (no list index)
// =====================================
//
// Upload: figures/gee_shp/Nanjing_phase2_audit70.zip
// Asset example: projects/healthy-area-463312-i4/assets/Nanjing_phase2_audit70
//
 // Fill CSV HumanBetter: Human / Class / Uncertain
//   Human = imagery supports PredNew more than GTClass
//   Class = imagery supports GTClass more than PredNew
// =====================================

var table = ee.FeatureCollection(
  'projects/healthy-area-463312-i4/assets/Nanjing_phase2_audit70'
);

// Rebuild geometry from lon/lat (robust if shp geometry is odd)
var points = table.map(function(f) {
  return ee.Feature(
    ee.Geometry.Point([
      ee.Number(f.get('lon')),
      ee.Number(f.get('lat'))
    ]),
    f.toDictionary()
  );
});

// =====================================
// SELECT POINT — only change this
 // =====================================
var auditId = 'H01';  // H01-H40 or E01-E30

var pointFc = points.filter(ee.Filter.eq('AuditID', auditId));
var point = pointFc.first();

// Safety: confirm exactly one match
print('Matches for', auditId, ':', pointFc.size());
print('---- Current point ----');
print('AuditID:', point.get('AuditID'));
print('InspID:', point.get('InspID'));
print('Layer:', point.get('Layer'));
print('OrigID:', point.get('OrigID'));
print('GTClass:', point.get('GTClass'));
print('PredBase:', point.get('PredBase'));
print('PredNew:', point.get('PredNew'));
print('methods:', point.get('methods'));
print('NSeeds:', point.get('NSeeds'));
print('Scene:', point.get('Scene'));
print('lon:', point.get('lon'));
print('lat:', point.get('lat'));
print('Judge: PredNew vs GTClass -> HumanBetter = Human / Class / Uncertain');

// =====================================
// Imagery
// =====================================
Map.setOptions('HYBRID');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(points)
  .filterDate('2021-01-01', '2021-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

Map.addLayer(s2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'Sentinel-2 RGB');
Map.addLayer(s2, {bands: ['B8', 'B4', 'B3'], min: 0, max: 4000}, 'Sentinel-2 False Color', false);

Map.addLayer(
  points.filter(ee.Filter.eq('Layer', 'hurt_gt')),
  {color: 'red'},
  'All hurt_gt (H)'
);
Map.addLayer(
  points.filter(ee.Filter.eq('Layer', 'exp_conflict')),
  {color: 'orange'},
  'All exp_conflict (E)'
);

Map.centerObject(point, 17);
Map.addLayer(ee.Feature(point), {color: 'yellow'}, 'Current Point');
Map.addLayer(
  ee.Feature(ee.Feature(point).geometry().buffer(75).bounds()),
  {color: 'cyan'},
  '75m box',
  false
);
