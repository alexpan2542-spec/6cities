// =====================================
// Hangzhou BAMS150 Manual Inspection
// =====================================
// Upload data/Hangzhou/03_top150/Hangzhou_BAMS150.csv as Asset:
//   projects/healthy-area-463312-i4/assets/Hangzhou_BAMS150
// Then set p = PointID (1..150) while labeling
// Manual file: data/Hangzhou/04_manual/Hangzhou_BAMS150_Manual.csv
// Columns: PointID, Original_ID, Human_Class, Scene, Confidence, Notes
// Human_Class: 1=built, 2=non-built, 3=water
// Confidence: H / M / L

var table = ee.FeatureCollection(
  'projects/healthy-area-463312-i4/assets/Hangzhou_BAMS150'
);

var points = table.map(function(f) {
  return ee.Feature(
    ee.Geometry.Point([
      ee.Number(f.get('lon')),
      ee.Number(f.get('lat'))
    ]),
    f.toDictionary()
  );
});

var sortedPoints = points.sort('PointID');

Map.setOptions('HYBRID');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(sortedPoints)
  .filterDate('2021-01-01', '2021-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

Map.addLayer(
  s2,
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000},
  'Sentinel-2 RGB'
);

Map.addLayer(
  s2,
  {bands: ['B8', 'B4', 'B3'], min: 0, max: 4000},
  'Sentinel-2 False Color',
  false
);

Map.addLayer(sortedPoints, {color: 'red'}, 'Hangzhou_BAMS150');

// ---- only change p ----
var p = 1;  // 1..150

var point = points.filter(ee.Filter.eq('PointID', p)).first();

print('---- Current point ----');
print('PointID:', p);
print('Original_ID:', point.get('Original_ID'));
print('WC Class:', point.get('Class'));
print('margin:', point.get('margin'));
print('lon:', point.get('lon'));
print('lat:', point.get('lat'));

Map.centerObject(point, 16);
Map.addLayer(ee.FeatureCollection([point]), {color: 'yellow'}, 'Current');
