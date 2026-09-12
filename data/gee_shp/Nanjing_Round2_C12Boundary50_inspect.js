// =====================================
// Nanjing Round2: 50 Class1–Class2 boundary points
// =====================================
// 1) Upload data/gee_shp/Nanjing_Round2_C12Boundary50.zip to GEE Assets
// 2) Replace the asset path below
// 3) Change r2 = 1..50 (Round2ID / R2ID)
// 4) Fill CSV: data/Nanjing/04_manual/Nanjing_Round2_C12Boundary50.csv

var table = ee.FeatureCollection(
  'projects/healthy-area-463312-i4/assets/Nanjing_Round2_C12Boundary50'
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

var sortedPoints = points.sort('R2ID');

Map.setOptions('HYBRID');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(sortedPoints)
  .filterDate('2021-01-01', '2021-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

Map.addLayer(s2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'Sentinel-2 RGB');
Map.addLayer(s2, {bands: ['B8', 'B4', 'B3'], min: 0, max: 4000}, 'Sentinel-2 False Color', false);
Map.addLayer(sortedPoints, {color: 'yellow'}, 'Round2_C12_50');

// ---- 只改这里 ----
var r2 = 1;  // 1..50 = Round2ID

var point = points.filter(ee.Filter.eq('R2ID', r2)).first();

print('---- Round2 point ----');
print('Round2ID (R2ID):', r2);
print('Original_ID:', point.get('OrigID'));
print('WC Class:', point.get('Class'));
print('RF_Pred:', point.get('RFPred'));
print('margin:', point.get('margin'));
print('p1/p2/p3:', point.get('p1'), point.get('p2'), point.get('p3'));
print('lon/lat:', point.get('lon'), point.get('lat'));

Map.centerObject(point, 18);
Map.addLayer(ee.FeatureCollection([point]), {color: 'cyan'}, 'Current');

// 30m pixel box (approx)
var box = ee.Geometry.Point([
  ee.Number(point.get('lon')),
  ee.Number(point.get('lat'))
]).buffer(15, 1).bounds();
Map.addLayer(box, {color: 'white'}, '≈30m pixel');
