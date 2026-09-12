// =====================================
// Nanjing BAMS150 Manual Inspection
// =====================================

// Asset（确认已上传 Nanjing_BAMS150.zip）
var table = ee.FeatureCollection(
  'projects/healthy-area-463312-i4/assets/Nanjing_BAMS150'
);

// =====================================
// Convert lon/lat -> Point Geometry
// =====================================

var points = table.map(function(f) {
  return ee.Feature(
    ee.Geometry.Point([
      ee.Number(f.get('lon')),
      ee.Number(f.get('lat'))
    ]),
    f.toDictionary()
  );
});

// 用 PointID 排序，保证 p=1 → PointID 1
var sortedPoints = points.sort('PointID');

// =====================================
// Background Imagery
// =====================================

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

// =====================================
// Show All Points
// =====================================

Map.addLayer(
  sortedPoints,
  {color: 'red'},
  'Nanjing_BAMS150'
);

// =====================================
// Select Point  ---- 只改这里 ----
// =====================================

var p = 1;  // 1..150 对应 PointID

var point = ee.Feature(
  sortedPoints.toList(sortedPoints.size()).get(p - 1)
);

// 更稳：按 PointID 精确选取（推荐，避免排序错位）
// var point = points.filter(ee.Filter.eq('PointID', p)).first();

// =====================================
// Print Information
// =====================================

print('---- Current point ----');
print('p / PointID:', p);
print('PointID:', point.get('PointID'));
print('Original_ID (OrigID):', point.get('OrigID'));
print('WC_Class (Class):', point.get('Class'));
print('Human_Class (HumClass):', point.get('HumClass'));
print('Scene:', point.get('Scene'));
print('Corrected:', point.get('Corrected'));
print('Margin:', point.get('margin'));
print('BoundaryScore (BndScore):', point.get('BndScore'));
print('Method:', point.get('Method'));
print('lon:', point.get('lon'));
print('lat:', point.get('lat'));

// 若字段名不对，取消下一行注释查看实际属性名
// print('property names', point.propertyNames());

// =====================================
// Zoom
// =====================================

Map.centerObject(point, 17);
Map.addLayer(point, {color: 'yellow'}, 'Current Point');
Map.addLayer(
  ee.Feature(point.geometry().buffer(75).bounds()),
  {color: 'cyan'},
  '75m box (~3x3 context)',
  false
);
