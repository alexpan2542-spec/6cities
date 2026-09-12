// =====================================
// Changsha BAMS150 Manual Inspection
// =====================================

var table = ee.FeatureCollection(
  'projects/healthy-area-463312-i4/assets/Changsha_BAMS150'
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

// 按 PointID 排序
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
  {
    bands: ['B4', 'B3', 'B2'],
    min: 0,
    max: 3000
  },
  'Sentinel-2 RGB'
);

Map.addLayer(
  s2,
  {
    bands: ['B8', 'B4', 'B3'],
    min: 0,
    max: 4000
  },
  'Sentinel-2 False Color',
  false
);

// =====================================
// Show All Points
// =====================================

Map.addLayer(
  sortedPoints,
  {color: 'red'},
  'Changsha_BAMS150'
);

// =====================================
// Select Point
// =====================================

var p = 1;   // 修改这里：1~150

var point = ee.Feature(
  sortedPoints.toList(sortedPoints.size()).get(p - 1)
);

// 更稳的方法（推荐）
// var point = points.filter(
//   ee.Filter.eq('PointID', p)
// ).first();

// =====================================
// Print Information
// =====================================

print('====================');
print('CHANGSHA BAMS SAMPLE');
print('====================');

print('p / PointID:', p);

print('PointID:', point.get('PointID'));

// 兼容不同导出字段名
print('Original_ID:', point.get('Original_ID'));
print('Original_ID (OrigID):', point.get('OrigID'));

print('WC_Class (Class):', point.get('Class'));

print('Human_Class:', point.get('Human_Class'));
print('Human_Class (HumClass):', point.get('HumClass'));

print('Scene:', point.get('Scene'));
print('Confidence:', point.get('Confidence'));
print('Conf:', point.get('Conf'));

print('Corrected:', point.get('Corrected'));

print('Margin:', point.get('margin'));

print('BoundaryScore:', point.get('BoundaryScore'));
print('BoundaryScore (BndScore):', point.get('BndScore'));

print('Method:', point.get('Method'));

print('lon:', point.get('lon'));
print('lat:', point.get('lat'));

// 如果不确定字段名称
// print(point.propertyNames());

// =====================================
// Zoom
// =====================================

Map.centerObject(point, 17);

Map.addLayer(
  point,
  {color: 'yellow'},
  'Current Point'
);

Map.addLayer(
  ee.Feature(
    point.geometry()
      .buffer(75)
      .bounds()
  ),
  {color: 'cyan'},
  '75m Context Box',
  false
);
