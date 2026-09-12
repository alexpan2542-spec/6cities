// =====================================
// Nanjing ErrorTop150 Manual Inspection
// =====================================
// Asset: upload data/Nanjing/04_manual/Nanjing_ErrorTop150.csv
// (or zip with lon/lat). If your asset name differs, edit ASSET below.
// =====================================

var ASSET = 'projects/healthy-area-463312-i4/assets/Nanjing_ErrorTop150';

var table = ee.FeatureCollection(ASSET);

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

print('N points:', sortedPoints.size());

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
  'Nanjing_ErrorTop150'
);

// Optional: color by error_type (string property — use filters)
Map.addLayer(
  sortedPoints.filter(ee.Filter.eq('error_type', 'C2_to_C1')),
  {color: 'orange'},
  'err C2→C1',
  false
);
Map.addLayer(
  sortedPoints.filter(ee.Filter.eq('error_type', 'C2_to_C3')),
  {color: 'magenta'},
  'err C2→C3',
  false
);
Map.addLayer(
  sortedPoints.filter(ee.Filter.eq('error_type', 'C1_to_C2')),
  {color: 'lime'},
  'err C1→C2',
  false
);

// =====================================
// Select Point
// =====================================

var p = 1;   // 修改这里：1~150

// 推荐：按 PointID 过滤（不依赖 list 顺序）
var point = sortedPoints.filter(ee.Filter.eq('PointID', p)).first();

// 备用：按排序后的列表取第 p 个
// var point = ee.Feature(
//   sortedPoints.toList(sortedPoints.size()).get(p - 1)
// );

// =====================================
// Print Information
// =====================================

print('====================');
print('NANJING ErrorTop150');
print('====================');
print('p / PointID:', p);
print('PointID:', point.get('PointID'));
print('Original_ID:', point.get('Original_ID'));
print('WC_Class:', point.get('WC_Class'));
print('RF_Pred (OOB):', point.get('RF_Pred'));
print('error_type:', point.get('error_type'));
print('margin:', point.get('margin'));
print('prob_1:', point.get('prob_1'));
print('prob_2:', point.get('prob_2'));
print('prob_3:', point.get('prob_3'));
print('NDVI_mean:', point.get('NDVI_mean'));
print('NDBI_mean:', point.get('NDBI_mean'));
print('MNDWI_mean:', point.get('MNDWI_mean'));
print('Human_Class:', point.get('Human_Class'));
print('Confidence:', point.get('Confidence'));
print('Scene:', point.get('Scene'));
print('Notes:', point.get('Notes'));
print('lon:', point.get('lon'));
print('lat:', point.get('lat'));
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
