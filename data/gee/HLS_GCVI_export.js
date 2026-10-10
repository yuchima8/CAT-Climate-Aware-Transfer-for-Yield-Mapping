var REGION = 'US';
var CROP = 'soybean';
var YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];
var US_STATES = {
  soybean: ['17', '18', '19', '20', '26', '27', '29', '31', '38', '39', '46', '55'],
  corn: ['17', '18', '19', '20', '26', '27', '29', '31', '38', '39', '46', '55'],
  winter_wheat: ['08', '17', '18', '20', '26', '27', '29', '31', '38', '39', '40', '46', '48', '55']
};
var ARG_BOUNDARIES = 'projects/your-project/assets/argentina_departments';
var ARG_CROP_MAP = 'projects/your-project/assets/argentina_crop_map_';
var ARG_CROP_CLASS = {soybean: 1, corn: 2, winter_wheat: 3};
var CDL_CLASS = {soybean: 5, corn: 1, winter_wheat: 24};
var EXPORT_FOLDER = 'CAT_HLS';
var SCALE = 30;

var WINDOWS = {
  US: {soybean: [3, 9], corn: [3, 9], winter_wheat: [9, 6]},
  Argentina: {soybean: [9, 5], corn: [9, 6], winter_wheat: [5, 12]}
};

var units = REGION === 'US'
  ? ee.FeatureCollection('TIGER/2018/Counties').filter(ee.Filter.inList('STATEFP', US_STATES[CROP])).select(['GEOID'])
  : ee.FeatureCollection(ARG_BOUNDARIES).select(['name_1', 'name_2']);
var idColumns = REGION === 'US' ? ['GEOID'] : ['name_1', 'name_2'];

function seasonRange(year) {
  var months = WINDOWS[REGION][CROP];
  var startYear = months[0] > months[1] ? year - 1 : year;
  var start = ee.Date.fromYMD(startYear, months[0], 1);
  var end = ee.Date.fromYMD(year, months[1], 1).advance(1, 'month');
  return {start: start, end: end};
}

function cropMask(year) {
  if (REGION === 'US') {
    var cdl = ee.ImageCollection('USDA/NASS/CDL').filter(ee.Filter.calendarRange(year, year, 'year')).first();
    return cdl.select('cropland').eq(CDL_CLASS[CROP]);
  }
  return ee.Image(ARG_CROP_MAP + year).eq(ARG_CROP_CLASS[CROP]);
}

function prepare(image, bands) {
  var clear = image.select('Fmask').bitwiseAnd(62).eq(0);
  var r = image.select(bands, ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']);
  var gcvi = r.select('NIR').divide(r.select('GREEN')).subtract(1).rename('GCVI');
  var ndvi = r.normalizedDifference(['NIR', 'RED']).rename('NDVI');
  return ee.Image(r.addBands(gcvi).addBands(ndvi)
    .updateMask(clear)
    .copyProperties(image, ['system:time_start']));
}

function hls(start, end, region) {
  var l30 = ee.ImageCollection('NASA/HLS/HLSL30/v002').filterDate(start, end).filterBounds(region)
    .map(function (img) { return prepare(img, ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']); });
  var s30 = ee.ImageCollection('NASA/HLS/HLSS30/v002').filterDate(start, end).filterBounds(region)
    .map(function (img) { return prepare(img, ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']); });
  return l30.merge(s30);
}

YEARS.forEach(function (year) {
  var range = seasonRange(year);
  var mask = cropMask(year);
  var region = units.geometry().bounds();
  var samples = hls(range.start, range.end, region).map(function (img) {
    var date = ee.Date(img.get('system:time_start'));
    var stats = img.updateMask(mask).reduceRegions({
      collection: units.filterBounds(img.geometry()),
      reducer: ee.Reducer.mean(),
      scale: SCALE,
      tileScale: 4
    });
    return stats.filter(ee.Filter.notNull(['GCVI'])).map(function (f) {
      return f.set({
        year: year,
        date: date.format('YYYY-MM-dd'),
        DOY: date.getRelative('day', 'year').add(1),
        season_day: date.difference(range.start, 'day').floor()
      });
    });
  }).flatten();
  var prefix = (REGION === 'US' ? 'county_samples_US_HLS_S2_' : 'district_samples_Argentina_HLS_S2_') + CROP + '_' + year;
  Export.table.toDrive({
    collection: samples,
    description: prefix,
    folder: EXPORT_FOLDER,
    fileNamePrefix: prefix,
    fileFormat: 'CSV',
    selectors: idColumns.concat(['year', 'date', 'DOY', 'season_day', 'GCVI', 'NDVI', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'])
  });
});
