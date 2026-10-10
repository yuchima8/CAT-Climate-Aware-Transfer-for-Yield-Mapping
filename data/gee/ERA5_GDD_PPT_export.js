var REGION = 'US';
var CROP = 'soybean';
var YEARS = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];
var US_STATES = {
  soybean: ['17', '18', '19', '20', '26', '27', '29', '31', '38', '39', '46', '55'],
  corn: ['17', '18', '19', '20', '26', '27', '29', '31', '38', '39', '46', '55'],
  winter_wheat: ['08', '17', '18', '20', '26', '27', '29', '31', '38', '39', '40', '46', '48', '55']
};
var ARG_BOUNDARIES = 'projects/your-project/assets/argentina_departments';
var EXPORT_FOLDER = 'CAT_ERA5';
var SCALE = 1000;

var THRESHOLDS = {soybean: [8, 30], corn: [8, 30], winter_wheat: [0, 26]};
var WINDOWS = {
  US: {soybean: [3, 9], corn: [3, 9], winter_wheat: [9, 6]},
  Argentina: {soybean: [10, 5], corn: [9, 6], winter_wheat: [6, 12]}
};

var units = REGION === 'US'
  ? ee.FeatureCollection('TIGER/2018/Counties').filter(ee.Filter.inList('STATEFP', US_STATES[CROP])).select(['GEOID'])
  : ee.FeatureCollection(ARG_BOUNDARIES).select(['name_1', 'name_2']);
var idColumns = REGION === 'US' ? ['GEOID'] : ['name_1', 'name_2'];

var tBase = THRESHOLDS[CROP][0];
var tUpper = THRESHOLDS[CROP][1];
var hours = ee.List.sequence(0, 23);

function dailyGDD(image) {
  var tmin = image.select('temperature_2m_min').subtract(273.15);
  var tmax = image.select('temperature_2m_max').subtract(273.15);
  var mean = tmax.add(tmin).divide(2);
  var amplitude = tmax.subtract(tmin).divide(2);
  var hourly = ee.ImageCollection(hours.map(function (h) {
    var phase = ee.Number(h).subtract(14).multiply(Math.PI / 12).cos();
    var t = mean.add(amplitude.multiply(phase));
    return t.max(tBase).min(tUpper).subtract(tBase);
  }));
  var gdd = hourly.sum().divide(24).rename('GDD');
  var ppt = image.select('total_precipitation_sum').rename('PPT');
  return ee.Image(gdd.addBands(ppt).copyProperties(image, ['system:time_start']));
}

function seasonMonths(year) {
  var months = WINDOWS[REGION][CROP];
  var startYear = months[0] > months[1] ? year - 1 : year;
  var start = ee.Date.fromYMD(startYear, months[0], 1);
  var end = ee.Date.fromYMD(year, months[1], 1).advance(1, 'month');
  var n = end.difference(start, 'month').round();
  return ee.List.sequence(0, n.subtract(1)).map(function (k) { return start.advance(k, 'month'); });
}

YEARS.forEach(function (year) {
  var monthly = seasonMonths(year).map(function (d) {
    var start = ee.Date(d);
    var month = start.get('month').format('%d');
    var days = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(start, start.advance(1, 'month')).map(dailyGDD);
    var total = days.sum();
    return total.select('GDD').rename(ee.String('GDD_').cat(month))
      .addBands(total.select('PPT').rename(ee.String('ppt_').cat(month)));
  });
  var stack = ee.ImageCollection(monthly).toBands().regexpRename('^[0-9]+_', '');
  var stats = stack.reduceRegions({
    collection: units,
    reducer: ee.Reducer.mean(),
    scale: SCALE,
    tileScale: 4
  }).map(function (f) { return f.set('year', year); });
  var prefix = 'ERA5_Monthly_GDD_PPT_' + REGION + '_' + CROP + '_' + year;
  Export.table.toDrive({
    collection: stats,
    description: prefix,
    folder: EXPORT_FOLDER,
    fileNamePrefix: prefix,
    fileFormat: 'CSV'
  });
});
