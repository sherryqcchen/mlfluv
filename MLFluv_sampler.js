//--------------Start using this script by defining a point on the map. --------------//
//---------------The point is buffered to a squre of target area. --------------------//
var target_area = ee.Number(2048).multiply(ee.Number(2048))
//asjudt maxError (second parameter in the bracket)    in the Point.buffer() to get the minimum error.
var AOI = point.buffer(target_area.sqrt().divide(2), 4).bounds(); 
print('Buffered AOI (m2):', AOI.area(1))
print("Expected area:",  target_area)
print("Error:", AOI.area(1).subtract(target_area))
Map.addLayer(AOI, null, 'point buffered AOI')
Map.centerObject(AOI,14)

//--------------------Get label from Dynamic world------------------------------------//
var YEAR = 2016;
var years = ee.List.sequence(2016, 2023, 1);
print(years)

var start_date = ee.Date.fromYMD(YEAR, 1, 1);
var end_date = start_date.advance(1, 'year');

var COL_FILTER = ee.Filter.and(
    ee.Filter.bounds(AOI),
    ee.Filter.date(start_date, end_date));
    
var dwCol = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filter(COL_FILTER);
var s2Col = ee.ImageCollection('COPERNICUS/S2').filter(COL_FILTER);

// Join corresponding DW and S2 images (by system:index).
var DwS2Col = ee.Join.saveFirst('s2_img').apply(dwCol, s2Col,
    ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'}));

// Define a function to count masked pixels in each DW image label.
var countMaskedPixels = function(image){
  var image = ee.Image(image);
  var imagePixels = image.select('label').reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: AOI,
    scale: image.select('label').projection().nominalScale()
  });
  var totalPixels = ee.Image(1).clip(AOI).reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: AOI,
    scale: image.select('label').projection().nominalScale()
  });
  var masked_percent = ee.Number(imagePixels.get('label')).divide(totalPixels.get('constant'));
  return image.set('masked_percentage', masked_percent)
              .set('date', image.date().format('YYYY-MM-dd'));
};

// add masked percentage in the property for each image in the collection
var DwS2Col_label = ee.ImageCollection(DwS2Col).map(countMaskedPixels);
print("How many labels included in the year?", DwS2Col_label.size());

// Plot non-masked pixel count in each image 
var chart = ui.Chart.image.series({
  imageCollection: DwS2Col_label.select('label'),
  region: AOI,
  reducer: ee.Reducer.count(),
  scale: DwS2Col_label.first().projection().nominalScale(),
  xProperty: 'system:time_start'
})
.setOptions({
  title: 'Pixel counts of labels in Dynamic World for AOI by date', 
  hAxis: {title: 'Date', titleTextStyle: {italic: false, bold: true}},
  vAxis: {title: 'Pixel count', titleTextStyle: {italic: false, bold: true}}
})
print(chart);

// Get the first image date (system: time_start) with max masked_percentage.
var max_percent = DwS2Col_label.aggregate_max('masked_percentage');
print("Maximum percentage of label unmasked:", max_percent);

var percent_arr = ee.Array(DwS2Col_label.aggregate_array('masked_percentage'))
var date_arr = ee.Array(DwS2Col_label.aggregate_array('system:time_start'))
//get the first max percent image index
var max_idx = percent_arr.argmax()
var max_time_start = date_arr.get(max_idx);
print("I find the most clear date for DW lable and S2:", ee.Date(max_time_start));

var clear_date = ee.Date(max_time_start);
// TODO: filter other image with this date
var new_col = DwS2Col.filterDate(clear_date, clear_date.advance(1, 'day'));//.filter(ee.Filter.eq('system:time_start', max_time_start));
print('DW label of clear date:', new_col)

// Extract an example DW image and its source S2 image.
var dwImage = ee.Image(new_col.first());
var clear_s2 = ee.Image(dwImage.get('s2_img')).select('B.*');
print('Sentinel-2 image of clear date:', clear_s2);

// Create a visualization that blends DW class label with probability.
// Define list pairs of DW LULC label and color.
var CLASS_NAMES = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops',
    'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'];

var VIS_PALETTE = ['419bdf', '397d49', '88b053', '7a87c6', 'e49635', 'dfc35a', 'c4281b',
    'a59b8f', 'b39fe1'];

// Create an RGB image of the label (most likely class) on [0, 1].
var dwRgb = dwImage.select('label').visualize({min: 0, max: 8, palette: VIS_PALETTE}).divide(255);
// Get the most likely class probability.
var top1Prob = dwImage.select(CLASS_NAMES).reduce(ee.Reducer.max());
// Create a hillshade of the most likely class probability on [0, 1];
var top1ProbHillshade = ee.Terrain.hillshade(top1Prob.multiply(100)).divide(255);
// Combine the RGB image with the hillshade.
var dwRgbHillshade = dwRgb.multiply(top1ProbHillshade);

// Display the Dynamic World visualization with the source Sentinel-2 image.
Map.addLayer(dwRgbHillshade, {min: 0, max: 0.65}, 'Dynamic World V1 label hillshade of the clear date');
Map.addLayer(clear_s2, s2VisParam, 'Clear S2 image of the clear date');

// Get S2 image from a cloudy date within a month after the chosen clear date.
var cloudy_s2 = s2Col.filterDate(clear_date.advance(1, 'day'), clear_date.advance(30, 'day'))
                    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 60))
                    .filter(ee.Filter.gte('CLOUDY_PIXEL_PERCENTAGE', 5))
                    .sort('system:time_start',true)
                    .first()
                    .select('B.*');

var cloudy_date = ee.Date(cloudy_s2.get('system:time_start'));
print("The date of chosen cloudy Sentinel-2 image is:", cloudy_date);
Map.addLayer(cloudy_s2, s2VisParam, 'Clear S2 image of the clear date');

// // // -------------------Fethcing and preprocessing Sentinel-1 images--------------------//
// Get preprocessed S1 image of clear date
var clear_range = clear_date.getRange('week');
var cloudy_range = cloudy_date.getRange('week');

print("Getting Sentinel-1 image collection...")

var clear_s1_col = getS1Preprocess(clear_range.start(), clear_range.end(), AOI)
if (clear_s1_col.size().getInfo() === 0) {
  // For situation that not even one S1 image can be found in 1 week, extend to 2 weeks
  var extend_date = clear_range.end();
  var new_end_date = extend_date.advance(1, 'week');
  var clear_s1_col = getS1Preprocess(clear_range.start(), new_end_date, AOI)
}
else {
  clear_s1_col = clear_s1_col
}
var clear_s1 = ee.Image(clear_s1_col.first()).select('VV','VH');
var clear_s1_date = ee.Date(clear_s1.get('system:time_start'));
print("The date of chosen Sentinel-1 image for clear date is: ", clear_s1_date);

var cloudy_s1_col = getS1Preprocess(cloudy_range.start(), cloudy_range.end(), AOI);
if (cloudy_s1_col.size().getInfo() === 0) {
  // For situation that not even one S1 image can be found in 1 week, extend to 2 weeks
  var extend_date = cloudy_range.end();
  var new_end_date = extend_date.advance(1, 'week');
  var cloudy_s1_col = getS1Preprocess(cloudy_range.start(), new_end_date, AOI)
}
else {
  cloudy_s1_col = cloudy_s1_col
}
var cloudy_s1 = ee.Image(cloudy_s1_col.first()).select('VV','VH');
var cloudy_s1_date = ee.Date(cloudy_s1.get('system:time_start'));
print("The date of chosen Sentinel-1 image for cloudy date is: ", cloudy_s1_date);


var wrapper = require('users/adugnagirma/gee_s1_ard:wrapper');
var helper = require('users/adugnagirma/gee_s1_ard:utilities');

var clear_s1_view = clear_s1_col.map(helper.add_ratio_lin).map(helper.add_ratio_lin).map(helper.lin_to_db2);
var cloudy_s1_view = cloudy_s1_col.map(helper.add_ratio_lin).map(helper.add_ratio_lin).map(helper.lin_to_db2);

var s1_visparam = {bands:['VV','VH','VVVH_ratio'],min: [-20, -25, 1],max: [0, -5, 15]};

Map.addLayer(clear_s1_view, s1_visparam, 'S1 from a clear date');
Map.addLayer(cloudy_s1_view, s1_visparam, 'S1 from a cloudy date');

// -------------Get averaged accumulative rainfall data ---------------//
var rainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .select('precipitation')
                  .filterDate(clear_date, cloudy_date)
                  .filterBounds(AOI)
                  
var acc = rainfall.reduce(ee.Reducer.sum());
// Calculate accumulative averaged rainfall in the AOI
var total = acc.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: AOI,
  scale: 5000,
  })
var rainfall_acc = total.get('precipitation_sum') // the unit is mm

//---Get averaged Sentinel-2 cloud probability within AOI of the cloudy date---//
var s2CloudProb = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filter(COL_FILTER).filterDate(cloudy_date.getRange('day'));
var cloudprob_aoi = s2CloudProb.first().select('probability').reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: AOI,
  scale: 5000,
}).get('probability');

//----------------------Export table and images------------------------------//
// Create an ee.Feature to store all the properties that I want to export
// Export images: dwImage, clear_s2, cloudy_s2, clear_s1, cloudy_s1
// Export parameters point, AOI, YEAR, max_percent, clear_date, cloudy_date, clear_s1_date, cloudy_s1_date

var label_projection = dwImage.select('label').projection().getInfo();
var system_index = dwImage.get('system:index').getInfo()

var dict = {
  centerPoint: point.coordinates(),
  aoi: AOI.coordinates(),
  year: YEAR,
  dwIndex: system_index,
  labelCoverPercent: max_percent,
  dwDate: clear_date,
  clearS2Date: clear_date,
  cloudyS2Date: cloudy_date,
  clearS1Date: clear_s1_date,
  cloudyS1Date: cloudy_s1_date, 
  labelPrj: label_projection,
  accRainfall: rainfall_acc,
  cloudProb: cloudprob_aoi
}

var para_feature_col = ee.FeatureCollection([ee.Feature(AOI, dict)]);

var para_filename = ee.String('metainfo_').cat(system_index)

// Export the FeatureCollection.
Export.table.toDrive({
  description: "metaInfo_"+system_index, 
  collection: para_feature_col,
  folder: "MLFluvData_DWS12_"+system_index,
  fileFormat: 'CSV'
});

Export.image.toDrive({
  image: dwImage.select('label'),
  description: 'DWlabel_'+system_index,
  crs: label_projection.crs,
  crsTransform: label_projection.transform,
  region: AOI,
  folder: "MLFluvData_DWS12_"+system_index,
});

var projection = clear_s2.select('B2').projection().getInfo();

Export.image.toDrive({
  image: clear_s1,
  description: 'clearS1_'+system_index,
  crs: projection.crs,
  crsTransform: projection.transform,
  region: AOI,
  folder: "MLFluvData_DWS12_"+system_index,
  scale:10
});

Export.image.toDrive({
  image: clear_s2,
  description: 'clearS2_'+system_index,
  crs: projection.crs,
  crsTransform: projection.transform,
  region: AOI,
  folder: "MLFluvData_DWS12_"+system_index,
  scale:10
});

Export.image.toDrive({
  image: cloudy_s1,
  description: 'cloudyS1_'+system_index,
  crs: projection.crs,
  crsTransform: projection.transform,
  region: AOI,
  folder: "MLFluvData_DWS12_"+system_index,
  scale:10
});

Export.image.toDrive({
  image: cloudy_s2,
  description: 'cloudyS2_'+system_index,
  crs: projection.crs,
  crsTransform: projection.transform,
  region: AOI,
  folder: "MLFluvData_DWS12_"+system_index,
  scale:10
});

//----------------------------- FUNCTIONS -----------------------------------//
function addVariables(image){
  //method from https://zhuanlan.zhihu.com/p/136929576
  var mndwi = image.normalizedDifference(['B3','B11']).rename('MNDWI');
  var ndwi = image.normalizedDifference(['B3','B8']).rename('NDWI');
  var ndbi = image.normalizedDifference(['B11','B8']).rename('NDBI');
  var ndvi = image.normalizedDifference(['B8','B4']).rename('NDVI');
  return image.addBands([mndwi,ndwi,ndbi,ndvi]);
}

function getS1Preprocess(startDate, endDate, geometry) {
  // Get two DEM options for terrain flattening 
  // var dem_srtm = ee.Image('USGS/SRTMGL1_003').clip(geometry);
  var dem_cop = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').filterBounds(geometry).mosaic();
  // // Sentinel-1 preprocessing code is from https://github.com/adugnag/gee_s1_ard
  var wrapper = require('users/adugnagirma/gee_s1_ard:wrapper');
  var helper = require('users/adugnagirma/gee_s1_ard:utilities');
    // // DEFINE PARAMETERS
  var parameter = {//1. Data Selection
                START_DATE: startDate,
                STOP_DATE: endDate,
                POLARIZATION:'VVVH',
                ORBIT : 'DESCENDING',
                GEOMETRY: geometry, 
                //2. Additional Border noise correction
                APPLY_ADDITIONAL_BORDER_NOISE_CORRECTION: true,
                //3.Speckle filter
                APPLY_SPECKLE_FILTERING: true,
                SPECKLE_FILTER_FRAMEWORK: 'MULTI',
                SPECKLE_FILTER: 'LEE',
                SPECKLE_FILTER_KERNEL_SIZE: 9,
                SPECKLE_FILTER_NR_OF_IMAGES: 10,
                //4. Radiometric terrain normalization
                APPLY_TERRAIN_FLATTENING: true,
                DEM: dem_cop, //dem_srtm,
                TERRAIN_FLATTENING_MODEL: 'VOLUME',
                TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER: 0,
                //5. Output
                FORMAT : 'DB',
                CLIP_TO_ROI: false,
                SAVE_ASSETS: false
  };
  // Preprocess the S1 collection
  var s1_preprocessed = wrapper.s1_preproc(parameter);
  // Unprocessed image collection is stored at index 0, preprocessed image collection stored at index 1.
  var s1 = s1_preprocessed[0];
  s1_preprocessed = s1_preprocessed[1];
  
  return s1_preprocessed;
}

function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000)
          .select("B.*")
          .copyProperties(image,["system:time_start"]);
}

function normalizeImage(image){
  //method from: https://gis.stackexchange.com/questions/313394/normalization-in-google-earth-engine
  // calculate the min and max value of an image
  var minMax = image.reduceRegion({
    reducer: ee.Reducer.minMax(),
    geometry: image.geometry(),
    scale: 30,
    maxPixels: 10e9,
    tileScale: 2
  }); 
  // use unit scale to normalize the pixel values
  var unitScale = ee.ImageCollection.fromImages(
    image.bandNames().map(function(name){
      name = ee.String(name);
      var band = image.select(name);
      return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))))
                  // eventually multiply by 100 to get range 0-100
                  //.multiply(100);
  })).toBands().rename(image.bandNames());
  
  print(unitScale)
  return unitScale
}