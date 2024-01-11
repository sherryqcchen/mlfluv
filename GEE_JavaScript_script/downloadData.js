// var sampler = require('users/qiuyangschen/MLFluv:pointSampler');
// var batch = require('users/fitoprincipe/geetools:batch')

// var YEAR = 2019;

// // Read the points from imported csv file, and convert them from feature collection to ee.Geometry object
// var points = ee.FeatureCollection('projects/mlfluvseg/assets/network_da_order_sample');
// // // print(points)
// // print(points.geometry().coordinates().get(0));
// var ml_data_point = sampler.sampleFromPoint(points.geometry().coordinates().get(0), YEAR);
// print('What is inside of a point data:', ml_data_point)
// print(ml_data_point.get('aoi'));
// print(ml_data_point.get('dwIndex'))
// print(ml_data_point.toList(5).get(0))

// var sub_points = ee.FeatureCollection(points.toList(1))
// print(sub_points)

// var ml_data = sub_points.map(function (point) {
//   var coords = point.geometry().coordinates();

//   var train_col = sampler.sampleFromPoint(coords, YEAR);
  
//   var system_index = train_col.get('dwIndex');
  
//   var label_projection = dwImage.select('label').projection();
//   var projection = clear_s2.select('B2').projection();
  
//   var para_filename = ee.String('metainfo_').cat(system_index);
//   var folder_name = "MLFluvData_DWS12_" + system_index
  
//   var dwLabel = train_col.toList(5).get(0);
//   var clearS1 = train_col.toList(5).get(1);
//   var clearS2 = train_col.toList(5).get(2);
//   var cloudyS1 = train_col.toList(5).get(3);
//   var cloudyS2 = train_col.toList(5).get(4);
  
  
//   // TODO: CHECK these possible resources: 
//   // https://gis.stackexchange.com/questions/445926/exporting-within-a-mapped-function-in-gee
//   // https://gis.stackexchange.com/questions/248216/exporting-each-image-from-collection-in-google-earth-engine#comment390829_248230
//   return train_col;  
// });
// print('What is inside of a collection:', ml_data)


// ----------------Try a more manual way---------------
var sampler = require('users/qiuyangschen/MLFluv:pointSamplerBackup');

var YEAR = 2019;

// Read the points from imported csv file, and convert them from feature collection to ee.Geometry object
var points = ee.FeatureCollection('projects/mlfluvseg/assets/network_da_order_sample');
// // print(points)
// print(points.geometry().coordinates().get(0));
// var ml_data_point = sampler.sampleFromPoint(points.geometry().coordinates().get(999), YEAR);

for (var i=798; i<799; i++) {
  print(i);
  sampler.sampleFromPoint(points.geometry().coordinates().get(i), YEAR);
}