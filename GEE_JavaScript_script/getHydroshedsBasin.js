// Author: Qiuyang Chen

// This file is used to obtain the river basin/sub-basin outline features of Amazon river from Google Earth Engine.
// Before running this script, draw a polygon within the amazon basin using any shape, name it "AOI" on the Map of GEE code editor.

// Import HydrSHEDS river basins 
var hv2 = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_2");
var hv4 = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_4");
var hv6 = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_6");

//Get the outline of a whole Amazon basin from HydroSHEDS
var amazon_v2 = hv2.filterBounds(AOI)
Map.addLayer(amazon_v2, {}, 'HydroSHEDS Basins level 2')
print(amazon_v2.first())

// create a buffer with slightly contracted edges (100 meters) from the outline of the amazon basin.
// In this way we do not get the neighbour basins that are connected to Amazon basin.
var buffer = amazon_v2.map(function(polygon){
  return polygon.buffer({'distance': -100})
});
  
print(buffer);
Map.addLayer(buffer, {}, 'buffered HydroSHEDS Basins level 2');

var amazon_v4 = hv4.filterBounds(buffer);
Map.addLayer(amazon_v4, {}, 'HydroSHEDS Basins level 4');

Export.table.toDrive(amazon_v4, 'Amazon_basin_v4', 'river_basin');

var amazon_v6 = hv6.filterBounds(buffer);
Map.addLayer(amazon_v6, {}, 'HydroSHEDS Basins level 6');

Export.table.toDrive(amazon_v6, 'Amazon_basin_v6', 'river_basin');


// Get the river networks from HydroSHEDS 
var river = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers");

var amazon_river = river.filterBounds(buffer).filter(
  ee.Filter.or(
    ee.Filter.eq('RIV_ORD',1), 
    ee.Filter.eq('RIV_ORD',2), 
    ee.Filter.eq('RIV_ORD',3), 
    ee.Filter.eq('RIV_ORD',4), 
    ee.Filter.eq('RIV_ORD',5),
    ee.Filter.eq('RIV_ORD',6)
  ))
Map.addLayer(amazon_river, {color: "B2B2B3",width: 1.0},"Filtered Amazon streams order 1-6");

Map.addLayer(amazon_river.filter(ee.Filter.eq('RIV_ORD',1)),{color:'00ffff',width:20},'river order = 1');
Map.addLayer(amazon_river.filter(ee.Filter.eq('RIV_ORD',2)),{color:'ccf9ff',width:15},'river order = 2');
Map.addLayer(amazon_river.filter(ee.Filter.eq('RIV_ORD',3)),{color:'7ce8ff',width:10},'river order = 3');
Map.addLayer(amazon_river.filter(ee.Filter.eq('RIV_ORD',4)),{color:'55d0ff',width:5},'river order = 4');
Map.addLayer(amazon_river.filter(ee.Filter.eq('RIV_ORD',5)),{color:'7ce8ff',width:3},'river order = 5');
Map.addLayer(amazon_river.filter(ee.Filter.eq('RIV_ORD',6)),{color:'ccf9ff',width:1},'river order = 6');

Export.table.toDrive(amazon_river, 'Amazon_river_network', 'river_network');


// Download the river basin and network shapefiles and store them in the folders under the main directory: Amazon_HydroSHEDS_river_basin/, Amazon_HydroSHEDS_river_networks/