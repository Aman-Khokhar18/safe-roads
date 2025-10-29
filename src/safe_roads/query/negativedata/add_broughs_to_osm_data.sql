DROP TABLE IF EXISTS london_osm_data;
CREATE TABLE london_osm_data AS
SELECT
  o.*,
  l."Borough" AS borough,
  l.res_11
FROM osm_h3_ml_features AS o
LEFT JOIN london_h3 AS l
  ON l.h3::h3index = o.h3::h3index; 