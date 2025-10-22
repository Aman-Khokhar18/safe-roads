DROP TABLE IF EXISTS osm_data_live_h3;

CREATE TABLE osm_data_live_h3 AS
SELECT
  o.*,                 
  hc.h3  AS h3_cell,   
  lh3.*                 
FROM osm_data_live         AS o
JOIN osm_data_live_h3cells AS hc
  ON hc.osmx_pk = o.osmx_pk
JOIN london_h3    AS lh3
  ON lh3.h3 = hc.h3;
