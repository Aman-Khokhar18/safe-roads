
-- 1) Fresh table with all OSM_data columns + h3
DROP TABLE IF EXISTS osm_data_live_h3cells;
CREATE UNLOGGED TABLE osm_data_live_h3cells AS
SELECT o.*, NULL::h3index AS h3
FROM osm_data_live o
WHERE FALSE;  -- structure only

-- 2a) LINES → corridor buffer → H3 cells (r=12)
INSERT INTO osm_data_live_h3cells
SELECT DISTINCT o.*, h3c.h3
FROM osm_data_live o
CROSS JOIN (
  SELECT 12::int AS r,
         (h3_get_hexagon_edge_length_avg(12,'m')/100)::float AS buf_m
) p
JOIN LATERAL (
  SELECT ST_Transform(o.geometry, 4326)::geography AS geog_4326
) t ON TRUE
JOIN LATERAL h3_polygon_to_cells_experimental(
  ST_Buffer(t.geog_4326, p.buf_m)::geometry,  -- corridor
  p.r,
  'overlapping'
) AS h3c(h3) ON TRUE
WHERE o.geometry IS NOT NULL
  AND ST_GeometryType(o.geometry) IN ('ST_LineString','ST_MultiLineString');

-- 2b) POINTS / MULTIPOINTS → dump → H3 cell (r=12)
INSERT INTO osm_data_live_h3cells
SELECT DISTINCT o.*, h3_lat_lng_to_cell(dp.pt_4326, p.r) AS h3
FROM osm_data_live o
CROSS JOIN (SELECT 12::int AS r) p
JOIN LATERAL (
  SELECT (d.geom)::geometry(Point, 4326) AS pt_4326
  FROM ST_DumpPoints(ST_Transform(o.geometry, 4326)) AS d
) dp ON TRUE
WHERE o.geometry IS NOT NULL
  AND ST_GeometryType(o.geometry) IN ('ST_Point','ST_MultiPoint');

ALTER TABLE osm_data_live_h3cells SET LOGGED;

DROP TABLE IF EXISTS osm_data_live_h3;
CREATE TABLE osm_data_live_h3 AS
SELECT
  o.*,
  l.district,
  l.parent_h3,
  l.neighbor_1,
  l.neighbor_2,
  l.neighbor_3,
  l.neighbor_4,
  l.neighbor_5,
  l.neighbor_6
  
FROM osm_data_live_h3cells AS o
JOIN london_h3 AS l
  ON o.h3 = l.h3;
