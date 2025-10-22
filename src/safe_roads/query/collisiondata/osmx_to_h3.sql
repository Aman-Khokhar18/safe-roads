-- 0) Ensure source has a stable PK
ALTER TABLE osmx_data
  ADD COLUMN IF NOT EXISTS osmx_pk BIGSERIAL PRIMARY KEY;

-- 1) Fresh table with all OSM_data columns + h3
DROP TABLE IF EXISTS osmx_h3 CASCADE;
CREATE UNLOGGED TABLE osmx_h3 AS
SELECT o.*, NULL::h3index AS h3
FROM osmx_data o
WHERE FALSE;  -- structure only

-- 2a) LINES → corridor buffer → H3 cells (r=12)
INSERT INTO osmx_h3
SELECT DISTINCT o.*, h3c.h3
FROM osmx_data o
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
INSERT INTO osmx_h3
SELECT DISTINCT o.*, h3_lat_lng_to_cell(dp.pt_4326, p.r) AS h3
FROM osmx_data o
CROSS JOIN (SELECT 12::int AS r) p
JOIN LATERAL (
  SELECT (d.geom)::geometry(Point, 4326) AS pt_4326
  FROM ST_DumpPoints(ST_Transform(o.geometry, 4326)) AS d
) dp ON TRUE
WHERE o.geometry IS NOT NULL
  AND ST_GeometryType(o.geometry) IN ('ST_Point','ST_MultiPoint');

ALTER TABLE osmx_h3 SET LOGGED;

ALTER TABLE osmx_h3
  ALTER COLUMN h3 TYPE h3index USING h3::h3index;

ALTER TABLE osmx_h3
  ADD PRIMARY KEY (osmx_pk, h3);

CREATE INDEX IF NOT EXISTS osmx_h3_h3_idx  ON osmx_h3 (h3);
CREATE INDEX IF NOT EXISTS osmx_h3_osm_idx ON osmx_h3 (osmx_pk);

