-- Rebuild table with base h3 at res 12 (adjust if you prefer a different base)
DROP TABLE IF EXISTS collisiondata_h3;

CREATE TABLE collisiondata_h3 AS
WITH wgs AS (
  SELECT
    ctid,
    ST_Transform(
      ST_SetSRID(
        ST_MakePoint(
          NULLIF(regexp_replace("Easting",  '[^0-9\.\-]', '', 'g'), '')::double precision,
          NULLIF(regexp_replace("Northing", '[^0-9\.\-]', '', 'g'), '')::double precision
        ),
        27700
      ),
      4326
    ) AS wgsgeom
  FROM collisiondata
)
SELECT
  c.*,
  CASE
    WHEN w.wgsgeom IS NOT NULL THEN h3_lat_lng_to_cell(w.wgsgeom, 12)::h3index
    ELSE NULL
  END AS h3
FROM collisiondata AS c
LEFT JOIN wgs AS w
  ON c.ctid = w.ctid;

-- Ensure native H3 type (no-op if already h3index)
ALTER TABLE collisiondata_h3
  ALTER COLUMN h3 TYPE h3index USING h3::h3index;

-- Add parent columns at specific resolutions
ALTER TABLE collisiondata_h3
  ADD COLUMN IF NOT EXISTS res_11 h3index,
  ADD COLUMN IF NOT EXISTS res_10 h3index,
  ADD COLUMN IF NOT EXISTS res_9  h3index;

-- Populate parent columns (robust to mixed input resolutions)
UPDATE collisiondata_h3
SET
  res_11 = CASE
              WHEN h3 IS NULL THEN NULL
              WHEN h3_get_resolution(h3) = 11 THEN h3
              WHEN h3_get_resolution(h3) > 11 THEN h3_cell_to_parent(h3, 11)
              ELSE NULL   -- input is coarser than 11
           END,
  res_10 = CASE
              WHEN h3 IS NULL THEN NULL
              WHEN h3_get_resolution(h3) = 10 THEN h3
              WHEN h3_get_resolution(h3) > 10 THEN h3_cell_to_parent(h3, 10)
              ELSE NULL   -- input is coarser than 10
           END,
  res_9  = CASE
              WHEN h3 IS NULL THEN NULL
              WHEN h3_get_resolution(h3) = 9  THEN h3
              WHEN h3_get_resolution(h3) > 9  THEN h3_cell_to_parent(h3, 9)
              ELSE NULL   -- input is coarser than 9
           END;

-- Useful indexes
CREATE INDEX IF NOT EXISTS collisiondata_h3_h3_idx     ON collisiondata_h3 (h3);
CREATE INDEX IF NOT EXISTS collisiondata_h3_res_11_idx ON collisiondata_h3 (res_11);
CREATE INDEX IF NOT EXISTS collisiondata_h3_res_10_idx ON collisiondata_h3 (res_10);
CREATE INDEX IF NOT EXISTS collisiondata_h3_res_9_idx  ON collisiondata_h3 (res_9);
