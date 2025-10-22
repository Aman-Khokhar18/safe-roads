DROP TABLE IF EXISTS collisiondata_h3;
CREATE TABLE collisiondata_h3 AS
WITH wgs AS (
  SELECT
    ctid,
    -- Build a WGS84 geometry point from Easting/Northing (robust to text input)
    ST_Transform(
      ST_SetSRID(
        ST_MakePoint(
          NULLIF(regexp_replace("Easting",  '[^0-10\.\-]', '', 'g'), '')::double precision,
          NULLIF(regexp_replace("Northing", '[^0-10\.\-]', '', 'g'), '')::double precision
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
    WHEN w.wgsgeom IS NOT NULL THEN
      -- Use the geometry overload you have: h3_lat_lng_to_cell(geometry, resolution)
      h3_lat_lng_to_cell(w.wgsgeom, 12)::h3index   -- <-- change 11 if you want a different working res
    ELSE NULL
  END AS h3
FROM collisiondata AS c
LEFT JOIN wgs AS w
  ON c.ctid = w.ctid;

-- 2) Ensure native H3 type (harmless if already h3index)
ALTER TABLE collisiondata_h3
  ALTER COLUMN h3 TYPE h3index USING h3::h3index;

-- 3) Add columns (parent named exactly parent_h3)
ALTER TABLE collisiondata_h3
  ADD COLUMN IF NOT EXISTS parent_h3 h3index,
  ADD COLUMN IF NOT EXISTS neighbor_1 h3index,
  ADD COLUMN IF NOT EXISTS neighbor_2 h3index,
  ADD COLUMN IF NOT EXISTS neighbor_3 h3index,
  ADD COLUMN IF NOT EXISTS neighbor_4 h3index,
  ADD COLUMN IF NOT EXISTS neighbor_5 h3index,
  ADD COLUMN IF NOT EXISTS neighbor_6 h3index;

-- 4) Populate parent_h3 at RES 10 (uses h3_cell_to_parent available in your build)
UPDATE collisiondata_h3
SET parent_h3 =
  CASE
    WHEN h3 IS NULL THEN NULL
    WHEN h3_get_resolution(h3) = 10 THEN h3
    WHEN h3_get_resolution(h3) > 10 THEN h3_cell_to_parent(h3, 10)
    ELSE NULL
  END;

-- 5) Compute 1-ring neighbors (6 cells) and update columns
WITH nbr_ranked AS (
  SELECT
    t.h3 AS center,
    n    AS nbr,
    ROW_NUMBER() OVER (PARTITION BY t.h3 ORDER BY n) AS pos
  FROM collisiondata_h3 AS t
  CROSS JOIN LATERAL h3_grid_disk(t.h3, 1) AS n
  WHERE t.h3 IS NOT NULL AND n <> t.h3
),
nbr_array AS (
  SELECT center, array_agg(nbr ORDER BY pos) AS nbrs
  FROM nbr_ranked
  GROUP BY center
)
UPDATE collisiondata_h3 AS t
SET neighbor_1 = na.nbrs[1],
    neighbor_2 = na.nbrs[2],
    neighbor_3 = na.nbrs[3],
    neighbor_4 = na.nbrs[4],
    neighbor_5 = na.nbrs[5],
    neighbor_6 = na.nbrs[6]
FROM nbr_array AS na
WHERE na.center = t.h3;

-- 6) Indexes
CREATE INDEX IF NOT EXISTS collisiondata_h3_h3_idx         ON collisiondata_h3 (h3);
CREATE INDEX IF NOT EXISTS collisiondata_h3_parent_h3_idx  ON collisiondata_h3 (parent_h3);
CREATE INDEX IF NOT EXISTS collisiondata_h3_neighbor_1_idx ON collisiondata_h3 (neighbor_1);
CREATE INDEX IF NOT EXISTS collisiondata_h3_neighbor_2_idx ON collisiondata_h3 (neighbor_2);
CREATE INDEX IF NOT EXISTS collisiondata_h3_neighbor_3_idx ON collisiondata_h3 (neighbor_3);
CREATE INDEX IF NOT EXISTS collisiondata_h3_neighbor_4_idx ON collisiondata_h3 (neighbor_4);
CREATE INDEX IF NOT EXISTS collisiondata_h3_neighbor_5_idx ON collisiondata_h3 (neighbor_5);
CREATE INDEX IF NOT EXISTS collisiondata_h3_neighbor_6_idx ON collisiondata_h3 (neighbor_6);
