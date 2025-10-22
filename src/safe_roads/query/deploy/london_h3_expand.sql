-- (Optional) ensure h3 is the native type
ALTER TABLE london_h3
  ALTER COLUMN h3 TYPE h3index USING h3::h3index;

-- 1) Add neighbor columns + parent column (do nothing if they already exist)
ALTER TABLE london_h3
  ADD COLUMN IF NOT EXISTS neighbor_1 h3index,
  ADD COLUMN IF NOT EXISTS neighbor_2 h3index,
  ADD COLUMN IF NOT EXISTS neighbor_3 h3index,
  ADD COLUMN IF NOT EXISTS neighbor_4 h3index,
  ADD COLUMN IF NOT EXISTS neighbor_5 h3index,
  ADD COLUMN IF NOT EXISTS neighbor_6 h3index,
  ADD COLUMN IF NOT EXISTS parent_h3 h3index;

-- 2) Compute neighbors and update the table
WITH nbr_ranked AS (
  SELECT
    t.h3 AS center,
    n    AS nbr,
    ROW_NUMBER() OVER (PARTITION BY t.h3 ORDER BY n) AS pos
  FROM london_h3 AS t
  CROSS JOIN LATERAL h3_grid_disk(t.h3, 1) AS n
  WHERE t.h3 IS NOT NULL AND n <> t.h3
),
nbr_array AS (
  SELECT
    center,
    array_agg(nbr ORDER BY pos) AS nbrs
  FROM nbr_ranked
  GROUP BY center
)
UPDATE london_h3 AS t
SET
  neighbor_1 = na.nbrs[1],
  neighbor_2 = na.nbrs[2],
  neighbor_3 = na.nbrs[3],
  neighbor_4 = na.nbrs[4],
  neighbor_5 = na.nbrs[5],
  neighbor_6 = na.nbrs[6]
FROM nbr_array AS na
WHERE na.center = t.h3;

-- 2b) Populate parent at RES 10
UPDATE london_h3
SET parent_h3 =
  CASE
    WHEN h3 IS NULL THEN NULL
    WHEN h3_get_resolution(h3) = 10 THEN h3
    WHEN h3_get_resolution(h3) > 10 THEN h3_cell_to_parent(h3, 10)
    ELSE NULL  -- if cell is coarser than 10, no parent at a finer res
  END;

-- 3) Indexes (optional but useful)
CREATE INDEX IF NOT EXISTS london_h3_neighbor_1_idx ON london_h3 (neighbor_1);
CREATE INDEX IF NOT EXISTS london_h3_neighbor_2_idx ON london_h3 (neighbor_2);
CREATE INDEX IF NOT EXISTS london_h3_neighbor_3_idx ON london_h3 (neighbor_3);
CREATE INDEX IF NOT EXISTS london_h3_neighbor_4_idx ON london_h3 (neighbor_4);
CREATE INDEX IF NOT EXISTS london_h3_neighbor_5_idx ON london_h3 (neighbor_5);
CREATE INDEX IF NOT EXISTS london_h3_neighbor_6_idx ON london_h3 (neighbor_6);
CREATE INDEX IF NOT EXISTS london_h3_parent_h3_idx ON london_h3 (parent_h3);
