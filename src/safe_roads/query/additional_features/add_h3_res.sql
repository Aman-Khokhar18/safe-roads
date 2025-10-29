-- Ensure native H3 type on the existing h3 column (no-op if already h3index)
ALTER TABLE combined_dataset
  ALTER COLUMN h3 TYPE h3index USING h3::h3index;

-- Add parent columns
ALTER TABLE combined_dataset
  ADD COLUMN IF NOT EXISTS res_11 h3index,
  ADD COLUMN IF NOT EXISTS res_10 h3index,
  ADD COLUMN IF NOT EXISTS res_9  h3index;

-- Populate parent columns (robust to mixed input resolutions)
UPDATE combined_dataset
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
CREATE INDEX IF NOT EXISTS road_features_collision_res_11_idx ON combined_dataset (res_11);
CREATE INDEX IF NOT EXISTS road_features_collision_res_10_idx ON combined_dataset (res_10);
CREATE INDEX IF NOT EXISTS road_features_collision_res_9_idx  ON combined_dataset (res_9);
