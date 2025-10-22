DROP TABLE IF EXISTS mv_idx;

CREATE UNLOGGED TABLE mv_idx AS
SELECT m.*,
       row_number() OVER (ORDER BY random())::int AS idx
FROM random_month_datetime_samples m;

CREATE INDEX ON mv_idx (idx);


DROP TABLE IF EXISTS london_h3_datetime;

CREATE UNLOGGED TABLE london_h3_datetime AS
WITH
params AS (
  SELECT 1::int AS min_per_cell, 3::int AS max_per_cell   -- Change min and max per_cell 
),
mv_size AS (
  SELECT count(*)::int AS m FROM mv_idx
),
per_cell AS (
  SELECT
    l.h3,
    (p.min_per_cell + floor(random() * (p.max_per_cell - p.min_per_cell + 1)))::int AS k,
    1 + (abs(mod(('x' || substr(md5(l.h3::text),1,8))::bit(32)::int, s.m - 1)))::int AS start_idx,
    s.m
  FROM london_h3_filtered l
  CROSS JOIN params p
  CROSS JOIN mv_size s
),
choices AS (
  SELECT
    h3,
    ((start_idx - 1 + offs) % m) + 1 AS idx
  FROM per_cell
  JOIN LATERAL generate_series(0, greatest(k,0) - 1) AS offs ON true
)
SELECT
  l.*,
  m.random_timestamp AS timestamp
FROM choices c
JOIN mv_idx m USING (idx)
JOIN london_h3_filtered l USING (h3)
ORDER BY l.h3, m.random_timestamp;

ALTER TABLE london_h3_datetime SET LOGGED;