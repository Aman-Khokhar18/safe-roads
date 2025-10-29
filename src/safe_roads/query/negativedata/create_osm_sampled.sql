DROP TABLE IF EXISTS london_osm_sample;

CREATE TABLE london_osm_sample AS
WITH s AS (
  -- 10% block sample; keep rows for years 2015..2024; compute a per-row hash key
  SELECT
    *,
    ("year")::int AS year_i,
    (('x'||substr(md5(ctid::text),1,16))::bit(64)::bigint) AS row_key
  FROM london_osm_data
  TABLESAMPLE SYSTEM (20)
  WHERE ("year")::int BETWEEN 2015 AND 2024
),
per_h3y AS (
  -- For each (h3, year), pick a random K rows to pair; precompute shared ts & pair_id
  SELECT
    h3,
    year_i,
    COUNT(*)::int AS cnt,
    floor(random() * (COUNT(*) + 1))::int AS k,  -- K in [0..cnt]
    /* shared hour within that year */
    make_timestamp(year_i,1,1,0,0,0) +
      (
        trunc(
          random() * (
            extract(epoch from (make_timestamp(year_i+1,1,1,0,0,0)
                                - make_timestamp(year_i,1,1,0,0,0))) / 3600
          )
        ) * interval '1 hour'
      ) AS ts_h3y,
    /* same pair_id for all paired rows of this (h3,year) */
    (('x'||substr(md5(h3::text || ':' || year_i::text),1,16))::bit(64)::bigint) AS pair_id_h3y
  FROM s
  GROUP BY h3, year_i
)
SELECT
  s.*,
  -- Paired subset gets shared ts; others get their own ts within that row's year
  CASE
    WHEN (s.row_key % per_h3y.cnt) < per_h3y.k
      THEN per_h3y.ts_h3y
      ELSE
        make_timestamp(s.year_i,1,1,0,0,0) +
        (
          trunc(
            random() * (
              extract(epoch from (make_timestamp(s.year_i+1,1,1,0,0,0)
                                  - make_timestamp(s.year_i,1,1,0,0,0))) / 3600
            )
          ) * interval '1 hour'
        )
  END AS ts,
  CASE
    WHEN (s.row_key % per_h3y.cnt) < per_h3y.k
      THEN per_h3y.pair_id_h3y          -- same id for the paired subset of (h3,year)
      ELSE s.row_key                    -- unique per-row otherwise
  END AS pair_id,
  ((s.row_key % per_h3y.cnt) < per_h3y.k) AS is_paired_row
FROM s
JOIN per_h3y USING (h3, year_i);

ALTER TABLE london_osm_sample DROP COLUMN row_key;


