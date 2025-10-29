DROP TABLE IF EXISTS public.osm_collision_h3;

CREATE TABLE public.osm_collision_h3 AS
WITH c AS (
  SELECT
    cdh3.*,
    EXTRACT(YEAR FROM (cdh3.event_dt AT TIME ZONE 'UTC'))::int AS event_year
  FROM public.collisiondata_h3 cdh3
),
f AS (
  SELECT
    f_raw.*,
    CASE
      WHEN coalesce(f_raw.year::text,'') ~ '^\s*\d{4}\s*$'
        THEN trim(f_raw.year::text)::int
      WHEN coalesce(f_raw.year::text,'') ~ '\d{4}'
        THEN ((regexp_match(f_raw.year::text, '(\d{4})'))[1])::int
      ELSE NULL
    END AS event_year
  FROM public.osm_h3_ml_features f_raw
)
SELECT *
FROM c
LEFT JOIN f USING (h3, event_year);

-- (optional) index to speed lookups/aggregations
CREATE INDEX IF NOT EXISTS idx_osm_collision_h3_h3_year
  ON public.osm_collision_h3 (h3, event_year);
