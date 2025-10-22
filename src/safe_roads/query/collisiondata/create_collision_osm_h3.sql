DROP TABLE IF EXISTS osm_collision_h3;

CREATE TABLE osm_collision_h3 AS
WITH c AS (
  SELECT
    cdh3.*,
    EXTRACT(YEAR FROM (cdh3.event_dt AT TIME ZONE 'UTC'))::int AS event_year
  FROM public.collisiondata_h3 cdh3
),
o AS (
  SELECT
    o_raw.*,
    CASE
      WHEN o_raw.year ~ '\d{4}'
        THEN ((regexp_match(o_raw.year, '(\d{4})'))[1])::int
      ELSE NULL
    END AS event_year
  FROM public.osm_h3 o_raw
)
SELECT *
FROM c
FULL JOIN o USING (h3, event_year);
