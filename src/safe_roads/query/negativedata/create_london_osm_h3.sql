DROP TABLE IF EXISTS public.london_osm_h3 CASCADE;

-- ensure london_h3.h3 is typed
ALTER TABLE public.london_h3
  ALTER COLUMN h3 TYPE h3index USING h3::h3index;

-- CREATE TABLE ... AS WITH ...
CREATE TABLE public.london_osm_h3 AS
WITH osmx_one AS (
  -- one representative osmx row per h3; keep h3 once + jsonb payload
  SELECT DISTINCT ON (x.h3)
         x.h3::h3index AS h3,
         to_jsonb(x)   AS j
  FROM public.osmx_h3 x
  ORDER BY x.h3
)
SELECT
  l.*,
  -- project osmx_h3 into osm_h3 schema but keep as JSONB to avoid dup column names
  to_jsonb(
    jsonb_populate_record(
      NULL::public.osm_h3,
      o.j - 'h3' - 'year' - 'month'
    )
  ) AS osmx_projected
FROM public.london_h3 AS l
JOIN osmx_one AS o USING (h3);
