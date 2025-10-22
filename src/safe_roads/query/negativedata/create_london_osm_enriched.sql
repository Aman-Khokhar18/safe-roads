-- ======================================================================
-- Build (no DISTINCT on combo_id; drop point-only rows)
-- ======================================================================
DROP TABLE IF EXISTS public.london_osm_daily_enriched;

CREATE TABLE public.london_osm_daily_enriched AS
WITH base AS (
  SELECT
    t.*,
    t.ctid AS _ctid,
    CASE
      WHEN lower(t.highway) IN (
        'motorway','motorway_link','trunk','trunk_link','motorway_junction',
        'primary','primary_link','secondary','secondary_link','tertiary','tertiary_link',
        'residential','unclassified','living_street','busway','track',
        'mini_roundabout','turning_circle','turning circle'
      ) THEN 'motorway'
      WHEN lower(t.highway) = 'cycleway' THEN 'cycleway'
      WHEN lower(t.highway) IN (
        'pedestrian','path','steps','bridleway','platform','corridor',
        'crossing','crossing;traffic_signals','elevator','traffic_island'
      ) THEN 'footway'
      WHEN t.highway IS NULL THEN 'unknown'
      ELSE 'other'
    END AS road_class3
  FROM public.london_osm_daily AS t
  WHERE t.geometry IS NOT NULL
    AND ST_Dimension(t.geometry) >= 1
),
junction_cell AS (
  SELECT
    j.h3,
    MAX(j.deg)             AS junction_degree,
    BOOL_OR(j.is_junction) AS is_junction,
    BOOL_OR(j.is_turn)     AS is_turn
  FROM public.osm_junctions_yearly j
  GROUP BY j.h3
)
SELECT
  b.*,
  -- add only the *new* parts
  EXTRACT(DAY    FROM b."timestamp")::int    AS day,
  EXTRACT(HOUR   FROM b."timestamp")::int    AS hour,
  EXTRACT(MINUTE FROM b."timestamp")::int    AS minute,
  CASE WHEN EXTRACT(DOW FROM b."timestamp") IN (0,6) THEN 1 ELSE 0 END::int AS is_weekend,
  -- rest
  b.district AS borough,
  CASE
    WHEN b.maxspeed ~* 'km/?h' AND b.maxspeed ~* '(\d+(\.\d+)?)'
      THEN ROUND(((regexp_match(b.maxspeed, '(\d+(\.\d+)?)'))[1])::numeric * 0.621371)::int
    WHEN b.maxspeed ~* '(\d+(\.\d+)?)'
      THEN ROUND(((regexp_match(b.maxspeed, '(\d+(\.\d+)?)'))[1])::numeric)::int
    ELSE NULL
  END AS maxspeed_mph,
  cc.traffic_lights,
  cc.crossings,
  cc.cycleways,
  GREATEST(
    cc.motorways_total - CASE WHEN b.road_class3 = 'motorway' THEN 1 ELSE 0 END,
    0
  ) AS motorways_other,
  COALESCE(jc.is_junction, FALSE) AS is_junction,
  COALESCE(jc.is_turn,     FALSE) AS is_turn,
  COALESCE(jc.junction_degree, 0) AS junction_degree
FROM base b
LEFT JOIN public.mv_ctx_counts cc ON cc.h3 = b.h3
LEFT JOIN junction_cell jc ON jc.h3 = b.h3;

-- Coerce existing year/month to INT (fallback to timestamp when dirty)
ALTER TABLE public.london_osm_daily_enriched
  ALTER COLUMN year  TYPE int USING (
    CASE
      WHEN year  ~ '^\s*\d{1,4}\s*$' THEN trim(year)::int
      ELSE EXTRACT(YEAR  FROM "timestamp")::int
    END
  ),
  ALTER COLUMN month TYPE int USING (
    CASE
      WHEN month ~ '^\s*\d{1,2}\s*$' THEN trim(month)::int
      ELSE EXTRACT(MONTH FROM "timestamp")::int
    END
  );


-- Indexes
CREATE INDEX IF NOT EXISTS idx_london_osm_daily_enriched_h3
  ON public.london_osm_daily_enriched (h3);

CREATE INDEX IF NOT EXISTS idx_london_osm_daily_enriched_roadclass
  ON public.london_osm_daily_enriched (road_class3);

CREATE INDEX IF NOT EXISTS idx_london_osm_daily_enriched_borough
  ON public.london_osm_daily_enriched (borough);

CREATE INDEX IF NOT EXISTS idx_london_osm_daily_enriched_year_month_day
  ON public.london_osm_daily_enriched (year, month, day);

CREATE INDEX IF NOT EXISTS idx_london_osm_daily_enriched_hour_minute
  ON public.london_osm_daily_enriched (hour, minute);

CREATE INDEX IF NOT EXISTS idx_london_osm_daily_enriched_is_weekend
  ON public.london_osm_daily_enriched (is_weekend);
