-- Build once: enriched collisions saved as a TABLE
DROP TABLE IF EXISTS public.osm_collision_enriched;

CREATE TABLE public.osm_collision_enriched AS
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
  FROM public.osm_collision_weather AS t
),
chosen AS (
  SELECT DISTINCT ON (b."collisionID")
    b.*
  FROM base b
  ORDER BY
    b."collisionID",
    (b.road_class3 = 'motorway') DESC,
    b.event_dt DESC NULLS LAST,
    b.station_id NULLS LAST,
    b._ctid
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
  c.*,

  -- Borough: lowercase, remove spaces & punctuation
  LOWER(REGEXP_REPLACE(c."Borough", '[^a-z0-9]+', '', 'gi')) AS borough,

  -- Maxspeed → mph (extract number; convert km/h→mph when needed)
  CASE
    WHEN c.maxspeed ~* 'km/?h' AND c.maxspeed ~* '(\d+(\.\d+)?)'
      THEN ROUND(((regexp_match(c.maxspeed, '(\d+(\.\d+)?)'))[1])::numeric * 0.621371)::int
    WHEN c.maxspeed ~* '(\d+(\.\d+)?)'
      THEN ROUND(((regexp_match(c.maxspeed, '(\d+(\.\d+)?)'))[1])::numeric)::int
    ELSE NULL
  END AS maxspeed_mph,

  -- Date parts (no "event_" prefix)
  EXTRACT(DAY    FROM c.event_dt)::int AS day,
  EXTRACT(HOUR   FROM c.event_dt)::int AS hour,
  EXTRACT(MINUTE FROM c.event_dt)::int AS minute,
  EXTRACT(MONTH  FROM c.event_dt)::int AS month_extract,
  EXTRACT(DOW    FROM c.event_dt)::int AS dow,   -- 0=Sun … 6=Sat
  (EXTRACT(DOW   FROM c.event_dt) IN (0,6)) AS is_weekend,

  -- Context counts
  cc.traffic_lights,
  cc.crossings,
  cc.cycleways,
  GREATEST(
    cc.motorways_total - CASE WHEN c.road_class3 = 'motorway' THEN 1 ELSE 0 END,
    0
  ) AS motorways_other,

  -- Junction features
  COALESCE(jc.is_junction, FALSE)   AS is_junction,
  COALESCE(jc.is_turn,     FALSE)   AS is_turn,
  COALESCE(jc.junction_degree, 0)   AS junction_degree

FROM chosen c
LEFT JOIN public.mv_ctx_counts cc
  ON cc.h3 = c.h3
 AND cc.hour_ts = date_trunc('hour', c.event_dt)
LEFT JOIN junction_cell jc
  ON jc.h3 = c.h3;



DELETE FROM public.osm_collision_enriched
WHERE geometry IS NULL OR ST_IsEmpty(geometry);