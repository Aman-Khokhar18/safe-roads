-- Start fresh
DROP TABLE IF EXISTS public.osm_data_live_h3_enriched;

-- 1) Build a staged table (no on-the-fly junction calc; join osm_junctions_yearly)
CREATE TABLE public._osm_data_live_h3_enriched AS
WITH
-- Context aggregation from your live H3 table
ctx_agg AS (
  SELECT
    h3,
    COUNT(*) FILTER (
      WHERE lower(highway) = 'traffic_signals'
         OR (lower(highway) = 'crossing' AND lower(coalesce(crossing,'')) ~ '\btraffic_signals\b')
    ) AS traffic_lights,
    COUNT(*) FILTER (WHERE lower(highway) = 'crossing') AS crossings,
    COUNT(*) FILTER (WHERE lower(highway) = 'cycleway') AS cycleways,
    COUNT(*) FILTER (WHERE lower(highway) IN (
      'motorway','motorway_link','trunk','trunk_link','motorway_junction',
      'primary','primary_link','secondary','secondary_link','tertiary','tertiary_link',
      'residential','unclassified','living_street','busway','track',
      'mini_roundabout','turning_circle','turning circle'
    )) AS motorways_total
  FROM public.osm_data_live_h3
  GROUP BY h3
),

-- Roll up the existing junctions table by H3 (handle NULLs robustly)
junction_cell AS (
  SELECT
    j.h3,
    MAX(COALESCE(j.deg, 0))                 AS junction_degree,
    BOOL_OR(COALESCE(j.is_junction, FALSE)) AS is_junction,
    BOOL_OR(COALESCE(j.is_turn,     FALSE)) AS is_turn
  FROM public.osm_junctions_yearly j
  GROUP BY j.h3
),

-- Base rows filtered for valid geometries + normalized road class
base AS (
  SELECT
    t.*,
    t.ctid AS _ctid,
    COALESCE(t.h3, t.h3) AS h3_key,  -- normalized H3 key (kept same name as yours)
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
  FROM public.osm_data_live_h3 AS t
  WHERE t.geometry IS NOT NULL
    AND ST_Dimension(t.geometry) >= 1
)

SELECT
  b.*,
  -- speed normalization
  CASE
    WHEN b.maxspeed ~* 'km/?h' AND b.maxspeed ~* '(\d+(\.\d+)?)'
      THEN ROUND(((regexp_match(b.maxspeed, '(\d+(\.\d+)?)'))[1])::numeric * 0.621371)::int
    WHEN b.maxspeed ~* '(\d+(\.\d+)?)'
      THEN ROUND(((regexp_match(b.maxspeed, '(\d+(\.\d+)?)'))[1])::numeric)::int
    ELSE NULL
  END AS maxspeed_mph,

  -- contextual counts
  ca.traffic_lights,
  ca.crossings,
  ca.cycleways,
  GREATEST(
    ca.motorways_total - CASE WHEN b.road_class3 = 'motorway' THEN 1 ELSE 0 END,
    0
  ) AS motorways_other,

  -- junction features from osm_junctions_yearly
  COALESCE(jc.is_junction, FALSE) AS is_junction,
  COALESCE(jc.is_turn,     FALSE) AS is_turn,
  COALESCE(jc.junction_degree, 0) AS junction_degree

FROM base b
LEFT JOIN ctx_agg      ca
  ON h3index_to_bigint(ca.h3) = h3index_to_bigint(b.h3_key)
LEFT JOIN junction_cell jc
  ON h3index_to_bigint(jc.h3) = h3index_to_bigint(b.h3_key)
;

-- 2) Drop every date/time column from the staged table (unchanged)
DO $$
DECLARE r record;
BEGIN
  FOR r IN
    SELECT c.column_name
    FROM information_schema.columns c
    WHERE c.table_schema = 'public'
      AND c.table_name   = '_osm_data_live_h3_enriched_stage'
      AND (
           c.data_type IN (
             'date',
             'time without time zone',
             'time with time zone',
             'timestamp without time zone',
             'timestamp with time zone',
             'interval'
           )
           OR c.udt_name IN (
             'date','time','timetz','timestamp','timestamptz','interval'
           )
      )
  LOOP
    EXECUTE format(
      'ALTER TABLE public._osm_data_live_h3_enriched_stage DROP COLUMN %I',
      r.column_name
    );
  END LOOP;
END $$;

-- 3) Rename staged table to final name
ALTER TABLE public._osm_data_live_h3_enriched
RENAME TO osm_data_live_h3_enriched;

-- 4) Indexes for common filters/joins
CREATE INDEX IF NOT EXISTS idx_osm_data_live_h3_enriched_h3key
  ON public.osm_data_live_h3_enriched (h3_key);

CREATE INDEX IF NOT EXISTS idx_osm_data_live_h3_enriched_roadclass
  ON public.osm_data_live_h3_enriched (road_class3);
