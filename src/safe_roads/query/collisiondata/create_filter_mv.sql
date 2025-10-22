CREATE MATERIALIZED VIEW IF NOT EXISTS public.mv_ctx_counts AS
SELECT
  h3,
  date_trunc('hour', event_dt) AS hour_ts,
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
FROM public.osm_collision_weather
GROUP BY h3, date_trunc('hour', event_dt);

-- Helpful index for joins
CREATE INDEX IF NOT EXISTS idx_mv_ctx_h3_hour ON public.mv_ctx_counts (h3, hour_ts);
