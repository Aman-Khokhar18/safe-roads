DROP TABLE IF EXISTS london_osm_weather;

CREATE UNLOGGED TABLE london_osm_weather AS
SELECT
  d.*,
  w.*                                  -- includes weather_datetime; no name clash
FROM london_osm_daily_enriched d
LEFT JOIN LATERAL (
  SELECT *
  FROM weather_hourly w
  WHERE w.datetime BETWEEN d.timestamp - interval '6 hours'
                               AND d.timestamp + interval '6 hours'
  ORDER BY ABS(EXTRACT(EPOCH FROM (w.datetime - d.timestamp)))  -- nearest first
  LIMIT 1
) AS w ON TRUE;

ALTER TABLE public.london_osm_weather SET LOGGED;