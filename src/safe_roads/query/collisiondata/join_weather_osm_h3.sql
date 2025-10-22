DROP TABLE IF EXISTS osm_collision_weather CASCADE;

CREATE UNLOGGED TABLE osm_collision_weather AS
SELECT o.*, w.*
FROM osm_collision_h3 o
JOIN weather_hourly w
  ON w.datetime = date_trunc('hour', o.event_dt + interval '30 minutes');

CREATE UNIQUE INDEX IF NOT EXISTS weather_hourly_dt_idx
  ON weather_hourly (datetime);

ALTER TABLE osm_collision_weather SET LOGGED;
