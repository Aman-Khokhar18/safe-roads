DELETE FROM london_osm_weather a
USING osm_collision_enriched b
WHERE b.h3 = a.h3
  AND ABS(EXTRACT(EPOCH FROM (a.timestamp - b.event_dt))) <= 3600;