DROP TABLE IF EXISTS london_osm_daily;

CREATE TABLE london_osm_daily AS
SELECT *
FROM london_h3_datetime AS l
JOIN osm_h3 AS o
  USING (h3)
WHERE o.year::int = EXTRACT(YEAR FROM l."timestamp")::int;
