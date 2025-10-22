CREATE TABLE london_h3_filtered AS
SELECT l.*
FROM london_h3 l
WHERE EXISTS (
  SELECT 1
  FROM osm_h3 o
  WHERE o.h3 = l.h3
);
