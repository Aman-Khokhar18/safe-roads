DROP TABLE IF EXISTS combined_dataset;

CREATE TABLE combined_dataset AS
SELECT rfc.*, 1::int AS collision
FROM roads_features_collision rfc
UNION ALL
SELECT rfn.*, 0::int AS collision
FROM roads_features_negatives rfn;
