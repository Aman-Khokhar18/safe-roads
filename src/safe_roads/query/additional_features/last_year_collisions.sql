DROP TABLE IF EXISTS public.h3_last_year_collisions_wide;

CREATE TABLE public.h3_last_year_collisions_wide AS
WITH y(yr) AS (
  SELECT EXTRACT(YEAR FROM CURRENT_DATE)::int - 1
),
-- If you can, materialize this mapping once (see note below).
mapping AS (
  SELECT DISTINCT ON (cd.h3)
         cd.h3,
         cd.parent_h3,
         cd.neighbor_1, cd.neighbor_2, cd.neighbor_3,
         cd.neighbor_4, cd.neighbor_5, cd.neighbor_6
  FROM public.combined_dataset cd
  WHERE cd.h3 IS NOT NULL
  ORDER BY cd.h3
),
-- Single pass for last year's collision rows; avoid collision::int
last_year AS (
  SELECT d.h3, d.parent_h3
  FROM public.combined_dataset d
  JOIN y ON d.year = y.yr
  WHERE d.collision = 1           -- <<== IMPORTANT: no cast here
),
h3_counts AS (
  SELECT ly.h3, COUNT(*) AS collisions_last_year
  FROM last_year ly
  GROUP BY ly.h3
),
parent_counts AS (
  SELECT ly.parent_h3, COUNT(*) AS parent_collisions_last_year
  FROM last_year ly
  WHERE ly.parent_h3 IS NOT NULL
  GROUP BY ly.parent_h3
)
SELECT
  y.yr AS year,
  m.h3,
  m.parent_h3,
  m.neighbor_1, m.neighbor_2, m.neighbor_3,
  m.neighbor_4, m.neighbor_5, m.neighbor_6,
  COALESCE(h3c.collisions_last_year, 0) AS h3_collisions_last_year,
  COALESCE(n.n1, 0) AS n1_collisions_last_year,
  COALESCE(n.n2, 0) AS n2_collisions_last_year,
  COALESCE(n.n3, 0) AS n3_collisions_last_year,
  COALESCE(n.n4, 0) AS n4_collisions_last_year,
  COALESCE(n.n5, 0) AS n5_collisions_last_year,
  COALESCE(n.n6, 0) AS n6_collisions_last_year,
  COALESCE(pc.parent_collisions_last_year, 0) AS parent_collisions_last_year
FROM mapping m
CROSS JOIN y
LEFT JOIN h3_counts h3c ON h3c.h3 = m.h3
-- Single LATERAL join to fetch all 6 neighbor counts in one go
LEFT JOIN LATERAL (
  WITH neigh(idx, h) AS (
    SELECT * FROM (VALUES
      (1, m.neighbor_1),
      (2, m.neighbor_2),
      (3, m.neighbor_3),
      (4, m.neighbor_4),
      (5, m.neighbor_5),
      (6, m.neighbor_6)
    ) v(idx, h)
  )
  SELECT
    MAX(CASE WHEN idx=1 THEN hc.collisions_last_year END) AS n1,
    MAX(CASE WHEN idx=2 THEN hc.collisions_last_year END) AS n2,
    MAX(CASE WHEN idx=3 THEN hc.collisions_last_year END) AS n3,
    MAX(CASE WHEN idx=4 THEN hc.collisions_last_year END) AS n4,
    MAX(CASE WHEN idx=5 THEN hc.collisions_last_year END) AS n5,
    MAX(CASE WHEN idx=6 THEN hc.collisions_last_year END) AS n6
  FROM neigh
  LEFT JOIN h3_counts hc ON hc.h3 = neigh.h
) n ON TRUE
LEFT JOIN parent_counts pc ON pc.parent_h3 = m.parent_h3
;
