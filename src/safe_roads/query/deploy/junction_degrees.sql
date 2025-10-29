-- 0) Columns (no-op if they exist)
ALTER TABLE public.osm_deploy_latest
  ADD COLUMN IF NOT EXISTS junction_degree integer,
  ADD COLUMN IF NOT EXISTS is_junction boolean,
  ADD COLUMN IF NOT EXISTS is_turn boolean;

-- 1) Helpful indexes (by h3 only)
CREATE INDEX IF NOT EXISTS osm_deploy_latest_h3_idx
  ON public.osm_deploy_latest (h3);

CREATE INDEX IF NOT EXISTS osm_junc_h3_idx
  ON public.osm_junctions_yearly (h3);

-- 2) Update by joining MV aggregated at h3 (across all years)
WITH mv_hex AS (
  SELECT
    h3,
    MAX(deg)             AS junction_degree,  -- pick your aggregator
    BOOL_OR(is_junction) AS is_junction,
    BOOL_OR(is_turn)     AS is_turn
  FROM public.osm_junctions_yearly
  GROUP BY h3
)
UPDATE public.osm_deploy_latest AS cd
SET
  junction_degree = mv.junction_degree,
  is_junction     = mv.is_junction,
  is_turn         = mv.is_turn
FROM mv_hex AS mv
WHERE mv.h3 = cd.h3;

-- 3) Defaults for rows with no matching h3
UPDATE public.osm_deploy_latest
SET junction_degree = 0,
    is_junction     = FALSE,
    is_turn         = FALSE
WHERE junction_degree IS NULL;
