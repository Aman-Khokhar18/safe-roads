-- 0) Columns (no-op if they exist)
ALTER TABLE public.combined_dataset
  ADD COLUMN IF NOT EXISTS junction_degree integer,
  ADD COLUMN IF NOT EXISTS is_junction boolean,
  ADD COLUMN IF NOT EXISTS is_turn boolean;

-- 1) Helpful indexes
CREATE INDEX IF NOT EXISTS combined_dataset_h3_year_idx
  ON public.combined_dataset (h3, dt_year);
-- If you don't have dt_year and year is text, use this instead:
-- CREATE INDEX IF NOT EXISTS combined_dataset_h3_yeartxt_idx
--   ON public.combined_dataset (h3, ("year"));

CREATE INDEX IF NOT EXISTS osm_junc_h3_year_idx
  ON public.osm_junctions_yearly (h3, year);

-- 2) Update by joining MV aggregated at (h3, year)
WITH mv_hex_year AS (
  SELECT
    h3,
    year,
    MAX(deg)                    AS junction_degree,  -- choose your aggregation
    BOOL_OR(is_junction)        AS is_junction,
    BOOL_OR(is_turn)            AS is_turn
  FROM public.osm_junctions_yearly
  GROUP BY h3, year
)
UPDATE public.combined_dataset AS cd
SET
  junction_degree = mv.junction_degree,
  is_junction     = mv.is_junction,
  is_turn         = mv.is_turn
FROM mv_hex_year AS mv
WHERE mv.h3 = cd.h3
  AND mv.year = COALESCE(cd.dt_year, NULLIF(cd."year",'')::int);

-- 3) Defaults for rows with no matching (h3,year)
UPDATE public.combined_dataset
SET junction_degree = 0,
    is_junction     = FALSE,
    is_turn         = FALSE
WHERE junction_degree IS NULL;
