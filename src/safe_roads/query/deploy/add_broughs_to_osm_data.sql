-- Add both columns
ALTER TABLE public.osm_latest_ml_features
  ADD COLUMN IF NOT EXISTS borough text,
  ADD COLUMN IF NOT EXISTS res_11 h3index;

-- Fill them from london_h3 using the h3 key
UPDATE public.osm_latest_ml_features AS o
SET
  borough = l."Borough",
  res_11  = l.res_11
FROM public.london_h3 AS l
WHERE o.h3::h3index = l.h3::h3index;
