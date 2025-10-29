-- 0) Add needed columns (no-ops if they already exist)
ALTER TABLE public.osm_deploy_latest_w
  ADD COLUMN IF NOT EXISTS dt_year       integer,
  ADD COLUMN IF NOT EXISTS dt_month      integer,
  ADD COLUMN IF NOT EXISTS dt_day        integer,
  ADD COLUMN IF NOT EXISTS dt_hour       integer,
  ADD COLUMN IF NOT EXISTS dt_is_weekend smallint,
  ADD COLUMN IF NOT EXISTS dt_is_weekday smallint,
  ADD COLUMN IF NOT EXISTS hour_sin      double precision,
  ADD COLUMN IF NOT EXISTS hour_cos      double precision,
  ADD COLUMN IF NOT EXISTS dow_sin       double precision,
  ADD COLUMN IF NOT EXISTS dow_cos       double precision,
  ADD COLUMN IF NOT EXISTS dom_sin       double precision,
  ADD COLUMN IF NOT EXISTS dom_cos       double precision,
  ADD COLUMN IF NOT EXISTS month_sin     double precision,
  ADD COLUMN IF NOT EXISTS month_cos     double precision,
  ADD COLUMN IF NOT EXISTS retrieved_at_utc timestamptz;

-- 1) Extract parts from weather_datetime
UPDATE public.osm_deploy_latest_w AS cd
SET
  dt_year  = CASE WHEN cd.weather_datetime IS NULL THEN NULL
                  ELSE EXTRACT(YEAR  FROM cd.weather_datetime)::int END,
  dt_month = CASE WHEN cd.weather_datetime IS NULL THEN NULL
                  ELSE EXTRACT(MONTH FROM cd.weather_datetime)::int END,
  dt_day   = CASE WHEN cd.weather_datetime IS NULL THEN NULL
                  ELSE EXTRACT(DAY   FROM cd.weather_datetime)::int END,
  dt_hour  = CASE WHEN cd.weather_datetime IS NULL THEN NULL
                  ELSE EXTRACT(HOUR  FROM cd.weather_datetime)::int END,
  dt_is_weekend = CASE
                    WHEN cd.weather_datetime IS NULL THEN NULL
                    WHEN EXTRACT(ISODOW FROM cd.weather_datetime) IN (6,7) THEN 1 ELSE 0
                  END,
  dt_is_weekday = CASE
                    WHEN cd.weather_datetime IS NULL THEN NULL
                    WHEN EXTRACT(ISODOW FROM cd.weather_datetime) BETWEEN 1 AND 5 THEN 1 ELSE 0
                  END;

-- 2) Cyclical encodings using weather_datetime-derived parts
UPDATE public.osm_deploy_latest_w AS cd
SET
  hour_sin  = CASE WHEN cd.dt_hour  IS NULL THEN NULL ELSE sin(2 * pi() * cd.dt_hour  / 24.0) END,
  hour_cos  = CASE WHEN cd.dt_hour  IS NULL THEN NULL ELSE cos(2 * pi() * cd.dt_hour  / 24.0) END,
  dow_sin   = CASE WHEN cd.weather_datetime IS NULL THEN NULL
                   ELSE sin(2 * pi() * EXTRACT(DOW  FROM cd.weather_datetime) / 7.0) END,
  dow_cos   = CASE WHEN cd.weather_datetime IS NULL THEN NULL
                   ELSE cos(2 * pi() * EXTRACT(DOW  FROM cd.weather_datetime) / 7.0) END,
  dom_sin   = CASE WHEN cd.weather_datetime IS NULL THEN NULL
                   ELSE sin(2 * pi() * EXTRACT(DAY  FROM cd.weather_datetime) / 31.0) END,
  dom_cos   = CASE WHEN cd.weather_datetime IS NULL THEN NULL
                   ELSE cos(2 * pi() * EXTRACT(DAY  FROM cd.weather_datetime) / 31.0) END,
  month_sin = CASE WHEN cd.dt_month IS NULL THEN NULL ELSE sin(2 * pi() * cd.dt_month / 12.0) END,
  month_cos = CASE WHEN cd.dt_month IS NULL THEN NULL ELSE cos(2 * pi() * cd.dt_month / 12.0) END;

-- 3) Fill retrieved_at_utc from the latest weather_live row (adjust if needed)
UPDATE public.osm_deploy_latest_w AS cd
SET retrieved_at_utc = w.retrieved_at_utc
FROM (
  SELECT retrieved_at_utc
  FROM public.weather_live
  ORDER BY retrieved_at_utc DESC
  LIMIT 1
) AS w
WHERE cd.retrieved_at_utc IS NULL;
