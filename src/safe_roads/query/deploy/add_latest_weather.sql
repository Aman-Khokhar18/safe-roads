CREATE OR REPLACE VIEW public.osm_deploy_latest_w AS
WITH latest_weather AS (
  SELECT *
  FROM public.weather_live
  LIMIT 1
)
SELECT
  d.*,
  w.weather_datetime,
  w.temp, w.dwpt, w.rhum, w.prcp, w.snow, w.wdir, w.wspd, w.wpgt, w.pres, w.tsun, w.coco,

  -- extracted parts
  EXTRACT(YEAR  FROM w.weather_datetime)::int  AS dt_year,
  EXTRACT(MONTH FROM w.weather_datetime)::int  AS dt_month,
  EXTRACT(DAY   FROM w.weather_datetime)::int  AS dt_day,
  EXTRACT(HOUR  FROM w.weather_datetime)::int  AS dt_hour,
  CASE WHEN EXTRACT(ISODOW FROM w.weather_datetime) IN (6,7) THEN 1 ELSE 0 END AS dt_is_weekend,
  CASE WHEN EXTRACT(ISODOW FROM w.weather_datetime) BETWEEN 1 AND 5 THEN 1 ELSE 0 END AS dt_is_weekday,

  -- cyclical encodings (ISODOW 1–7 -> shift to 0–6)
  sin(2*pi() * EXTRACT(HOUR  FROM w.weather_datetime)/24.0)                     AS hour_sin,
  cos(2*pi() * EXTRACT(HOUR  FROM w.weather_datetime)/24.0)                     AS hour_cos,
  sin(2*pi() * (EXTRACT(ISODOW FROM w.weather_datetime)-1)/7.0)                 AS dow_sin,
  cos(2*pi() * (EXTRACT(ISODOW FROM w.weather_datetime)-1)/7.0)                 AS dow_cos,
  sin(2*pi() * EXTRACT(DAY   FROM w.weather_datetime)/31.0)                     AS dom_sin,
  cos(2*pi() * EXTRACT(DAY   FROM w.weather_datetime)/31.0)                     AS dom_cos,
  sin(2*pi() * EXTRACT(MONTH FROM w.weather_datetime)/12.0)                     AS month_sin,
  cos(2*pi() * EXTRACT(MONTH FROM w.weather_datetime)/12.0)                     AS month_cos,

  w.retrieved_at_utc
FROM public.osm_deploy_latest d
CROSS JOIN latest_weather w;
