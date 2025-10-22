DROP MATERIALIZED VIEW IF EXISTS random_month_datetime_samples;
CREATE MATERIALIZED VIEW random_month_datetime_samples AS
WITH
-- >>> Edit these knobs:
params AS (
  SELECT
    30::int  AS days_per_month,   -- how many distinct days to pick per month
    30::int  AS times_per_day     -- how many distinct HH:MM times per picked day
),

months AS (
  SELECT generate_series(
           date '2015-01-01',
           date '2024-12-01',
           interval '1 month'
         )::date AS month_start
),

bounds AS (
  SELECT
    m.month_start,
    (date_trunc('month', m.month_start) + interval '1 month' - interval '1 day')::date AS month_end,
    extract(day from (date_trunc('month', m.month_start) + interval '1 month' - interval '1 day'))::int AS days_in_month
  FROM months m
),

-- Sample distinct days per month (without replacement)
picked_days AS (
  SELECT month_start, day_num
  FROM (
    SELECT
      b.month_start,
      gs AS day_num,
      row_number() OVER (PARTITION BY b.month_start ORDER BY random()) AS rn,
      b.days_in_month
    FROM bounds b
    CROSS JOIN LATERAL generate_series(1, b.days_in_month) AS gs
  ) s
  CROSS JOIN params p
  WHERE rn <= LEAST(p.days_per_month, s.days_in_month)
),

-- Sample distinct minutes per (month, day) (without replacement)
picked_minutes AS (
  SELECT month_start, day_num, minute_of_day
  FROM (
    SELECT
      pd.month_start,
      pd.day_num,
      gs AS minute_of_day,
      row_number() OVER (PARTITION BY pd.month_start, pd.day_num ORDER BY random()) AS rn
    FROM picked_days pd
    CROSS JOIN LATERAL generate_series(0, 1439) AS gs
  ) t
  CROSS JOIN params p
  WHERE rn <= p.times_per_day
)

SELECT
  to_char(pm.month_start, 'YYYY-MM') AS year_month,
  (date_trunc('month', pm.month_start)::date + (pm.day_num - 1))                    AS random_date,
  ((time '00:00') + (pm.minute_of_day || ' minutes')::interval)::time(0)           AS random_time,
  ((date_trunc('month', pm.month_start)::date + (pm.day_num - 1))
     + (pm.minute_of_day || ' minutes')::interval)                                  AS random_timestamp
FROM picked_minutes pm
ORDER BY year_month, random_date, random_time;
