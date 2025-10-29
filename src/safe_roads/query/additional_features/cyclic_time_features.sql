ALTER TABLE public.combined_dataset
  ADD COLUMN hour_sin  double precision,
  ADD COLUMN hour_cos  double precision,
  ADD COLUMN dow_sin   double precision,
  ADD COLUMN dow_cos   double precision,
  ADD COLUMN dom_sin   double precision,
  ADD COLUMN dom_cos   double precision,
  ADD COLUMN month_sin double precision,
  ADD COLUMN month_cos double precision;

UPDATE public.combined_dataset AS cd
SET
  hour_sin  = sin(2 * pi() * cd.dt_hour / 24.0),
  hour_cos  = cos(2 * pi() * cd.dt_hour / 24.0),
  dow_sin   = sin(2 * pi() * extract(dow  from cd."datetime") / 7.0),
  dow_cos   = cos(2 * pi() * extract(dow  from cd."datetime") / 7.0),
  dom_sin   = sin(2 * pi() * extract(day  from cd."datetime") / 31.0),
  dom_cos   = cos(2 * pi() * extract(day  from cd."datetime") / 31.0),
  month_sin = sin(2 * pi() * cd.dt_month / 12.0),
  month_cos = cos(2 * pi() * cd.dt_month / 12.0);
