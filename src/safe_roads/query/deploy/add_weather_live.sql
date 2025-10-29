-- new table with live weather added to every row
DROP TABLE IF EXISTS public.osm_deploy_latest_w;
CREATE TABLE public.osm_deploy_latest_w AS
SELECT
  d.*,
  w.weather_datetime,
  w.temp,       
  w.dwpt,
  w.rhum,
  w.prcp,
  w.snow,
  w.wdir,
  w.wspd,
  w.wpgt,
  w.pres,
  w.tsun,
  w.coco
FROM public.osm_deploy_latest d
CROSS JOIN (
  SELECT * FROM public.weather_live
  LIMIT 1
) AS w;
