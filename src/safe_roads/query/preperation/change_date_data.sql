ALTER TABLE public.collisiondata
DROP COLUMN IF EXISTS date_parsed,
ADD COLUMN date_parsed date;

UPDATE public.collisiondata
SET date_parsed = CASE
  WHEN trim("date") ~ '^\d{1,2}/\d{1,2}/\d{4}\s+\d{2}:\d{2}(:\d{2})?$'
    THEN to_timestamp(trim("date"), 'DD/MM/YYYY HH24:MI:SS')::date

  WHEN trim("date") ~ '^\d{1,2}-\d{1,2}-\d{4}\s+\d{2}:\d{2}(:\d{2})?$'
    THEN to_timestamp(trim("date"), 'DD-MM-YYYY HH24:MI:SS')::date

  -- 19/01/2021
  WHEN trim("date") ~ '^\d{1,2}/\d{1,2}/\d{4}$'
    THEN to_date(trim("date"), 'DD/MM/YYYY')

  -- 19-01-2021
  WHEN trim("date") ~ '^\d{1,2}-\d{1,2}-\d{4}$'
    THEN to_date(trim("date"), 'DD-MM-YYYY')

  -- 19-Jan-21 or 19 Jan 21
  WHEN trim("date") ~ '^\d{1,2}[- ][A-Za-z]{3}[- ]\d{2}$'
    THEN to_date(trim("date"), 'DD-Mon-YY')

  -- 19-Jan-2021 or 19 Jan 2021
  WHEN trim("date") ~ '^\d{1,2}[- ][A-Za-z]{3}[- ]\d{4}$'
    THEN to_date(trim("date"), 'DD-Mon-YYYY')

  -- blank -> NULL
  WHEN "date" IS NULL OR trim("date") = '' THEN NULL

  ELSE NULL  -- unrecognized; inspect later
END
WHERE date_parsed IS NULL;  -- keep it idempotent if you re-run
