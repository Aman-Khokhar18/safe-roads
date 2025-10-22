ALTER TABLE public.collisiondata
  DROP COLUMN IF EXISTS time_parsed,
  ADD COLUMN time_parsed time;

UPDATE public.collisiondata
SET time_parsed = CASE
  -- 1521 
  WHEN trim("Time") ~ '^''?\d{3,4}$'
  THEN to_timestamp(
         lpad(regexp_replace(trim("Time"), '^''', ''), 4, '0'),
         'HH24MI'
       )::time
       
  -- HH:MM:SS
  WHEN trim("Time") ~ '^\d{1,2}:\d{2}:\d{2}$'
    THEN to_timestamp(trim("Time"), 'HH24:MI:SS')::time

  -- HH:MM
  WHEN trim("Time") ~ '^\d{1,2}:\d{2}$'
    THEN to_timestamp(trim("Time"), 'HH24:MI')::time

  WHEN "Time" IS NULL OR trim("Time") = '' THEN NULL
  ELSE NULL
END
WHERE time_parsed IS NULL;
