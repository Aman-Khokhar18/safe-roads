ALTER TABLE collisiondata
  ADD COLUMN IF NOT EXISTS event_dt timestamptz;

UPDATE collisiondata
SET event_dt = (date_parsed::date + time_parsed::time) AT TIME ZONE 'UTC'
WHERE event_dt IS NULL;