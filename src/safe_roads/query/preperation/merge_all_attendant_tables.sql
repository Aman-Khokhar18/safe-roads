DROP TABLE IF EXISTS public.CollisionData CASCADE;

CREATE TABLE public.collisiondata (
  "collisionID"  text,
  "Borough"      text,
  "Easting"      text,
  "Northing"     text,
  "date"         text,
  "Time"         text,
  "source"       text
);


DO $$
DECLARE r record;
BEGIN
  FOR r IN
    SELECT n.nspname AS schema_name, c.relname AS table_name
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relkind = 'r'
      AND n.nspname NOT IN ('pg_catalog','information_schema')
      AND c.relname ILIKE '%attendant%'
  LOOP
    EXECUTE format(
      'INSERT INTO public.CollisionData ("collisionID","Borough","Easting","Northing","date","Time","source")
       SELECT "collisionID","Borough","Easting","Northing","date","Time","source"
       FROM %I.%I;',
      r.schema_name, r.table_name
    );
  END LOOP;
END
$$;

DELETE FROM public.collisiondata
WHERE "Time" IS NULL


