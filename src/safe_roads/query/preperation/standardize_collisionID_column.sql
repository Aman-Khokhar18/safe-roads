DO $$
DECLARE r record;
BEGIN
  FOR r IN
    SELECT c.table_schema, c.table_name, c.column_name
    FROM information_schema.columns c
    JOIN information_schema.tables t
      ON t.table_schema = c.table_schema AND t.table_name = c.table_name
    WHERE t.table_type = 'BASE TABLE'
      AND c.ordinal_position = 1
      AND c.table_schema NOT IN ('pg_catalog', 'information_schema')
      AND c.table_name ILIKE '%attendant%'            -- table name contains "attendant" (case-insensitive)
      AND NOT EXISTS (                                 -- skip if collisionID already exists
        SELECT 1
        FROM information_schema.columns c2
        WHERE c2.table_schema = c.table_schema
          AND c2.table_name = c.table_name
          AND c2.column_name = 'collisionID'
      )
  LOOP
    EXECUTE format(
      'ALTER TABLE %I.%I RENAME COLUMN %I TO %I',
      r.table_schema, r.table_name, r.column_name, 'collisionID'
    );
  END LOOP;
END
$$;
