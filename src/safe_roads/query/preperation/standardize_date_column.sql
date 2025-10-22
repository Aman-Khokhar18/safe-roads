DO $$
DECLARE
  r RECORD;
BEGIN
  FOR r IN
    SELECT t.table_schema,
           t.table_name,
           (SELECT c.column_name
            FROM information_schema.columns c
            WHERE c.table_schema = t.table_schema
              AND c.table_name   = t.table_name
              AND c.column_name ILIKE '%date%'
            ORDER BY c.ordinal_position
            LIMIT 1) AS src_col
    FROM information_schema.tables t
    WHERE t.table_type = 'BASE TABLE'
      AND t.table_schema NOT IN ('pg_catalog','information_schema')
      AND t.table_name ILIKE '%attendant%'
  LOOP
    -- if table already has a column named (case-insensitively) 'date', skip
    IF EXISTS (
      SELECT 1
      FROM information_schema.columns c2
      WHERE c2.table_schema = r.table_schema
        AND c2.table_name   = r.table_name
        AND lower(c2.column_name) = 'date'
    ) THEN
      RAISE NOTICE 'Skipping %.%: column "date" already exists',
                   r.table_schema, r.table_name;
    ELSIF r.src_col IS NULL THEN
      RAISE NOTICE 'Skipping %.%: no column containing "date" found',
                   r.table_schema, r.table_name;
    ELSE
      EXECUTE format(
        'ALTER TABLE %I.%I RENAME COLUMN %I TO %I;',
        r.table_schema, r.table_name, r.src_col, 'date'
      );
      RAISE NOTICE 'Renamed %.% column % to "date"',
                   r.table_schema, r.table_name, r.src_col;
    END IF;
  END LOOP;
END
$$;
