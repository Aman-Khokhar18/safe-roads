SELECT
  n.nspname AS schema,
  c.relname AS table,
  ROUND(pg_total_relation_size(c.oid) / 1024.0 / 1024.0 / 1024.0, 2) AS total_gib
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r','p')                   -- tables + partitioned tables
  AND n.nspname NOT IN ('pg_catalog','information_schema')
ORDER BY total_gib DESC;
