DROP MATERIALIZED VIEW IF EXISTS public.osm_junctions_yearly;

CREATE MATERIALIZED VIEW public.osm_junctions_yearly AS
WITH roads_osm AS (  -- primary (has year)
  SELECT
    (year)::int AS year,
    ST_LineMerge(ST_CollectionExtract(ST_Force2D("geometry"), 2)) AS geom_osm_4326,
    osm_pk
  FROM public."OSM_data"
  WHERE "geometry" IS NOT NULL
),
roads_osmx AS (       -- fallback: alias osmx_pk -> osm_pk
  SELECT
    ST_LineMerge(ST_CollectionExtract(ST_Force2D(geometry), 2)) AS geom_osmx_4326,
    osmx_pk AS osm_pk
  FROM public.osmx_data
  WHERE geometry IS NOT NULL
),
roads AS (            -- prefer OSM_data geom; else take osmx_data geom (by pk)
  SELECT
    r.year,
    COALESCE(r.geom_osm_4326, x.geom_osmx_4326) AS geom_4326,
    r.osm_pk
  FROM roads_osm r
  LEFT JOIN roads_osmx x USING (osm_pk)
),
roads_lines AS (
  SELECT *
  FROM roads
  WHERE NOT ST_IsEmpty(geom_4326)
    AND ST_GeometryType(geom_4326) IN ('ST_LineString','ST_MultiLineString')
),
noded AS (
  SELECT year, ST_Node(geom_4326) AS geom_4326, osm_pk
  FROM roads_lines
),
verts AS (
  SELECT
    year,
    ST_Transform((dp).geom, 27700) AS pt_27700,   -- BNG meters for 1 m snapping
    osm_pk
  FROM (
    SELECT year, ST_DumpPoints(geom_4326) AS dp, osm_pk
    FROM noded
  ) s
),
junction_candidates AS (
  SELECT
    year,
    ST_SnapToGrid(pt_27700, 1.0) AS snapped_pt_27700,  -- 1 m grid
    COUNT(DISTINCT osm_pk) AS deg
  FROM verts
  GROUP BY year, ST_SnapToGrid(pt_27700, 1.0)
)
SELECT
  year,
  ST_Transform(snapped_pt_27700, 4326)              AS geom,       -- geometry(Point,4326)
  (ST_Transform(snapped_pt_27700, 4326))::geography AS geom_geog,  -- geography(Point)
  deg,
  (deg >= 3) AS is_junction,
  (deg >= 2) AS is_turn,
  public.h3_lat_lng_to_cell(
    point(
      ST_X(ST_Transform(snapped_pt_27700, 4326)),   -- lon
      ST_Y(ST_Transform(snapped_pt_27700, 4326))    -- lat
    ),
    12
  ) AS h3,
  md5(
    year::text ||
    ST_AsEWKB(ST_SnapToGrid(snapped_pt_27700, 1.0))
  ) AS junction_key
FROM junction_candidates;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_osm_junc_year_btree ON public.osm_junctions_yearly (year);
CREATE INDEX IF NOT EXISTS idx_osm_junc_geom_gist  ON public.osm_junctions_yearly USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_osm_junc_geog_gist  ON public.osm_junctions_yearly USING GIST (geom_geog);
CREATE INDEX IF NOT EXISTS idx_osm_junc_year_h3    ON public.osm_junctions_yearly (year, h3);

CREATE UNIQUE INDEX IF NOT EXISTS ux_osm_junc_year_unique
  ON public.osm_junctions_yearly (year, junction_key);
