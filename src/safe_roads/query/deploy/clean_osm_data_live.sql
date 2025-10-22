DROP TABLE IF EXISTS roads_features;
CREATE TABLE roads_features AS
WITH normed AS (
  SELECT
    -- was: r.*
    o.*,
    w.*,

    -- counts
    CASE WHEN o.traffic_lights::text   ~ '^\s*\d+\s*$' THEN o.traffic_lights::int    END AS traffic_light_count_i,
    CASE WHEN o.crossings::text        ~ '^\s*\d+\s*$' THEN o.crossings::int         END AS crossing_count_i,
    CASE WHEN o.motorways_other::text  ~ '^\s*\d+\s*$' THEN o.motorways_other::int   END AS motorway_other_count_i,
    CASE WHEN o.cycleways::text        ~ '^\s*\d+\s*$' THEN o.cycleways::int         END AS cycleway_count_i,

    -- normalizers (qualify likely-overlapping IDs)
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.highway::text)), 'null'), 'none'), '')           AS highway_n,
    NULLIF(NULLIF(NULLIF(BTRIM(o.name::text), 'null'), 'none'), '')                     AS name_n,

    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.lanes::text)), 'null'), 'none'), '')             AS lanes_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.surface::text)), 'null'), 'none'), '')           AS surface_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.oneway::text)), 'null'), 'none'), '')            AS oneway_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.junction::text)), 'null'), 'none'), '')          AS junction_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.traffic_signals::text)), 'null'), 'none'), '')   AS traffic_signals_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.traffic_calming::text)), 'null'), 'none'), '')   AS traffic_calming_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.crossing::text)), 'null'), 'none'), '')          AS crossing_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.sidewalk::text)), 'null'), 'none'), '')          AS sidewalk_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.cycleway::text)), 'null'), 'none'), '')          AS cycleway_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.bicycle::text)), 'null'), 'none'), '')           AS bicycle_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.lit::text)), 'null'), 'none'), '')               AS lit_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.access::text)), 'null'), 'none'), '')            AS access_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.vehicle::text)), 'null'), 'none'), '')           AS vehicle_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.hgv::text)), 'null'), 'none'), '')               AS hgv_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.psv::text)), 'null'), 'none'), '')               AS psv_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.bus::text)), 'null'), 'none'), '')               AS bus_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.overtaking::text)), 'null'), 'none'), '')        AS overtaking_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.bridge::text)), 'null'), 'none'), '')            AS bridge_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.tunnel::text)), 'null'), 'none'), '')            AS tunnel_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.layer::text)), 'null'), 'none'), '')             AS layer_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.incline::text)), 'null'), 'none'), '')           AS incline_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.barrier::text)), 'null'), 'none'), '')           AS barrier_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.amenity::text)), 'null'), 'none'), '')           AS amenity_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.road_class3::text)), 'null'), 'none'), '')       AS road_class3_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.borough::text)), 'null'), 'none'), '')           AS borough_n,

    -- H3-ish identifiers (take from OSM to avoid ambiguity)
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.h3::text)), 'null'), 'none'), '')                AS h3_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.neighbor_1::text)), 'null'), 'none'), '')        AS neighbor_1_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.neighbor_2::text)), 'null'), 'none'), '')        AS neighbor_2_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.neighbor_3::text)), 'null'), 'none'), '')        AS neighbor_3_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.neighbor_4::text)), 'null'), 'none'), '')        AS neighbor_4_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.neighbor_5::text)), 'null'), 'none'), '')        AS neighbor_5_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.neighbor_6::text)), 'null'), 'none'), '')        AS neighbor_6_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.parent_h3::text)), 'null'), 'none'), '')         AS parent_h3_n,

    NULLIF(
      regexp_replace(COALESCE(o.maxspeed_mph::text, o.maxspeed_mph::text), '[^0-9\.]', '', 'g'),
      ''
    )::float AS maxspeed_mph_f,

    -- width/smoothness
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.width::text)), 'null'), 'none'), '')             AS width_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(o.smoothness::text)), 'null'), 'none'), '')        AS smoothness_n,

    -- weather (from weather_live w)
    NULLIF(LOWER(BTRIM(w.temp::text)) ,'null')::float  AS temp_f,
    NULLIF(LOWER(BTRIM(w.dwpt::text)) ,'null')::float  AS dwpt_f,
    NULLIF(LOWER(BTRIM(w.rhum::text)) ,'null')::float  AS rhum_f,
    NULLIF(LOWER(BTRIM(w.prcp::text)) ,'null')::float  AS prcp_f,
    NULLIF(LOWER(BTRIM(w.snow::text)) ,'null')::float  AS snow_f,
    NULLIF(LOWER(BTRIM(w.wdir::text)) ,'null')::float  AS wdir_f,
    NULLIF(LOWER(BTRIM(w.wspd::text)) ,'null')::float  AS wspd_f,
    NULLIF(LOWER(BTRIM(w.wpgt::text)) ,'null')::float  AS wpgt_f,
    NULLIF(LOWER(BTRIM(w.pres::text)) ,'null')::float  AS pres_f,
    NULLIF(LOWER(BTRIM(w.tsun::text)) ,'null')::float  AS tsun_f,
    NULLIF(LOWER(BTRIM(w.coco::text)) ,'null')::int    AS coco_i,
    NULLIF(LOWER(BTRIM(w.weather_datetime::text)),'null')::timestamp without time zone AS weather_dt

    -- your existing year/month/day/hour/is_weekend/is_junction/is_turn logic unchanged...
    -- (keep the rest of your CTE as-is)

  FROM osm_data_live_h3_enriched o
  CROSS JOIN weather_live w
),
tok AS (
  SELECT
    n.*,

    -- split on comma/semicolon (NULL stays NULL)
    CASE WHEN surface_n          IS NULL THEN NULL ELSE regexp_split_to_array(surface_n,          '\s*[,;]\s*') END AS surface_a,
    CASE WHEN traffic_calming_n  IS NULL THEN NULL ELSE regexp_split_to_array(traffic_calming_n,  '\s*[,;]\s*') END AS traffic_calming_a,
    CASE WHEN crossing_n         IS NULL THEN NULL ELSE regexp_split_to_array(crossing_n,         '\s*[,;]\s*') END AS crossing_a,
    CASE WHEN sidewalk_n         IS NULL THEN NULL ELSE regexp_split_to_array(sidewalk_n,         '\s*[,;]\s*') END AS sidewalk_a,
    CASE WHEN cycleway_n         IS NULL THEN NULL ELSE regexp_split_to_array(cycleway_n,         '\s*[,;]\s*') END AS cycleway_a,
    CASE WHEN bicycle_n          IS NULL THEN NULL ELSE regexp_split_to_array(bicycle_n,          '\s*[,;]\s*') END AS bicycle_a,
    CASE WHEN access_n           IS NULL THEN NULL ELSE regexp_split_to_array(access_n,           '\s*[,;]\s*') END AS access_a,
    CASE WHEN vehicle_n          IS NULL THEN NULL ELSE regexp_split_to_array(vehicle_n,          '\s*[,;]\s*') END AS vehicle_a,
    CASE WHEN hgv_n              IS NULL THEN NULL ELSE regexp_split_to_array(hgv_n,              '\s*[,;]\s*') END AS hgv_a,
    CASE WHEN psv_n              IS NULL THEN NULL ELSE regexp_split_to_array(psv_n,              '\s*[,;]\s*') END AS psv_a,
    CASE WHEN bus_n              IS NULL THEN NULL ELSE regexp_split_to_array(bus_n,              '\s*[,;]\s*') END AS bus_a,
    CASE WHEN bridge_n           IS NULL THEN NULL ELSE regexp_split_to_array(bridge_n,           '\s*[,;]\s*') END AS bridge_a,
    CASE WHEN tunnel_n           IS NULL THEN NULL ELSE regexp_split_to_array(tunnel_n,           '\s*[,;]\s*') END AS tunnel_a,
    CASE WHEN amenity_n          IS NULL THEN NULL ELSE regexp_split_to_array(amenity_n,          '\s*[,;]\s*') END AS amenity_a
  FROM normed n
),
canon AS (
  SELECT
    t.*,

    -- very light canonicalization where it helps
    (SELECT CASE WHEN COUNT(*)=0 THEN NULL ELSE ARRAY_AGG(val ORDER BY ord) END
     FROM (
       SELECT ord,
              CASE s
                WHEN 'paing_stones' THEN 'paving_stones'
                WHEN 'paved_stones' THEN 'paving_stones'
                WHEN 'paving_slabs' THEN 'paving_stones'
                WHEN 'block_paved'  THEN 'paving_stones'
                WHEN 'brick_paver'  THEN 'brick'
                WHEN 'pebblestone'  THEN 'pebble_stone'
                WHEN 'cobblestone:flattened' THEN 'cobblestone'
                WHEN 'concrete:plates' THEN 'concrete'
                WHEN 'fine_gravel' THEN 'gravel'
                WHEN 'loose_surface' THEN 'ground'
                ELSE s
              END AS val
       FROM unnest(COALESCE(t.surface_a, ARRAY[]::text[])) WITH ORDINALITY AS u(s, ord)
     ) z) AS surface_c,

    (SELECT CASE WHEN COUNT(*)=0 THEN NULL ELSE ARRAY_AGG(val ORDER BY ord) END
     FROM (
       SELECT ord, CASE x WHEN 'humps' THEN 'hump'
                          WHEN 'speed_cushions' THEN 'cushions'
                          ELSE x END AS val
       FROM unnest(COALESCE(t.traffic_calming_a, ARRAY[]::text[])) WITH ORDINALITY AS u(x, ord)
     ) q) AS traffic_calming_c,

    -- crossing/cycleway/tunnel: a couple of common typos
    (SELECT CASE WHEN COUNT(*)=0 THEN NULL ELSE ARRAY_AGG(val ORDER BY ord) END
     FROM (
       SELECT ord, CASE c WHEN 'traffic_lights' THEN 'traffic_signals'
                          WHEN 'pedestrian_signals' THEN 'signals'
                          ELSE c END AS val
       FROM unnest(COALESCE(t.crossing_a, ARRAY[]::text[])) WITH ORDINALITY AS u(c, ord)
     ) q) AS crossing_c,

    (SELECT CASE WHEN COUNT(*)=0 THEN NULL ELSE ARRAY_AGG(val ORDER BY ord) END
     FROM (
       SELECT ord, CASE y WHEN 'trck' THEN 'track'
                          WHEN 'sidewlk' THEN 'sidewalk'
                          WHEN 'shared use' THEN 'shared'
                          ELSE y END AS val
       FROM unnest(COALESCE(t.cycleway_a, ARRAY[]::text[])) WITH ORDINALITY AS u(y, ord)
     ) q) AS cycleway_c,

    (SELECT CASE WHEN COUNT(*)=0 THEN NULL ELSE ARRAY_AGG(val ORDER BY ord) END
     FROM (
       SELECT ord, CASE u WHEN 'bulding_passage' THEN 'building_passage'
                          ELSE u END AS val
       FROM unnest(COALESCE(t.tunnel_a, ARRAY[]::text[])) WITH ORDINALITY AS u(u, ord)
     ) q) AS tunnel_c
  FROM tok t
)
SELECT
  -- ===== keep EXACT names you requested, with cleaned values =====
  h3_n           AS h3,
  neighbor_1_n   AS neighbor_1,
  neighbor_2_n   AS neighbor_2,
  neighbor_3_n   AS neighbor_3,
  neighbor_4_n   AS neighbor_4,
  neighbor_5_n   AS neighbor_5,
  neighbor_6_n   AS neighbor_6,
  parent_h3_n    AS parent_h3,

  highway_n      AS highway,
  name_n         AS name,

  /* lanes: parse all numeric tokens and average them (e.g., '2; 3' -> 2.5) */
  CASE
    WHEN lanes_n IS NULL THEN NULL
    WHEN lanes_n ~ '^\s*\d+(\.\d+)?\s*$' THEN lanes_n::float
    ELSE (
      SELECT AVG((m[1])::float)
      FROM regexp_matches(lanes_n, '([0-9]+(?:\.[0-9]+)?)', 'g') AS m
    )
  END AS lanes,

  -- arrays → comma-joined strings (no {} anymore)
  CASE WHEN surface_c          IS NULL THEN NULL ELSE array_to_string(surface_c,         ',') END AS surface,
  oneway_n       AS oneway,
  junction_n     AS junction,
  traffic_signals_n AS traffic_signals,
  CASE WHEN traffic_calming_c  IS NULL THEN NULL ELSE array_to_string(traffic_calming_c, ',') END AS traffic_calming,
  CASE WHEN crossing_c         IS NULL THEN NULL ELSE array_to_string(crossing_c,        ',') END AS crossing,
  CASE WHEN sidewalk_a         IS NULL THEN NULL ELSE array_to_string(sidewalk_a,        ',') END AS sidewalk,
  CASE WHEN cycleway_c         IS NULL THEN NULL ELSE array_to_string(cycleway_c,        ',') END AS cycleway,
  CASE WHEN bicycle_a          IS NULL THEN NULL ELSE array_to_string(bicycle_a,         ',') END AS bicycle,
  lit_n         AS lit,
  CASE WHEN access_a           IS NULL THEN NULL ELSE array_to_string(access_a,          ',') END AS access,
  CASE WHEN vehicle_a          IS NULL THEN NULL ELSE array_to_string(vehicle_a,         ',') END AS vehicle,
  CASE WHEN hgv_a              IS NULL THEN NULL ELSE array_to_string(hgv_a,             ',') END AS hgv,
  CASE WHEN psv_a              IS NULL THEN NULL ELSE array_to_string(psv_a,             ',') END AS psv,
  CASE WHEN bus_a              IS NULL THEN NULL ELSE array_to_string(bus_a,             ',') END AS bus,
  overtaking_n  AS overtaking,
  CASE WHEN bridge_a           IS NULL THEN NULL ELSE array_to_string(bridge_a,          ',') END AS bridge,
  CASE WHEN tunnel_c           IS NULL THEN NULL ELSE array_to_string(tunnel_c,          ',') END AS tunnel,
  layer_n       AS layer,
  incline_n     AS incline,
  barrier_n     AS barrier,
  CASE WHEN amenity_a          IS NULL THEN NULL ELSE array_to_string(amenity_a,         ',') END AS amenity,
  road_class3_n AS road_class3,
  borough_n     AS borough,

  CASE
    WHEN width_n ~ '[-+]?\d*[\.,]?\d+'
      THEN REPLACE(SUBSTRING(width_n FROM '[-+]?\d*[\.,]?\d+'), ',', '.')::float
    ELSE NULL
  END AS width,

  -- smoothness: map OSM categories to an ordinal
  CASE LOWER(smoothness_n)
    WHEN 'excellent'      THEN 5
    WHEN 'good'           THEN 4
    WHEN 'intermediate'   THEN 3
    WHEN 'bad'            THEN 2
    WHEN 'very_bad'       THEN 1
    WHEN 'horrible'       THEN 0
    WHEN 'very_horrible'  THEN -1
    ELSE NULL
  END AS smoothness,

  -- numeric keep same names
  temp_f   AS temp,
  dwpt_f   AS dwpt,
  rhum_f   AS rhum,
  prcp_f   AS prcp,
  snow_f   AS snow,
  wdir_f   AS wdir,
  wspd_f   AS wspd,
  wpgt_f   AS wpgt,
  pres_f   AS pres,
  tsun_f   AS tsun,
  coco_i   AS coco,
  maxspeed_mph_f AS maxspeed_mph,
  CASE
    WHEN junction_degree::text ~ '^\s*-?\d+\s*$' THEN junction_degree::int
  END AS junction_degree,

  CASE
    WHEN (is_junction::text) ~ '^\s*(t|true|1|yes|y)\s*$'  THEN 1
    WHEN (is_junction::text) ~ '^\s*(f|false|0|no|n)\s*$'  THEN 0
    ELSE NULL
  END AS is_junction,
  CASE
    WHEN (is_turn::text)      ~ '^\s*(t|true|1|yes|y)\s*$' THEN 1
    WHEN (is_turn::text)      ~ '^\s*(f|false|0|no|n)\s*$' THEN 0
    ELSE NULL
  END AS is_turn,


  -- ==== calendar parts from weather_dt ====
  CASE WHEN weather_dt IS NOT NULL THEN EXTRACT(YEAR  FROM weather_dt)::int END AS year,
  CASE WHEN weather_dt IS NOT NULL THEN EXTRACT(MONTH FROM weather_dt)::int END AS month,
  CASE WHEN weather_dt IS NOT NULL THEN EXTRACT(DAY   FROM weather_dt)::int END AS day,
  CASE WHEN weather_dt IS NOT NULL THEN EXTRACT(HOUR  FROM weather_dt)::int END AS hour,
  CASE
    WHEN weather_dt IS NOT NULL THEN CASE WHEN EXTRACT(DOW FROM weather_dt) IN (0,6) THEN 1 ELSE 0 END
    ELSE NULL
  END AS is_weekend,

  -- cyclic encodings from weather_dt
  CASE WHEN weather_dt IS NOT NULL THEN
    sin(2 * pi() * (EXTRACT(HOUR FROM weather_dt) / 24.0))
  END AS hour_sin,
  CASE WHEN weather_dt IS NOT NULL THEN
    cos(2 * pi() * (EXTRACT(HOUR FROM weather_dt) / 24.0))
  END AS hour_cos,

  CASE WHEN weather_dt IS NOT NULL THEN
    sin(2 * pi() * (EXTRACT(DOW FROM weather_dt) / 7.0))
  END AS dow_sin,
  CASE WHEN weather_dt IS NOT NULL THEN
    cos(2 * pi() * (EXTRACT(DOW FROM weather_dt) / 7.0))
  END AS dow_cos,

  CASE WHEN weather_dt IS NOT NULL THEN
    sin(
      2 * pi() * (
        (EXTRACT(DAY FROM weather_dt) - 1)
        / EXTRACT(DAY FROM (date_trunc('month', weather_dt) + interval '1 month - 1 day'))
      )
    )
  END AS dom_sin,
  CASE WHEN weather_dt IS NOT NULL THEN
    cos(
      2 * pi() * (
        (EXTRACT(DAY FROM weather_dt) - 1)
        / EXTRACT(DAY FROM (date_trunc('month', weather_dt) + interval '1 month - 1 day'))
      )
    )
  END AS dom_cos,

  CASE WHEN weather_dt IS NOT NULL THEN
    sin(2 * pi() * ((EXTRACT(MONTH FROM weather_dt)::int - 1) / 12.0))
  END AS month_sin,
  CASE WHEN weather_dt IS NOT NULL THEN
    cos(2 * pi() * ((EXTRACT(MONTH FROM weather_dt)::int - 1) / 12.0))
  END AS month_cos,

  -- === published count fields (explicitly INT) ===
  traffic_light_count_i::int   AS traffic_light_count,
  crossing_count_i::int        AS crossing_count,
  motorway_other_count_i::int  AS motorway_other_count,
  cycleway_count_i::int        AS cycleway_count

FROM canon;
