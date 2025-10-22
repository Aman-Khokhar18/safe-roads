DROP TABLE IF EXISTS roads_features_collision;
CREATE TABLE roads_features_collision AS
WITH normed AS (
  SELECT
    r.*,

    -- === counts that already live in osm_collision_enriched (null-safe whether INT or numeric TEXT) ===
    CASE WHEN r.traffic_lights::text   ~ '^\s*\d+\s*$' THEN r.traffic_lights::int    END AS traffic_light_count_i,
    CASE WHEN r.crossings::text        ~ '^\s*\d+\s*$' THEN r.crossings::int         END AS crossing_count_i,
    CASE WHEN r.motorways_other::text  ~ '^\s*\d+\s*$' THEN r.motorways_other::int   END AS motorway_other_count_i,
    CASE WHEN r.cycleways::text        ~ '^\s*\d+\s*$' THEN r.cycleways::int         END AS cycleway_count_i,

    -- ========== core text normalizers ==========
    -- helper to null 'null'/'none'/'' and lowercase where appropriate
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(highway::text)), 'null'), 'none'), '')           AS highway_n,
    NULLIF(NULLIF(NULLIF(BTRIM(name::text), 'null'), 'none'), '')                     AS name_n,

    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(lanes::text)), 'null'), 'none'), '')             AS lanes_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(surface::text)), 'null'), 'none'), '')           AS surface_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(oneway::text)), 'null'), 'none'), '')            AS oneway_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(junction::text)), 'null'), 'none'), '')          AS junction_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(traffic_signals::text)), 'null'), 'none'), '')   AS traffic_signals_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(traffic_calming::text)), 'null'), 'none'), '')   AS traffic_calming_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(crossing::text)), 'null'), 'none'), '')          AS crossing_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(sidewalk::text)), 'null'), 'none'), '')          AS sidewalk_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(cycleway::text)), 'null'), 'none'), '')          AS cycleway_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(bicycle::text)), 'null'), 'none'), '')           AS bicycle_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(lit::text)), 'null'), 'none'), '')               AS lit_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(access::text)), 'null'), 'none'), '')            AS access_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(vehicle::text)), 'null'), 'none'), '')           AS vehicle_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(hgv::text)), 'null'), 'none'), '')               AS hgv_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(psv::text)), 'null'), 'none'), '')               AS psv_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(bus::text)), 'null'), 'none'), '')               AS bus_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(overtaking::text)), 'null'), 'none'), '')        AS overtaking_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(bridge::text)), 'null'), 'none'), '')            AS bridge_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(tunnel::text)), 'null'), 'none'), '')            AS tunnel_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(layer::text)), 'null'), 'none'), '')             AS layer_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(incline::text)), 'null'), 'none'), '')           AS incline_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(barrier::text)), 'null'), 'none'), '')           AS barrier_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(amenity::text)), 'null'), 'none'), '')           AS amenity_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(road_class3::text)), 'null'), 'none'), '')       AS road_class3_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(borough::text)), 'null'), 'none'), '')           AS borough_n,

    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(h3::text)), 'null'), 'none'), '')                AS h3_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(neighbor_1::text)), 'null'), 'none'), '')        AS neighbor_1_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(neighbor_2::text)), 'null'), 'none'), '')        AS neighbor_2_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(neighbor_3::text)), 'null'), 'none'), '')        AS neighbor_3_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(neighbor_4::text)), 'null'), 'none'), '')        AS neighbor_4_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(neighbor_5::text)), 'null'), 'none'), '')        AS neighbor_5_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(neighbor_6::text)), 'null'), 'none'), '')        AS neighbor_6_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(parent_h3::text)), 'null'), 'none'), '')         AS parent_h3_n,

    NULLIF(
      regexp_replace(COALESCE(maxspeed_mph::text, maxspeed_mph::text), '[^0-9\.]', '', 'g'),
      ''
    )::float AS maxspeed_mph_f,

    -- width/smoothness (kept as cleaned text)
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(width::text)), 'null'), 'none'), '')             AS width_n,
    NULLIF(NULLIF(NULLIF(LOWER(BTRIM(smoothness::text)), 'null'), 'none'), '')        AS smoothness_n,

    -- weather & misc numeric (keep exact names, cast)
    NULLIF(LOWER(BTRIM(temp::text)) ,'null')::float  AS temp_f,
    NULLIF(LOWER(BTRIM(dwpt::text)) ,'null')::float  AS dwpt_f,
    NULLIF(LOWER(BTRIM(rhum::text)) ,'null')::float  AS rhum_f,
    NULLIF(LOWER(BTRIM(prcp::text)) ,'null')::float  AS prcp_f,
    NULLIF(LOWER(BTRIM(snow::text)) ,'null')::float  AS snow_f,
    NULLIF(LOWER(BTRIM(wdir::text)) ,'null')::float  AS wdir_f,
    NULLIF(LOWER(BTRIM(wspd::text)) ,'null')::float  AS wspd_f,
    NULLIF(LOWER(BTRIM(wpgt::text)) ,'null')::float  AS wpgt_f,
    NULLIF(LOWER(BTRIM(pres::text)) ,'null')::float  AS pres_f,
    NULLIF(LOWER(BTRIM(tsun::text)) ,'null')::float  AS tsun_f,
    NULLIF(LOWER(BTRIM(coco::text)) ,'null')::int    AS coco_i,

    CASE WHEN year::text   ~ '^\s*-?\d+\s*$' THEN year::int   END AS year_i,
    CASE WHEN month::text  ~ '^\s*-?\d+\s*$' THEN month::int  END AS month_i,
    CASE WHEN day::text    ~ '^\s*-?\d+\s*$' THEN day::int    END AS day_i,
    CASE WHEN hour::text   ~ '^\s*-?\d+\s*$' THEN hour::int   END AS hour_i,

    CASE
      WHEN LOWER(BTRIM(is_weekend::text)) IN ('true','t','yes','y','1')  THEN 1
      WHEN LOWER(BTRIM(is_weekend::text)) IN ('false','f','no','n','0')  THEN 0
      WHEN is_weekend::text ~ '^\s*-?\d+\s*$'                            THEN is_weekend::int
      ELSE NULL
    END AS is_weekend_i,

    CASE
      WHEN LOWER(BTRIM(is_junction::text)) IN ('true','t','yes','y','1') THEN 1
      WHEN LOWER(BTRIM(is_junction::text)) IN ('false','f','no','n','0') THEN 0
      WHEN is_junction::text ~ '^\s*-?\d+\s*$'                           THEN is_junction::int
      ELSE NULL
    END AS is_junction_i,

    CASE
      WHEN LOWER(BTRIM(is_turn::text)) IN ('true','t','yes','y','1') THEN 1
      WHEN LOWER(BTRIM(is_turn::text)) IN ('false','f','no','n','0') THEN 0
      WHEN is_turn::text ~ '^\s*-?\d+\s*$'                           THEN is_turn::int
      ELSE NULL
    END AS is_turn_i
  FROM osm_collision_enriched r
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
  year_i   AS year,
  month_i  AS month,
  day_i    AS day,
  hour_i   AS hour,
  is_weekend_i AS is_weekend,
  maxspeed_mph_f AS maxspeed_mph,
  CASE
    WHEN junction_degree::text ~ '^\s*-?\d+\s*$' THEN junction_degree::int
  END AS junction_degree,

  -- booleans as ints, same names
  is_junction_i AS is_junction,
  is_turn_i     AS is_turn,

  -- hour of day (0–24)
  CASE
    WHEN hour_i IS NOT NULL THEN
      sin(2 * pi() * ((hour_i) / 24.0))
  END AS hour_sin,
  CASE
    WHEN hour_i IS NOT NULL THEN
      cos(2 * pi() * ((hour_i) / 24.0))
  END AS hour_cos,

  -- day of week (Sun=0 … Sat=6)
  CASE
    WHEN year_i IS NOT NULL AND month_i IS NOT NULL AND day_i IS NOT NULL THEN
      sin(2 * pi() * (extract(dow FROM make_date(year_i, month_i, day_i)) / 7.0))
  END AS dow_sin,
  CASE
    WHEN year_i IS NOT NULL AND month_i IS NOT NULL AND day_i IS NOT NULL THEN
      cos(2 * pi() * (extract(dow FROM make_date(year_i, month_i, day_i)) / 7.0))
  END AS dow_cos,

  -- day of month (cyclic within month) using event_dt present in r
  CASE
    WHEN year_i IS NOT NULL AND month_i IS NOT NULL AND day_i IS NOT NULL THEN
      sin(
        2 * pi() * (
          (extract(day FROM "event_dt") - 1)
          / extract(day FROM (date_trunc('month', "event_dt") + interval '1 month - 1 day'))
        )
      )
  END AS dom_sin,
  CASE
    WHEN year_i IS NOT NULL AND month_i IS NOT NULL AND day_i IS NOT NULL THEN
      cos(
        2 * pi() * (
          (extract(day FROM "event_dt") - 1)
          / extract(day FROM (date_trunc('month', "event_dt") + interval '1 month - 1 day'))
        )
      )
  END AS dom_cos,

  -- month of year (1–12) → 0-based
  CASE
    WHEN month_i BETWEEN 1 AND 12 THEN
      sin(2 * pi() * ((month_i - 1) / 12.0))
  END AS month_sin,
  CASE
    WHEN month_i BETWEEN 1 AND 12 THEN
      cos(2 * pi() * ((month_i - 1) / 12.0))
  END AS month_cos,

  -- === published count fields (explicitly INT) ===
  traffic_light_count_i::int   AS traffic_light_count,
  crossing_count_i::int        AS crossing_count,
  motorway_other_count_i::int  AS motorway_other_count,
  cycleway_count_i::int        AS cycleway_count

FROM canon;
