-- drop first (optional)
DROP TABLE IF EXISTS public.negative_dataset;

-- create from the single WITH…SELECT statement
CREATE TABLE public.negative_dataset AS
WITH
-- 1) Per-collision aggregation from raw OSM rows (no arrays)
per_collision AS (
  SELECT
    pair_id,
    COUNT(*)::int AS n_osm_features,

    /* numeric aggregates */
    AVG(lanes_num)::float           AS lanes_num_avg,
    MAX(lanes_num)::float           AS lanes_num_max,
    AVG(width_m)::float             AS width_m_avg,
    MAX(width_m)::float             AS width_m_max,
    AVG(incline_pc)::float          AS incline_pc_avg,
    AVG(incline_abs_pc)::float      AS incline_abs_pc_avg,
    AVG(maxspeed_mph)::float        AS maxspeed_mph_avg,
    MAX(maxspeed_mph)::float        AS maxspeed_mph_max,
    AVG(n_left_lanes)::float        AS n_left_lanes_avg,
    AVG(n_through_lanes)::float     AS n_through_lanes_avg,
    AVG(n_right_lanes)::float       AS n_right_lanes_avg,
    SUM(COALESCE(bus_routes_cnt,0))::int AS bus_routes_cnt_sum,
    MAX(bus_routes_cnt)::int        AS bus_routes_cnt_max,
    AVG(smoothness_score)::float    AS smoothness_score_avg,

    /* boolean counts (NULL-safe) */
    SUM(COALESCE(is_motorway::int,0))           AS cnt_is_motorway,
    SUM(COALESCE(is_trunk::int,0))              AS cnt_is_trunk,
    SUM(COALESCE(is_primary::int,0))            AS cnt_is_primary,
    SUM(COALESCE(is_secondary::int,0))          AS cnt_is_secondary,
    SUM(COALESCE(is_tertiary::int,0))           AS cnt_is_tertiary,
    SUM(COALESCE(is_residential::int,0))        AS cnt_is_residential,
    SUM(COALESCE(is_service::int,0))            AS cnt_is_service,
    SUM(COALESCE(is_track_or_path::int,0))      AS cnt_is_track_or_path,
    SUM(COALESCE(is_foot_or_ped::int,0))        AS cnt_is_foot_or_ped,
    SUM(COALESCE(has_pt_stop::int,0))           AS cnt_has_pt_stop,

    SUM(COALESCE(is_paved::int,0))              AS cnt_is_paved,
    SUM(COALESCE(is_national_speed_limit::int,0)) AS cnt_is_nsl,
    SUM(COALESCE(is_variable_or_none::int,0))   AS cnt_speed_var_or_none,

    SUM(COALESCE(has_signals::int,0))           AS cnt_has_signals,
    SUM(COALESCE(has_traffic_calming::int,0))   AS cnt_has_traffic_calming,
    SUM(COALESCE(has_crossing::int,0))          AS cnt_has_crossing,
    SUM(COALESCE(crossing_signalised::int,0))   AS cnt_crossing_signalised,
    SUM(COALESCE(crossing_zebra::int,0))        AS cnt_crossing_zebra,

    SUM(COALESCE(has_cycle_infra::int,0))       AS cnt_has_cycle_infra,
    SUM(COALESCE(has_contraflow_cycle::int,0))  AS cnt_has_contraflow_cycle,
    SUM(COALESCE(has_cycle_track::int,0))       AS cnt_has_cycle_track,
    SUM(COALESCE(has_cycle_lane::int,0))        AS cnt_has_cycle_lane,

    SUM(COALESCE(is_lit::int,0))                AS cnt_is_lit,
    SUM(COALESCE(has_lit_schedule::int,0))      AS cnt_has_lit_schedule,

    SUM(COALESCE(motor_restricted::int,0))      AS cnt_motor_restricted,
    SUM(COALESCE(hgv_restricted::int,0))        AS cnt_hgv_restricted,
    SUM(COALESCE(psv_priority::int,0))          AS cnt_psv_priority,
    SUM(COALESCE(bus_priority::int,0))          AS cnt_bus_priority,

    SUM(COALESCE(is_bridge::int,0))             AS cnt_is_bridge,
    SUM(COALESCE(is_tunnel::int,0))             AS cnt_is_tunnel,
    SUM(COALESCE(has_barrier::int,0))           AS cnt_has_barrier,
    SUM(COALESCE(has_amenity::int,0))           AS cnt_has_amenity,
    SUM(COALESCE(has_bus_stop::int,0))          AS cnt_has_bus_stop,
    SUM(COALESCE(has_mini_roundabout::int,0))   AS cnt_has_mini_roundabout,
    SUM(COALESCE(has_speed_camera::int,0))      AS cnt_has_speed_camera,
    SUM(COALESCE(overtaking_allowed::int,0))    AS cnt_overtaking_allowed,

    /* oneway distribution */
    COUNT(*) FILTER (WHERE oneway_code =  1) AS cnt_oneway_forward,
    COUNT(*) FILTER (WHERE oneway_code =  0) AS cnt_oneway_bidirectional,
    COUNT(*) FILTER (WHERE oneway_code = -1) AS cnt_oneway_reverse,

    /* categorical -> counts (extend as needed) */
    COUNT(*) FILTER (WHERE surface_cat = 'paved')    AS cnt_surface_paved,
    COUNT(*) FILTER (WHERE surface_cat = 'unpaved')  AS cnt_surface_unpaved,
    COUNT(*) FILTER (WHERE surface_cat = 'asphalt')  AS cnt_surface_asphalt,
    COUNT(*) FILTER (WHERE surface_cat = 'gravel')   AS cnt_surface_gravel,
    COUNT(*) FILTER (WHERE surface_cat = 'dirt')     AS cnt_surface_dirt,

    COUNT(*) FILTER (WHERE sidewalk_cat = 'both')    AS cnt_sidewalk_both,
    COUNT(*) FILTER (WHERE sidewalk_cat = 'left')    AS cnt_sidewalk_left,
    COUNT(*) FILTER (WHERE sidewalk_cat = 'right')   AS cnt_sidewalk_right,
    COUNT(*) FILTER (WHERE sidewalk_cat = 'none')    AS cnt_sidewalk_none,

    COUNT(*) FILTER (WHERE bicycle_access = 'yes')        AS cnt_bicycle_yes,
    COUNT(*) FILTER (WHERE bicycle_access = 'designated') AS cnt_bicycle_designated,
    COUNT(*) FILTER (WHERE bicycle_access = 'permissive') AS cnt_bicycle_permissive,
    COUNT(*) FILTER (WHERE bicycle_access = 'no')         AS cnt_bicycle_no,

    COUNT(*) FILTER (WHERE access_cat = 'permissive') AS cnt_access_permissive,
    COUNT(*) FILTER (WHERE access_cat = 'destination') AS cnt_access_destination,
    COUNT(*) FILTER (WHERE access_cat = 'private')     AS cnt_access_private,
    COUNT(*) FILTER (WHERE access_cat = 'no')          AS cnt_access_no,

    COUNT(*) FILTER (WHERE lit_mode = 'always')   AS cnt_litmode_always,
    COUNT(*) FILTER (WHERE lit_mode = 'limited')  AS cnt_litmode_limited
  FROM public.london_osm_weather
  GROUP BY pair_id
),

-- proportions to de-bias by number of OSM rows
per_collision_with_shares AS (
  SELECT
    p.*,
    (cnt_is_motorway::float      / NULLIF(n_osm_features,0)) AS share_is_motorway,
    (cnt_is_trunk::float         / NULLIF(n_osm_features,0)) AS share_is_trunk,
    (cnt_is_primary::float       / NULLIF(n_osm_features,0)) AS share_is_primary,
    (cnt_is_secondary::float     / NULLIF(n_osm_features,0)) AS share_is_secondary,
    (cnt_is_tertiary::float      / NULLIF(n_osm_features,0)) AS share_is_tertiary,
    (cnt_is_residential::float   / NULLIF(n_osm_features,0)) AS share_is_residential,
    (cnt_is_service::float       / NULLIF(n_osm_features,0)) AS share_is_service,
    (cnt_is_track_or_path::float / NULLIF(n_osm_features,0)) AS share_is_track_or_path,
    (cnt_is_foot_or_ped::float   / NULLIF(n_osm_features,0)) AS share_is_foot_or_ped
  FROM per_collision p
),

-- 2) One "dimension" row per collision (carry res_11, weather, etc.)
event_dim AS (
  SELECT DISTINCT ON (pair_id)
    pair_id,
    h3,
    res_11,
    borough,
    year,
    month,
    datetime,
    geometry,
    temp, dwpt, rhum, prcp, snow,
    wdir, wspd, wpgt, pres, tsun, coco, station_id
  FROM public.london_osm_weather
  ORDER BY
    pair_id,
    (ts IS NULL), ts DESC,
    (datetime IS NULL), datetime DESC
),

-- 3) res_11 context by averaging per-collision features within each res_11
res11_context AS (
  SELECT
    d.res_11,

    COUNT(*)::int                   AS r11_n_collisions,
    AVG(f.n_osm_features)::float    AS r11_mean_n_osm_features,
    SUM(f.n_osm_features)::int      AS r11_total_osm_features,

    /* numeric context (means across collisions in this res_11) */
    AVG(f.maxspeed_mph_avg)::float  AS r11_mean_maxspeed_mph,
    AVG(f.lanes_num_avg)::float     AS r11_mean_lanes,
    AVG(f.width_m_avg)::float       AS r11_mean_width_m,
    AVG(f.smoothness_score_avg)::float AS r11_mean_smoothness,

    /* safety-relevant means (counts per collision) */
    AVG(f.cnt_has_signals)::float       AS r11_mean_cnt_has_signals,
    AVG(f.cnt_has_crossing)::float      AS r11_mean_cnt_has_crossing,
    AVG(f.cnt_has_speed_camera)::float  AS r11_mean_cnt_speed_camera,
    AVG(f.cnt_has_bus_stop)::float      AS r11_mean_cnt_bus_stop,
    AVG(f.cnt_has_amenity)::float       AS r11_mean_cnt_amenity,

    /* road class mix (mean counts and shares) */
    AVG(f.cnt_is_primary)::float        AS r11_mean_cnt_is_primary,
    AVG(f.cnt_is_secondary)::float      AS r11_mean_cnt_is_secondary,
    AVG(f.cnt_is_tertiary)::float       AS r11_mean_cnt_is_tertiary,
    AVG(f.cnt_is_residential)::float    AS r11_mean_cnt_is_residential,
    AVG(f.cnt_is_service)::float        AS r11_mean_cnt_is_service,
    AVG(f.cnt_is_track_or_path)::float  AS r11_mean_cnt_is_track_or_path,

    AVG(f.share_is_primary)::float       AS r11_share_is_primary,
    AVG(f.share_is_secondary)::float     AS r11_share_is_secondary,
    AVG(f.share_is_tertiary)::float      AS r11_share_is_tertiary,
    AVG(f.share_is_residential)::float   AS r11_share_is_residential,
    AVG(f.share_is_service)::float       AS r11_share_is_service,
    AVG(f.share_is_track_or_path)::float AS r11_share_is_track_or_path,

    /* oneway mix (mean shares) */
    AVG( (f.cnt_oneway_forward::float)      / NULLIF(f.n_osm_features,0) ) AS r11_share_oneway_forward,
    AVG( (f.cnt_oneway_bidirectional::float)/ NULLIF(f.n_osm_features,0) ) AS r11_share_oneway_bidirectional,
    AVG( (f.cnt_oneway_reverse::float)      / NULLIF(f.n_osm_features,0) ) AS r11_share_oneway_reverse,

    /* sidewalks / cycling (mean counts per collision) */
    AVG(f.cnt_sidewalk_none)::float     AS r11_mean_cnt_sidewalk_none,
    AVG(f.cnt_sidewalk_both)::float     AS r11_mean_cnt_sidewalk_both,
    AVG(f.cnt_has_cycle_infra)::float   AS r11_mean_cnt_cycle_infra,
    AVG(f.cnt_has_cycle_lane)::float    AS r11_mean_cnt_cycle_lane,
    AVG(f.cnt_has_cycle_track)::float   AS r11_mean_cnt_cycle_track

  FROM per_collision_with_shares f
  JOIN event_dim d USING (pair_id)
  GROUP BY d.res_11
)

-- 4) Final one-row-per-collision result becomes the table
SELECT
  d.pair_id,
  d.h3,
  d.res_11,
  d.year,
  d.borough,
  d.datetime,
  d.geometry,

  /* ===== Derived datetime features (from d.datetime) ===== */
  EXTRACT(YEAR  FROM d.datetime)::int    AS dt_year,
  EXTRACT(MONTH FROM d.datetime)::int    AS dt_month,
  EXTRACT(DAY   FROM d.datetime)::int    AS dt_day,
  EXTRACT(HOUR  FROM d.datetime)::int    AS dt_hour,
  /* weekend/weekday as 0/1 flags; NULL if datetime is NULL */
  CASE
    WHEN d.datetime IS NULL THEN NULL
    WHEN EXTRACT(ISODOW FROM d.datetime) IN (6,7) THEN 1 ELSE 0
  END::smallint                           AS dt_is_weekend,
  CASE
    WHEN d.datetime IS NULL THEN NULL
    WHEN EXTRACT(ISODOW FROM d.datetime) BETWEEN 1 AND 5 THEN 1 ELSE 0
  END::smallint                           AS dt_is_weekday,

  d.temp, d.dwpt, d.rhum, d.prcp, d.snow,
  d.wdir, d.wspd, d.wpgt, d.pres, d.tsun, d.coco, d.station_id,

  f.n_osm_features,

  /* per-collision numeric */
  f.lanes_num_avg, f.lanes_num_max,
  f.width_m_avg,  f.width_m_max,
  f.incline_pc_avg, f.incline_abs_pc_avg,
  f.maxspeed_mph_avg, f.maxspeed_mph_max,
  f.n_left_lanes_avg, f.n_through_lanes_avg, f.n_right_lanes_avg,
  f.bus_routes_cnt_sum, f.bus_routes_cnt_max,
  f.smoothness_score_avg,

  /* per-collision boolean counts */
  f.cnt_is_motorway, f.cnt_is_trunk, f.cnt_is_primary, f.cnt_is_secondary, f.cnt_is_tertiary,
  f.cnt_is_residential, f.cnt_is_service, f.cnt_is_track_or_path, f.cnt_is_foot_or_ped,
  f.cnt_has_pt_stop, f.cnt_is_paved, f.cnt_is_nsl, f.cnt_speed_var_or_none,
  f.cnt_has_signals, f.cnt_has_traffic_calming, f.cnt_has_crossing,
  f.cnt_crossing_signalised, f.cnt_crossing_zebra,
  f.cnt_has_cycle_infra, f.cnt_has_contraflow_cycle, f.cnt_has_cycle_track, f.cnt_has_cycle_lane,
  f.cnt_is_lit, f.cnt_has_lit_schedule,
  f.cnt_motor_restricted, f.cnt_hgv_restricted, f.cnt_psv_priority, f.cnt_bus_priority,
  f.cnt_is_bridge, f.cnt_is_tunnel, f.cnt_has_barrier, f.cnt_has_amenity, f.cnt_has_bus_stop,
  f.cnt_has_mini_roundabout, f.cnt_has_speed_camera, f.cnt_overtaking_allowed,

  /* per-collision distributions & categorical counts */
  f.cnt_oneway_forward, f.cnt_oneway_bidirectional, f.cnt_oneway_reverse,
  f.cnt_surface_paved, f.cnt_surface_unpaved, f.cnt_surface_asphalt, f.cnt_surface_gravel, f.cnt_surface_dirt,
  f.cnt_sidewalk_both, f.cnt_sidewalk_left, f.cnt_sidewalk_right, f.cnt_sidewalk_none,
  f.cnt_bicycle_yes, f.cnt_bicycle_designated, f.cnt_bicycle_permissive, f.cnt_bicycle_no,
  f.cnt_access_permissive, f.cnt_access_destination, f.cnt_access_private, f.cnt_access_no,
  f.cnt_litmode_always, f.cnt_litmode_limited,

  /* per-collision shares */
  f.share_is_motorway, f.share_is_trunk, f.share_is_primary, f.share_is_secondary, f.share_is_tertiary,
  f.share_is_residential, f.share_is_service, f.share_is_track_or_path, f.share_is_foot_or_ped,

  /* res_11 context (wider-area features) */
  r11.r11_n_collisions,
  r11.r11_mean_n_osm_features,
  r11.r11_total_osm_features,

  r11.r11_mean_maxspeed_mph,
  r11.r11_mean_lanes,
  r11.r11_mean_width_m,
  r11.r11_mean_smoothness,

  r11.r11_mean_cnt_has_signals,
  r11.r11_mean_cnt_has_crossing,
  r11.r11_mean_cnt_speed_camera,
  r11.r11_mean_cnt_bus_stop,
  r11.r11_mean_cnt_amenity,

  r11.r11_mean_cnt_is_primary,
  r11.r11_mean_cnt_is_secondary,
  r11.r11_mean_cnt_is_tertiary,
  r11.r11_mean_cnt_is_residential,
  r11.r11_mean_cnt_is_service,
  r11.r11_mean_cnt_is_track_or_path,

  r11.r11_share_is_primary,
  r11.r11_share_is_secondary,
  r11.r11_share_is_tertiary,
  r11.r11_share_is_residential,
  r11.r11_share_is_service,
  r11.r11_share_is_track_or_path,

  r11.r11_share_oneway_forward,
  r11.r11_share_oneway_bidirectional,
  r11.r11_share_oneway_reverse,

  r11.r11_mean_cnt_sidewalk_none,
  r11.r11_mean_cnt_sidewalk_both,
  r11.r11_mean_cnt_cycle_infra,
  r11.r11_mean_cnt_cycle_lane,
  r11.r11_mean_cnt_cycle_track

FROM event_dim d
JOIN per_collision_with_shares f
  ON d.pair_id = f.pair_id
LEFT JOIN res11_context r11
  ON d.res_11 = r11.res_11;
