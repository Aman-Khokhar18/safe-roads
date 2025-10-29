-- Rebuild ML features table from OSM H3 cells (keeps year/month untouched)
-- Speeds kept in **mph** and we DROP all `miss_` columns (leave NULLs as-is).
DROP TABLE IF EXISTS public.osm_latest_ml_features;

CREATE TABLE public.osm_latest_ml_features AS
WITH base AS (
  SELECT
    h3,
    "geometry"        AS geometry,   -- carry original geometry
    lower(btrim(highway))           AS highway_raw,
    lower(btrim(lanes))             AS lanes_raw,
    lower(btrim(width))             AS width_raw,
    lower(btrim(surface))           AS surface_raw,
    lower(btrim(smoothness))        AS smoothness_raw,
    lower(btrim(oneway))            AS oneway_raw,
    lower(btrim(maxspeed))          AS maxspeed_raw,
    lower(btrim(traffic_signals))   AS traffic_signals_raw,
    lower(btrim(traffic_calming))   AS traffic_calming_raw,
    lower(btrim(crossing))          AS crossing_raw,
    lower(btrim(sidewalk))          AS sidewalk_raw,
    lower(btrim(cycleway))          AS cycleway_raw,
    lower(btrim(bicycle))           AS bicycle_raw,
    lower(btrim(lit))               AS lit_raw,
    lower(btrim("turn:lanes"))      AS turn_lanes_raw,
    lower(btrim(access))            AS access_raw,
    lower(btrim(vehicle))           AS vehicle_raw,
    lower(btrim(hgv))               AS hgv_raw,
    lower(btrim(psv))               AS psv_raw,
    lower(btrim(bus))               AS bus_raw,
    lower(btrim(overtaking))        AS overtaking_raw,
    lower(btrim(bridge))            AS bridge_raw,
    lower(btrim(tunnel))            AS tunnel_raw,
    lower(btrim(layer))             AS layer_raw,
    lower(btrim(incline))           AS incline_raw,
    lower(btrim(barrier))           AS barrier_raw,
    lower(btrim(amenity))           AS amenity_raw,
    year,
    month
  FROM public.osm_latest_h3
),

norm AS (
  SELECT
    *,

    /* Canonical highway class */
    CASE
      WHEN highway_raw IN ('motorway','motorway_link') THEN 'motorway'
      WHEN highway_raw IN ('trunk','trunk_link') THEN 'trunk'
      WHEN highway_raw IN ('primary','primary_link') THEN 'primary'
      WHEN highway_raw IN ('secondary','secondary_link') THEN 'secondary'
      WHEN highway_raw IN ('tertiary','tertiary_link') THEN 'tertiary'
      WHEN highway_raw IN ('residential','living_street','unclassified') THEN 'residential'
      WHEN highway_raw IN ('service','services','ser') THEN 'service'
      WHEN highway_raw IN ('footway','steps','corridor','pedestrian') THEN 'foot/ped'
      WHEN highway_raw IN ('cycleway','busway','opposite_lane','opposite_track','share_busway','shared_busway') THEN 'cycle/bus'
      WHEN highway_raw IN ('track','path','bridleway','byway','unsurfaced') THEN 'track/path'
      WHEN highway_raw IN ('construction','proposed','demolished','disused') THEN 'nonactive'
      WHEN highway_raw IN ('bus_stop','bus_stand','bus_station','platform','stop_position') THEN 'pt_stop'
      WHEN highway_raw IN ('mini_roundabout','junction') THEN 'junction'
      WHEN highway_raw IN ('raceway') THEN 'raceway'
      ELSE 'other'
    END AS highway_class,

    /* lanes -> numeric (split on ;, whitespace, or |) */
    (
      WITH toks AS (
        SELECT unnest(
          regexp_split_to_array(
            coalesce(lanes_raw,''),
            '[[:space:]]*;[[:space:]]*|[[:space:]]+|[|]'
          )
        ) AS t
      )
      SELECT
        CASE
          WHEN EXISTS (SELECT 1 FROM toks WHERE t ~ '^[0-9]+(\.[0-9]+)?$')
          THEN (SELECT avg((t)::numeric) FROM toks WHERE t ~ '^[0-9]+(\.[0-9]+)?$')
          ELSE NULL
        END
    )::numeric AS lanes_num,

    /* width -> meters (handles "4 m", "1,5 m", "6'6\"", "1..5") */
    CASE
      WHEN width_raw ~ '^[[:space:]]*[0-9]+([.,][0-9]+)?[[:space:]]*(m|metre|meter|metres|meters)?[[:space:]]*$'
        THEN (regexp_replace(regexp_replace(width_raw, ',', '.', 'g'), '[^0-9\.]', '', 'g'))::numeric
      WHEN width_raw ~ '^[[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]*$'
        THEN width_raw::numeric
      WHEN width_raw ~ '^[[:space:]]*([0-9]+)''[[:space:]]*([0-9]+)"?[[:space:]]*$'
        THEN (
          (regexp_replace(width_raw, '^[[:space:]]*([0-9]+)''[[:space:]]*([0-9]+)".*$', E'\\1')::numeric * 0.3048) +
          (regexp_replace(width_raw, '^[[:space:]]*([0-9]+)''[[:space:]]*([0-9]+)".*$', E'\\2')::numeric * 0.0254)
        )
      WHEN width_raw ~ '^[0-9]+\.\.[0-9]+$'
        THEN (regexp_replace(width_raw, '\.\.', '.', 'g'))::numeric
      ELSE NULL
    END AS width_m,

    /* surface collapse + paved flag */
    CASE
      WHEN surface_raw ~ '(asphalt|tarmac)' THEN 'asphalt'
      WHEN surface_raw ~ 'concrete' THEN 'concrete'
      WHEN surface_raw ~ '(paving_stones|slab|block|sett|cobblestone|setts?)' THEN 'paving'
      WHEN surface_raw ~ 'gravel' THEN 'gravel'
      WHEN surface_raw ~ '(dirt|mud|earth|ground|sand)' THEN 'dirt'
      WHEN surface_raw ~ 'grass' THEN 'grass'
      WHEN surface_raw ~ '(wood|timber)' THEN 'wood'
      WHEN surface_raw ~ '(metal|steel)' THEN 'metal'
      WHEN surface_raw IS NULL OR surface_raw IN ('no','yes') THEN 'unknown'
      ELSE 'other'
    END AS surface_cat,
    CASE
      WHEN surface_raw ~ '(asphalt|tarmac|concrete|paving_stones|slab|block|sett|cobblestone|setts?)' THEN 1
      WHEN surface_raw ~ '(unpaved|gravel|dirt|earth|grass|mud|sand)' THEN 0
      ELSE NULL
    END AS is_paved,

    /* smoothness -> ordinal */
    CASE smoothness_raw
      WHEN 'excellent'     THEN 5
      WHEN 'good'          THEN 4
      WHEN 'intermediate'  THEN 3
      WHEN 'bad'           THEN 2
      WHEN 'very_bad'      THEN 1
      WHEN 'horrible'      THEN 0
      WHEN 'very_horrible' THEN -1
      WHEN 'impassable'    THEN -2
      ELSE NULL
    END AS smoothness_score,

    /* oneway -> code */
    CASE
      WHEN oneway_raw IN ('yes','1')       THEN 1
      WHEN oneway_raw IN ('-1','opposite') THEN -1
      WHEN oneway_raw = 'no'               THEN 0
      WHEN oneway_raw IN ('alternating','reversible') THEN 2
      ELSE NULL
    END AS oneway_code,

    /* maxspeed -> **MPH** + flags (UK context) */
    CASE
      WHEN maxspeed_raw ~ '^[0-9]+$' THEN (maxspeed_raw::numeric)                         -- bare number = mph
      WHEN maxspeed_raw ~ 'mph'      THEN (regexp_replace(maxspeed_raw,'[^0-9\.]','','g')::numeric)
      WHEN maxspeed_raw IN ('walk')  THEN 3                                               -- ~3 mph
      ELSE NULL
    END AS maxspeed_mph,
    (maxspeed_raw = 'national')                     AS is_national_speed_limit,
    (maxspeed_raw IN ('variable','signals','none')) AS is_variable_or_none,

    /* control / calming / crossings */
    (traffic_signals_raw IN ('yes','signal','signals','traffic_lights','crossing','junction','forward','pedestrian_crossing')) AS has_signals,
    (traffic_calming_raw IS NOT NULL AND traffic_calming_raw NOT IN ('no','none')) AS has_traffic_calming,
    (crossing_raw IS NOT NULL AND crossing_raw NOT IN ('no','unknown')) AS has_crossing,
    (crossing_raw ~ '(pelican|puffin|toucan|signals)') AS crossing_signalised,
    (crossing_raw ~ 'zebra')                           AS crossing_zebra,

    /* sidewalks / cycleways / bicycle access */
    CASE
      WHEN sidewalk_raw IN ('both','both;right','right;both','left;right','left;both','both; separate','both;separate') THEN 'both'
      WHEN sidewalk_raw ~ '^left'    THEN 'left'
      WHEN sidewalk_raw ~ '^right'   THEN 'right'
      WHEN sidewalk_raw ~ 'separate' THEN 'separate'
      WHEN sidewalk_raw IN ('no','none') THEN 'none'
      WHEN sidewalk_raw = 'yes'      THEN 'present'
      ELSE 'unknown'
    END AS sidewalk_cat,
    (cycleway_raw IS NOT NULL AND cycleway_raw NOT IN ('no','none')) AS has_cycle_infra,
    (cycleway_raw ~ 'opposite')  AS has_contraflow_cycle,
    (cycleway_raw ~ 'track')     AS has_cycle_track,
    (cycleway_raw ~ 'lane')      AS has_cycle_lane,
    CASE
      WHEN bicycle_raw IN ('no')                                THEN 'no'
      WHEN bicycle_raw IN ('designated','yes','permit','permissive') THEN 'allowed'
      WHEN bicycle_raw IN ('private')                           THEN 'private'
      ELSE 'unknown'
    END AS bicycle_access,

    /* lighting: passthrough flags (no datetime logic yet) */
    CASE
      WHEN lit_raw IN ('yes','24/7','sunset-sunrise','automatic','limited','mo-su dusk-dawn') THEN 1
      WHEN lit_raw = 'no' THEN 0
      ELSE NULL
    END AS is_lit,
    CASE
      WHEN lit_raw IN ('yes','24/7','automatic') THEN 'always_on'
      WHEN lit_raw IN ('sunset-sunrise','mo-su dusk-dawn') THEN 'dusk_dawn'
      WHEN lit_raw ~ '^[[:digit:]]{1,2}:[[:digit:]]{2}[[:space:]]*-[[:space:]]*[[:digit:]]{1,2}:[[:digit:]]{2}$' THEN 'timed'
      WHEN position(';' in coalesce(lit_raw,'')) > 0 THEN 'mixed'
      WHEN lit_raw = 'no' THEN 'none'
      ELSE 'unknown'
    END AS lit_mode,
    CASE
      WHEN lit_raw IN ('sunset-sunrise','mo-su dusk-dawn') THEN TRUE
      WHEN lit_raw ~ '^[[:digit:]]{1,2}:[[:digit:]]{2}[[:space:]]*-[[:space:]]*[[:digit:]]{1,2}:[[:digit:]]{2}$' THEN TRUE
      ELSE FALSE
    END AS has_lit_schedule,

    /* access + vehicle mix */
    CASE
      WHEN access_raw IN ('private','staff','members','residents','official') THEN 'private'
      WHEN access_raw IN ('destination','customers','customer')               THEN 'destination'
      WHEN access_raw IN ('permissive','permit','public','yes')               THEN 'permitted'
      WHEN access_raw = 'no'                                                 THEN 'no'
      ELSE 'unknown'
    END AS access_cat,

    (
      coalesce(vehicle_raw,'') ~* '(^|[^[:alnum:]_])no([^[:alnum:]_]|$)'
      OR coalesce(psv_raw,'')  ~* '(^|[^[:alnum:]_])only([^[:alnum:]_]|$)'
      OR coalesce(bus_raw,'')  ~* '(^|[^[:alnum:]_])only([^[:alnum:]_]|$)'
    ) AS motor_restricted,

    (coalesce(hgv_raw,'') ~* 'no|private|discouraged') AS hgv_restricted,
    (coalesce(psv_raw,'') ~* 'yes|designated|only|opposite_lane') AS psv_priority,
    (coalesce(bus_raw,'') ~* 'only|share_busway|shared_busway|opposite_lane') AS bus_priority,

    /* structure / level */
    (bridge_raw IN ('yes','covered','viaduct','suspension','boardwalk','pier','lift','gangway','building_passage')) AS is_bridge,
    (tunnel_raw IN ('yes','covered','culvert','subway','passage','building_passage'))                               AS is_tunnel,
    CASE WHEN layer_raw ~ '^-?[0-9]+(\.[0-9]+)?$' THEN layer_raw::numeric ELSE NULL END AS layer_num,

    /* incline: signed % and absolute % */
    CASE
      WHEN incline_raw ~ '^-?[0-9]+%$' THEN regexp_replace(incline_raw,'%','')::numeric
      WHEN incline_raw ~ '^-?[0-9]+°$' THEN (tan(radians(regexp_replace(incline_raw,'°','')::numeric)) * 100.0)
      WHEN incline_raw IN ('up','up-','up/down') THEN 5
      WHEN incline_raw IN ('down','doen','own')  THEN -5
      WHEN incline_raw IN ('steep','yes')        THEN 10
      WHEN incline_raw IN ('flat','0','0%')      THEN 0
      ELSE NULL
    END AS incline_pc,
    ABS(
      CASE
        WHEN incline_raw ~ '^-?[0-9]+%$' THEN regexp_replace(incline_raw,'%','')::numeric
        WHEN incline_raw ~ '^-?[0-9]+°$' THEN (tan(radians(regexp_replace(incline_raw,'°','')::numeric)) * 100.0)
        ELSE NULL
      END
    ) AS incline_abs_pc,

    /* barriers / amenities */
    (barrier_raw IS NOT NULL AND barrier_raw <> 'no') AS has_barrier,
    (amenity_raw IS NOT NULL)                          AS has_amenity,

    /* turn:lanes counts (normalize repeated pipes) */
    (
      SELECT count(*) FILTER (WHERE x IN ('left','slight_left'))
      FROM unnest(
        regexp_split_to_array(
          coalesce(regexp_replace(turn_lanes_raw,'[|]{2,}','|','g'),''),
          '\|'
        )
      ) AS t(x)
    ) AS n_left_lanes,
    (
      SELECT count(*) FILTER (WHERE x IN ('through'))
      FROM unnest(
        regexp_split_to_array(
          coalesce(regexp_replace(turn_lanes_raw,'[|]{2,}','|','g'),''),
          '\|'
        )
      ) AS t(x)
    ) AS n_through_lanes,
    (
      SELECT count(*) FILTER (WHERE x IN ('right','slight_right'))
      FROM unnest(
        regexp_split_to_array(
          coalesce(regexp_replace(turn_lanes_raw,'[|]{2,}','|','g'),''),
          '\|'
        )
      ) AS t(x)
    ) AS n_right_lanes,

    /* bus stop / camera / overtaking */
    (highway_raw IN ('bus_stop','bus_stand','bus_station','platform','stop_position')) AS has_bus_stop,
    (highway_raw = 'mini_roundabout')  AS has_mini_roundabout,
    (highway_raw IN ('speed_camera','speed_display')) AS has_speed_camera,
    CASE
      WHEN overtaking_raw = 'yes' THEN 1
      WHEN overtaking_raw = 'no'  THEN 0
      ELSE NULL
    END AS overtaking_allowed,

    /* rough bus routes count from "54; 89; 108" etc. */
    (
      WITH br AS (
        SELECT regexp_split_to_table(coalesce(bus_raw,''), '[;,[:space:]]+') AS tok
      )
      SELECT count(*) FROM br WHERE tok ~ '^[0-9]+[a-zA-Z]?$'
    ) AS bus_routes_cnt

  FROM base
)

SELECT
  h3,
  geometry,            -- include geometry from source
  highway_class,
  (highway_class='motorway')::int     AS is_motorway,
  (highway_class='trunk')::int        AS is_trunk,
  (highway_class='primary')::int      AS is_primary,
  (highway_class='secondary')::int    AS is_secondary,
  (highway_class='tertiary')::int     AS is_tertiary,
  (highway_class='residential')::int  AS is_residential,
  (highway_class='service')::int      AS is_service,
  (highway_class='track/path')::int   AS is_track_or_path,
  (highway_class='foot/ped')::int     AS is_foot_or_ped,
  (highway_class='pt_stop')::int      AS has_pt_stop,

  lanes_num,
  width_m,

  is_paved, surface_cat,
  smoothness_score,

  oneway_code,

  -- speeds in **mph**
  maxspeed_mph,
  is_national_speed_limit, is_variable_or_none,

  has_signals, has_traffic_calming,
  has_crossing, crossing_signalised, crossing_zebra,

  sidewalk_cat,
  has_cycle_infra, has_contraflow_cycle, has_cycle_track, has_cycle_lane,
  bicycle_access,

  -- lighting passthrough
  is_lit,
  lit_mode,
  has_lit_schedule,

  access_cat, motor_restricted, hgv_restricted, psv_priority, bus_priority,

  is_bridge, is_tunnel, layer_num,
  incline_pc, incline_abs_pc,

  has_barrier, has_amenity, has_bus_stop, has_mini_roundabout, has_speed_camera,
  overtaking_allowed,
  n_left_lanes, n_through_lanes, n_right_lanes,
  bus_routes_cnt,

  -- keep datetime cols untouched
  year,
  month
FROM norm;


