-- 2) Delete duplicate rows, keeping the first (arbitrary) row in each set
WITH dupes AS (
  SELECT
    ctid,
    ROW_NUMBER() OVER (
      PARTITION BY
        h3, neighbor_1, neighbor_2, neighbor_3, neighbor_4, neighbor_5, neighbor_6,
        parent_h3, highway, name, lanes, surface, oneway, junction, traffic_signals,
        traffic_calming, crossing, sidewalk, cycleway, bicycle, lit, access, vehicle,
        hgv, psv, bus, overtaking, bridge, tunnel, layer, incline, barrier, amenity,
        road_class3, borough, width, smoothness, temp, dwpt, rhum, prcp, snow, wdir,
        wspd, wpgt, pres, tsun, coco, year, month, day, hour, is_weekend, maxspeed_mph,
        junction_degree, is_junction, is_turn, hour_sin, hour_cos, dow_sin, dow_cos,
        dom_sin, dom_cos, month_sin, month_cos, traffic_light_count, crossing_count,
        motorway_other_count, cycleway_count
      ORDER BY ctid
    ) AS rn
  FROM public.roads_features_collision
)
DELETE FROM public.roads_features_collision t
USING dupes d
WHERE t.ctid = d.ctid
  AND d.rn > 1;
