CREATE OR REPLACE FUNCTION public.normalize_london_borough(b TEXT)
RETURNS TEXT
LANGUAGE sql IMMUTABLE AS $$
SELECT CASE
  WHEN b IS NULL OR b ~ '^\s*$' THEN NULL
  WHEN lower(b) IN ('barking & dagenham','barking and dagenham','barking &amp; dagenham') THEN 'Barking and Dagenham'
  WHEN lower(b) = 'barnet' THEN 'Barnet'
  WHEN lower(b) = 'bexley' THEN 'Bexley'
  WHEN lower(b) = 'brent' THEN 'Brent'
  WHEN lower(b) = 'bromley' THEN 'Bromley'
  WHEN lower(b) = 'camden' THEN 'Camden'
  WHEN lower(b) IN ('city of london','city of  london') THEN 'City of London'
  WHEN lower(b) = 'croydon' THEN 'Croydon'
  WHEN lower(b) = 'ealing' THEN 'Ealing'
  WHEN lower(b) = 'enfield' THEN 'Enfield'
  WHEN lower(b) = 'greenwich' THEN 'Greenwich'
  WHEN lower(b) = 'hackney' THEN 'Hackney'
  WHEN lower(b) IN ('hammersmith & fulham','hammersmith and fulham') THEN 'Hammersmith and Fulham'
  WHEN lower(b) = 'haringey' THEN 'Haringey'
  WHEN lower(b) = 'harrow' THEN 'Harrow'
  WHEN lower(b) = 'havering' THEN 'Havering'
  WHEN lower(b) = 'hillingdon' THEN 'Hillingdon'
  WHEN lower(b) = 'hounslow' THEN 'Hounslow'
  WHEN lower(b) = 'islington' THEN 'Islington'
  WHEN lower(b) IN ('kensington & chelsea','kensington and chelsea') THEN 'Kensington and Chelsea'
  WHEN lower(b) IN ('kingston-upon-thames','kingston upon thames','kingston-upon thames','kingston upon-thames') THEN 'Kingston upon Thames'
  WHEN lower(b) = 'lambeth' THEN 'Lambeth'
  WHEN lower(b) = 'lewisham' THEN 'Lewisham'
  WHEN lower(b) = 'merton' THEN 'Merton'
  WHEN lower(b) = 'newham' THEN 'Newham'
  WHEN lower(b) = 'redbridge' THEN 'Redbridge'
  WHEN lower(b) IN ('richmond-upon-thames','richmond upon thames','richmond-upon thames','richmond upon-thames') THEN 'Richmond upon Thames'
  WHEN lower(b) = 'southwark' THEN 'Southwark'
  WHEN lower(b) = 'sutton' THEN 'Sutton'
  WHEN lower(b) = 'tower hamlets' THEN 'Tower Hamlets'
  WHEN lower(b) = 'waltham forest' THEN 'Waltham Forest'
  WHEN lower(b) = 'wandsworth' THEN 'Wandsworth'
  WHEN lower(b) = 'westminster' THEN 'Westminster'
  ELSE initcap(regexp_replace(b, '&', 'and', 'gi'))
END
$$;

ALTER TABLE public.osm_collision_weather
ADD COLUMN borough_norm TEXT
  GENERATED ALWAYS AS (public.normalize_london_borough("Borough")) STORED;
