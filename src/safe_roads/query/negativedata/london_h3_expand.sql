ALTER TABLE public.london_h3
  ADD COLUMN IF NOT EXISTS res_11 h3index
  GENERATED ALWAYS AS (
    CASE WHEN h3 IS NULL THEN NULL ELSE h3_cell_to_parent(h3::h3index, 11) END
  ) STORED;


CREATE INDEX IF NOT EXISTS london_h3_res_11_idx ON public.london_h3 (res_11);
