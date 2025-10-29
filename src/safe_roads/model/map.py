# pip install folium h3 pandas pyarrow branca

import json, gzip, base64
import numpy as np
import pandas as pd
import folium
import h3
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template

# ========= Config =========
INPUT_PARQUET   = "predictions.parquet"
OUTPUT_HTML     = "h3_probability_dynamic_view.html"
PROB_COL        = "probability"
H3_COL          = "h3"

START_ZOOM      = 16
MIN_DRAW_ZOOM   = 7
MAX_RENDERED    = 5000           # cap displayed shapes
FILL_OPACITY    = 0.30
TILES           = "cartodbpositron"

DRAW_BORDERS    = False
BOUNDS_PADDING_FACTOR = 0.12

# ---- Threshold ----
PROB_THRESHOLD  = 0.50           # only show cells with p >= this value

# ---- Gradient/legend controls ----
AUTO_SCALE        = True
SCALE_MIN         = 0.0
SCALE_MAX         = 1.0
GRADIENT_COLORS   = ["#00a651", "#ffd400", "#ff0033"]
LEGEND_CAPTION    = "Probability"

# ---- Performance knobs ----
ENABLE_GZIP_EMBED = True
QUANTIZE_LEVELS   = 256          # 8-bit probs → tiny payload
CANVAS_PADDING    = 0.5          # Leaflet canvas renderer padding
DRAW_CHUNK_SIZE   = 1500         # how many shapes per rAF batch

# Optional: use circles at low zoom for speed
CIRCLES_WHEN_ZOOMED_OUT = True
CIRCLE_ZOOM_MAX   = 12           # <= this zoom, draw circles (faster)
CIRCLE_PIX_RADIUS = 6            # circle radius in pixels
# ==================================

# -------- Load & prep --------
df = pd.read_parquet(INPUT_PARQUET)

p = df[PROB_COL].astype(float)
if p.max() > 1.0:
    p = p / 100.0
df["p"] = p.clip(0.0, 1.0)

if AUTO_SCALE:
    SCALE_MIN = float(df["p"].min())
    SCALE_MAX = float(df["p"].max())
    if SCALE_MIN == SCALE_MAX:
        SCALE_MIN = max(0.0, SCALE_MIN - 0.01)
        SCALE_MAX = min(1.0, SCALE_MAX + 0.01)

# Map center = mean centroid
latlons = df[H3_COL].apply(h3.cell_to_latlng)
center_lat = float(np.mean([lat for lat, _ in latlons]))
center_lon = float(np.mean([lon for _, lon in latlons]))

# -------- Base map --------
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=START_ZOOM,
    tiles=TILES,
    prefer_canvas=True,  # global hint; we’ll still pass an explicit canvas renderer
)

# Legend
cmap = LinearColormap(colors=GRADIENT_COLORS, vmin=SCALE_MIN, vmax=SCALE_MAX)
cmap.caption = f"{LEGEND_CAPTION} ({SCALE_MIN} → {SCALE_MAX})"
cmap.add_to(m)

# ---- Build compact payload ----
if QUANTIZE_LEVELS:
    levels = int(QUANTIZE_LEVELS)
    q = np.clip(
        np.round((df["p"] - SCALE_MIN) / max(1e-9, (SCALE_MAX - SCALE_MIN)) * (levels - 1)),
        0, levels - 1
    ).astype(np.uint16)
    records = list(map(list, zip(df[H3_COL].astype(str).tolist(), q.tolist())))
    payload = json.dumps({"levels": levels, "min": SCALE_MIN, "max": SCALE_MAX, "data": records}, separators=(",", ":"))
else:
    records = list(map(list, zip(df[H3_COL].astype(str).tolist(), df["p"].astype(float).tolist())))
    payload = json.dumps({"levels": None, "min": SCALE_MIN, "max": SCALE_MAX, "data": records}, separators=(",", ":"))

if ENABLE_GZIP_EMBED:
    compressed_b64 = base64.b64encode(gzip.compress(payload.encode("utf-8"))).decode("ascii")
    data_expr = f'"{compressed_b64}"'
    data_is_gz = True
else:
    data_expr = payload
    data_is_gz = False

map_ref = m.get_name()
colors_json = json.dumps(GRADIENT_COLORS)

tmpl = Template(f"""
{{% macro header(this, kwargs) %}}
  <script src="https://unpkg.com/h3-js@4.1.0/dist/h3-js.umd.js"></script>
  {'<script src="https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js"></script>' if ENABLE_GZIP_EMBED else ''}
  <style>
    .h3-warn {{
      position:absolute; top:10px; left:50%; transform:translateX(-50%);
      background:#ffefc1; color:#8a6d3b; padding:6px 10px; border:1px solid #e0c97f; border-radius:4px;
      font-family:sans-serif; font-size:12px; z-index:9999; display:none;
    }}
    .h3-badge {{
      position:absolute; bottom:12px; right:12px;
      background:rgba(0,0,0,0.65); color:#fff; padding:6px 8px; border-radius:6px;
      font-family:sans-serif; font-size:12px; z-index:9999;
    }}
  </style>
{{% endmacro %}}

{{% macro script(this, kwargs) %}}
  const map = {map_ref};

  // Explicit canvas renderer (faster than SVG for many paths)
  const canvasRenderer = L.canvas({{padding:{CANVAS_PADDING}}});

  // ---- Load embedded data ----
  const EMBED_IS_GZ = {str(data_is_gz).lower()};
  const EMBED = {data_expr};

  function b64ToUint8(b64) {{
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i=0; i<bin.length; i++) arr[i] = bin.charCodeAt(i);
    return arr;
  }}
  function loadData() {{
    if (!EMBED_IS_GZ) return JSON.parse(EMBED);
    const u8 = b64ToUint8(EMBED);
    const inflated = (typeof pako !== 'undefined') ? pako.ungzip(u8) : u8;
    const text = new TextDecoder().decode(inflated);
    return JSON.parse(text);
  }}
  const packed = loadData(); // {{levels, min, max, data: [[h,q|p], ...]}}
  const LEVELS = packed.levels;
  const SCALE_MIN = packed.min;
  const SCALE_MAX = packed.max;
  const rawData = packed.data;

  // Accessor with dequantization
  function getP(v) {{
    if (LEVELS) return SCALE_MIN + (v / (LEVELS - 1)) * (SCALE_MAX - SCALE_MIN);
    return v;
  }}

  // ---- Config ----
  const PROB_THRESHOLD = {PROB_THRESHOLD};
  const GRADIENT = {colors_json};
  const CIRCLES_WHEN_ZOOMED_OUT = {str(CIRCLES_WHEN_ZOOMED_OUT).lower()};
  const CIRCLE_ZOOM_MAX = {CIRCLE_ZOOM_MAX};

  const warn = L.DomUtil.create('div', 'h3-warn', map.getContainer());
  warn.textContent = "h3-js not available.";
  function showWarn(s) {{ warn.style.display = s ? 'block' : 'none'; }}

  const badge = L.DomUtil.create('div','h3-badge', map.getContainer());
  badge.innerHTML = `Threshold: <b>${{PROB_THRESHOLD.toFixed(2)}}</b> &nbsp; Shown: <b>0</b>`;

  const layerRoot = L.layerGroup().addTo(map);

  // ---- Color ramp helpers ----
  function hexToRgb(hex) {{
    const h = hex.replace('#','');
    const n = h.length === 3 ? h.split('').map(c=>c+c).join('') : h;
    const bigint = parseInt(n, 16);
    return {{ r: (bigint>>16)&255, g: (bigint>>8)&255, b: bigint&255 }};
  }}
  function lerp(a,b,t) {{ return a + (b-a)*t; }}
  function lerpRgb(a,b,t) {{
    return {{
      r: Math.round(lerp(a.r,b.r,t)),
      g: Math.round(lerp(a.g,b.g,t)),
      b: Math.round(lerp(a.b,b.b,t))
    }};
  }}
  const GRADIENT_RGB = GRADIENT.map(hexToRgb);
  function valueToColor(v) {{
    const min = SCALE_MIN, max = (SCALE_MAX===SCALE_MIN) ? SCALE_MIN+1e-6 : SCALE_MAX;
    let t = (v - min) / (max - min);
    t = Math.max(0, Math.min(1, t));
    const n = GRADIENT_RGB.length;
    if (n === 1) return `rgb(${{GRADIENT_RGB[0].r}},${{GRADIENT_RGB[0].g}},${{GRADIENT_RGB[0].b}})`;
    const seg = Math.min(n-2, Math.floor(t*(n-1)));
    const localT = (t*(n-1)) - seg;
    const c = lerpRgb(GRADIENT_RGB[seg], GRADIENT_RGB[seg+1], localT);
    return `rgb(${{c.r}}, ${{c.g}}, ${{c.b}})`;
  }}

  function debounce(fn, ms) {{ let t; return (...args)=>{{ clearTimeout(t); t=setTimeout(()=>fn(...args), ms); }}; }}
  function getPaddedBounds(b) {{
    const latPad = (b.getNorth() - b.getSouth()) * {BOUNDS_PADDING_FACTOR};
    const lngPad = (b.getEast() - b.getWest()) * {BOUNDS_PADDING_FACTOR};
    return L.latLngBounds(
      L.latLng(b.getSouth()-latPad, b.getWest()-lngPad),
      L.latLng(b.getNorth()+latPad, b.getEast()+lngPad)
    );
  }}

  // ---- Caches to avoid recomputation and DOM churn ----
  const boundaryCache = new Map();   // h3id -> [[lat,lng]...]
  const centerCache   = new Map();   // h3id -> [lat,lng]
  const shapePool     = new Map();   // h3id -> Leaflet layer (polygon or circle)

  const hoverTooltip = L.tooltip({{sticky:true}});
  let lastShape = null;

  function getBoundary(h) {{
    let b = boundaryCache.get(h);
    if (b) return b;
    try {{
      b = h3.cellToBoundary(h).map(([lat,lng]) => [lat,lng]);
      boundaryCache.set(h, b);
      return b;
    }} catch {{
      return null;
    }}
  }}
  function getCenter(h) {{
    let c = centerCache.get(h);
    if (c) return c;
    try {{
      const cc = h3.cellToLatLng(h);
      c = [cc[0], cc[1]];
      centerCache.set(h, c);
      return c;
    }} catch {{
      return null;
    }}
  }}
  function inView(centerLL, padded) {{
    return padded.contains(centerLL);
  }}

  // Incremental draw in animation frames for responsiveness
  function drawInView() {{
    if (typeof h3 === 'undefined' || !h3.cellToLatLng) {{
      console.error("h3-js not available");
      showWarn(true);
      return;
    }}
    showWarn(false);

    const z = map.getZoom();
    if (z < {MIN_DRAW_ZOOM}) {{
      // Hide everything but keep in pool for reuse later
      shapePool.forEach(l => l.remove());
      badge.innerHTML = `Threshold: <b>${{PROB_THRESHOLD.toFixed(2)}}</b> &nbsp; Shown: <b>0</b>`;
      return;
    }}
    badge.innerHTML = `Threshold: <b>${{PROB_THRESHOLD.toFixed(2)}}</b> &nbsp; Shown: <b>…</b>`;

    const padded = getPaddedBounds(map.getBounds());
    const useCircles = {str(CIRCLES_WHEN_ZOOMED_OUT).lower()} && z <= {CIRCLE_ZOOM_MAX};

    // Build list of targets that pass the threshold
    const targets = [];
    for (let i=0; i<rawData.length; i++) {{
      const h = rawData[i][0];
      const v = rawData[i][1];
      const p = getP(v);
      if (p < PROB_THRESHOLD) continue;

      const c = getCenter(h);
      if (!c) continue;
      const ll = L.latLng(c[0], c[1]);
      if (!inView(ll, padded)) continue;

      targets.push([h, p, [c[0], c[1]]]);
      if (targets.length >= {MAX_RENDERED}) break;
    }}

    // Hide shapes that are no longer in view/passing threshold
    const keep = new Set(targets.map(t => t[0]));
    shapePool.forEach((layer, key) => {{
      if (!keep.has(key)) layer.remove();
    }});

    if (targets.length === 0) {{
      badge.innerHTML = `Threshold: <b>${{PROB_THRESHOLD.toFixed(2)}}</b> &nbsp; Shown: <b>0</b>`;
      return;
    }}

    // Draw/update in chunks
    let shown = 0;
    let idx = 0;
    function step() {{
      const limit = Math.min(idx + {DRAW_CHUNK_SIZE}, targets.length);
      for (; idx < limit; idx++) {{
        const [id, pVal, center] = targets[idx];
        const color = valueToColor(pVal);

        let layer = shapePool.get(id);
        if (!layer) {{
          if (useCircles) {{
            // circles are much faster at low zoom
            layer = L.circleMarker(center, {{
              renderer: canvasRenderer,
              radius: {CIRCLE_PIX_RADIUS},
              fill: true,
              fillOpacity: {FILL_OPACITY},
              fillColor: color,
              stroke: {str(DRAW_BORDERS).lower()},
              weight: {0 if not DRAW_BORDERS else 1}
            }});
          }} else {{
            const boundary = getBoundary(id);
            if (!boundary) continue;
            layer = L.polygon(boundary, {{
              renderer: canvasRenderer,
              fill: true,
              fillOpacity: {FILL_OPACITY},
              fillColor: color,
              stroke: {str(DRAW_BORDERS).lower()},
              weight: {0 if not DRAW_BORDERS else 1}
            }});
          }}

          // Lazy tooltip
          layer.on('mouseover', () => {{
            hoverTooltip.setContent(`H3: ${{id}}<br>p: ${{pVal.toFixed(3)}}`);
            const pos = layer.getBounds ? layer.getBounds().getCenter() : layer.getLatLng();
            hoverTooltip.setLatLng(pos);
            map.openTooltip(hoverTooltip);
            lastShape = layer;
          }});
          layer.on('mouseout', () => {{
            if (lastShape === layer) map.closeTooltip(hoverTooltip);
          }});

          layer.addTo(layerRoot);
          shapePool.set(id, layer);
        }} else {{
          // Update style/geometry without recreating
          if (useCircles && layer.setLatLng) {{
            layer.setLatLng(center);
          }} else if (!useCircles && layer.setLatLngs) {{
            if (!layer.getLatLngs || layer.getLatLngs().length === 0) {{
              const boundary = getBoundary(id);
              if (boundary) layer.setLatLngs([boundary]);
            }}
          }}
          layer.setStyle({{fillColor: color}});
          if (!layer._map) layer.addTo(layerRoot);
        }}
        shown++;
      }}
      badge.innerHTML = `Threshold: <b>${{PROB_THRESHOLD.toFixed(2)}}</b> &nbsp; Shown: <b>${{shown}}</b>`;
      if (idx < targets.length) {{
        requestAnimationFrame(step);
      }}
    }}
    requestAnimationFrame(step);
  }}

  const redraw = debounce(drawInView, 80);
  map.on('moveend', redraw);
  map.on('zoomend', redraw);
  map.whenReady(() => {{ drawInView(); }});
{{% endmacro %}}
""")

class ViewportHexRenderer(MacroElement):
    def __init__(self):
        super().__init__()
        self._template = tmpl

m.get_root().add_child(ViewportHexRenderer())

# -------- Save --------
m.save(OUTPUT_HTML)
print(f"Saved {OUTPUT_HTML}")
