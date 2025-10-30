# Safe Roads London: Collision Risk Prediction

Predict the probability of a road collision at any location and time in Greater London by combining ten years of Transport for London collision records with OpenStreetMap history and historical weather. This repository covers data collection, H3 based feature engineering, model training with XGBoost, experiment tracking, and end to end deployment.

- Live demo: https://safe-roads-london.onrender.com/
- Model repo: https://huggingface.co/AmanKhokhar/safe-roads/tree/main
- API on Hugging Face Spaces: https://huggingface.co/spaces/AmanKhokhar/safe-roads-catboost  (the name still says catboost from early experiments; the final model uses XGBoost)
- Web app repo: https://github.com/Aman-Khokhar18/safe-roads-london
- Main repo: https://github.com/Aman-Khokhar18/safe-roads

---

## Table of contents

- Scope
- Data sources and storage
- Why H3 and how it is used
- OSM history with ohsome
- Weather with Meteostat
- Data modeling
- Feature engineering
- Modeling
- Experiment tracking and optimization
- Results
- ETL and orchestration
- Training
- Deployment
- Reports and figures
- Cost choices
- Limitations
- Future work and use cases
- Getting started
- Architecture
- Acknowledgements

---

## Scope

- Build a spatiotemporal risk model that outputs the probability of a collision for a given hex cell and timestamp.
- Cover 2015 to 2024 using TfL road safety data across Greater London.
- Enrich collision points with street network context from historical OSM, weather from Meteostat, and temporal signals.
- Aggregate everything on H3 hexagons so the model learns neighborhood level conditions rather than only point coordinates.
- Serve predictions with a FastAPI service on Hugging Face Spaces, store results in PostgreSQL, and visualize on the web with Leaflet.

---

## Data sources and storage

### Sources

1) Collisions  
   Transport for London Road Safety Data  
   https://tfl.gov.uk/corporate/publications-and-reports/road-safety  
   Years: 2015 to 2024.

2) Street and road context  
   OpenStreetMap via the ohsome API to fetch historical snapshots that match each collision timestamp.

3) Weather  
   Meteostat for historical hourly and daily weather aligned to each record.

### Storage

- All raw and processed tables live in PostgreSQL on AWS RDS.
- The ETL normalizes schema, converts coordinates, assigns H3, and records provenance.

---

## Why H3 and how it is used

The TfL dataset includes precise British National Grid easting and northing per collision.

Steps:
1) Convert easting and northing to latitude and longitude.  
2) Map lat and lon to H3 cells at a chosen resolution.  
3) Use hex IDs to join collisions with OSM features and weather in a consistent spatial frame.

Why H3 helps:
- Local context  
  A single point rarely explains a collision. H3 lifts a point into its surrounding area, which captures street layout, controls, speed limits, and pedestrian or cycling infrastructure. This provides richer environmental context.
- Uniform tiling and hierarchy  
  Hex cells form a global grid with consistent neighborhood relationships. You can roll up or drill down by resolution, build k rings, and compute neighbor statistics.
- Multi scale analysis  
  Resolution can be tuned to the density of London roads. This pipeline also computes neighbor summaries at resolution 11 for stability and smoother signals.
- Time alignment  
  Each record is hex plus timestamp, which makes it straightforward to join the correct historical OSM snapshot and weather observation.
- Aggregation friendly  
  Hex bins support counts, shares, means, and graph summaries. This reduces noise and lets the model learn from area level patterns instead of only raw points.

![H3 hex overlay with sample collisions](https://i.ibb.co/VpJyQDV7/h3.png)

---

## OSM history with ohsome

- Fetch historical OSM features that exist at or before each collision time, not only the latest map.
- Extract tags for ways and nodes that matter for safety:
  - highway class, maxspeed, lanes, width, surface, smoothness
  - one way flags, sidewalks, cycle lanes or tracks
  - bridges, tunnels, barriers, amenities
  - crossings, traffic signals, bus stops, speed cameras, mini roundabouts
- Aggregate per H3 cell and align to the timestamp.

---

## Weather with Meteostat

- Pull weather observations aligned to each collision time and also for negative samples.
- Features include temperature, dew point, relative humidity, precipitation, snowfall, wind direction and speed, gusts, pressure, sunshine duration, and weather code.

---

## Data modeling

- Target  
  Binary label indicating whether a collision occurred for a hex cell and timestamp pair.

- Negative sampling  
  Collisions are rare. src/safe_roads/flows/ingest_negativedata.py creates realistic negatives by sampling times and hex cells without recorded collisions. The sampling preserves temporal structure to avoid easy negatives that inflate scores.

- Spatiotemporal joins  
  Each hex time record joins to OSM features from the correct historical snapshot and the matching weather observation. This avoids look ahead bias.

- Neighborhood smoothing  
  Additional stability comes from neighbor aggregates at H3 resolution 11, which reduce noise in sparse areas.

- Schema overview  
  Suggested tables and key columns:
  - tfl_collisions: collision_id, occurred_at, easting, northing, lat, lon, h3
  - osm_features_h3: h3, valid_at, counts, shares, means
  - weather_observations: ts, station fields, and per hour metrics
  - ml_dataset: h3, ts, engineered features, label
  - predictions: h3, ts, score, model_version

- Validation  
  Time based splits are used for evaluation. Spatial cross validation by borough or by H3 partitions is recommended when changing resolutions or adding new coverage.

---

## Feature engineering

Time and calendar
- dt_year, dt_month, dt_day, dt_hour
- dt_is_weekend, dt_is_weekday
- cyclic encodings: hour_sin, hour_cos, dow_sin, dow_cos, dom_sin, dom_cos, month_sin, month_cos

Weather
- temp, dwpt, rhum, prcp, snow, wdir, wspd, wpgt, pres, tsun, coco

OSM derived numeric aggregates
- Capacity and speed context  
  lanes_num_avg, lanes_num_max, width_m_avg, width_m_max, smoothness_score_avg, maxspeed_mph_avg, maxspeed_mph_max
- Counts of key attributes in the hex  
  cnt_is_primary, cnt_is_secondary, cnt_is_tertiary, cnt_is_residential, cnt_is_service, cnt_is_track_or_path
- One way and sidewalks  
  cnt_oneway_forward, cnt_oneway_bidirectional, cnt_oneway_reverse, cnt_sidewalk_both, cnt_sidewalk_left, cnt_sidewalk_right, cnt_sidewalk_none
- Cycling and access  
  cnt_bicycle_yes, cnt_bicycle_designated, cnt_bicycle_permissive, cnt_bicycle_no, cnt_access_permissive, cnt_access_destination, cnt_access_private, cnt_access_no
- Safety related infrastructure  
  cnt_is_bridge, cnt_is_tunnel, cnt_has_barrier, cnt_has_amenity, cnt_has_bus_stop, cnt_has_mini_roundabout, cnt_has_speed_camera

OSM shares by class
- share_is_motorway, share_is_trunk, share_is_primary, share_is_secondary, share_is_tertiary, share_is_residential, share_is_service, share_is_track_or_path, share_is_foot_or_ped

r11 aggregates
- r11_mean_maxspeed_mph, r11_mean_lanes, r11_mean_width_m, r11_mean_smoothness
- r11_mean_cnt_has_signals, r11_mean_cnt_has_crossing, r11_mean_cnt_speed_camera, r11_mean_cnt_bus_stop, r11_mean_cnt_amenity
- r11_mean_cnt_is_primary, r11_mean_cnt_is_secondary, r11_mean_cnt_is_tertiary, r11_mean_cnt_is_residential, r11_mean_cnt_is_service, r11_mean_cnt_is_track_or_path
- r11_share_oneway_forward, r11_share_oneway_bidirectional, r11_share_oneway_reverse
- r11_mean_cnt_sidewalk_none, r11_mean_cnt_sidewalk_both, r11_mean_cnt_cycle_infra, r11_mean_cnt_cycle_lane, r11_mean_cnt_cycle_track

Graph features
- junction_degree approximates how connected a hex is in the road graph.

Boolean flags
- is_junction, is_turn  
  Kept as booleans in the schema and can be cast to integers in the pipeline.

---

## Modeling

Algorithm: XGBoost  
Objective: binary:logistic  
Eval metric: logloss  
Hardware: device="cuda" when available

Key parameters and intent:
- learning_rate=0.02 with n_estimators=15000 and early_stopping_rounds=20  
  Many small steps with early stopping to prevent overfitting.
- max_depth=6 and min_child_weight=2  
  Control tree complexity and reduce variance.
- subsample=0.8 and colsample_bytree=0.8  
  Stochasticity to improve generalization.
- tree_method="hist"  
  Fast histogram based training.
- reg_lambda=1.0  
  L2 regularization for stability.

Training script path:
- src/saferoads/model/train_xgboost

---

## Experiment tracking and optimization

- MLflow  
  Logs parameters, metrics, models, and artifacts. Each run records dataset slice, H3 resolution, feature set, and training metadata.

- Hyperopt  
  Bayesian hyperparameter search over learning rate, depth, min child weight, and subsampling. Hyperopt runs are tracked in MLflow.

---

## Results

| Metric | Value |
|-------|------:|
| AUC | 0.9896938971323956 |
| PRAUC | 0.9221513692370529 |
| LogLoss | 0.03291765577176357 |
| Brier | 0.007874666076258969 |
| Brier Skill Score | 0.7920958585742299 |

![Confusion Matrix](https://i.ibb.co/B5p2tNkG/confusion-matrix.png)


Interpretation:
- AUC near 1 indicates strong rank ordering of risky versus non risky cases.
- PRAUC is high, which helps when positives are rare.
- Low LogLoss and Brier indicate well calibrated probabilities.
- Brier Skill Score around 0.79 shows a large improvement over a climatology baseline.

![ROC Curve](https://ibb.co/LzBFMP4R)



---

## ETL and orchestration

A full ETL pipeline orchestrated with Prefect. I did not use Prefect scheduled deployments because cron lives in GitHub Actions for transparency.

Flows
- src/safe_roads/flows/ingest_collisiondata.py  
  Ingest TfL collisions, convert coordinates, assign H3, write to Postgres.
- src/safe_roads/flows/ingest_negativedata.py  
  Create negative samples for training by sampling times and hex cells without recorded collisions.

Run flows locally
```bash
- collision data  
    python src/safe_roads/flows/ingest_collisiondata.py --years 2015 2024 --h3-res 11
- negative data  
    python src/safe_roads/flows/ingest_negativedata.py --years 2015 2024 --h3-res 11
```

GitHub Actions schedule example

```yml
- refresh weather and ETL maintenance  
    name: refresh-weather-and-etl  
    on:  
      schedule:  
        - cron: "0 */6 * * *"  
      workflow_dispatch:  
    jobs:  
      refresh:  
        runs-on: ubuntu-latest  
        steps:  
          - uses: actions/checkout@v4  
          - uses: actions/setup-python@v5  
            with:  
              python-version: "3.11"  
          - run: pip install -r requirements.txt  
          - name: Run weather update  
            run: python etl/weather_features.py  
          - name: Rebuild training dataset if needed  
            run: python etl/build_dataset.py --incremental
```
---

## Training

Run training
```bash
- python src/saferoads/model/train_xgboost --config configs/xgb.yaml --mlflow-uri sqlite:///mlruns.db
```
---

## Deployment

Serving
- Model artifacts are versioned on Hugging Face  
  https://huggingface.co/AmanKhokhar/safe-roads/tree/main
- FastAPI application exposes prediction endpoints on Hugging Face Spaces  
  https://huggingface.co/spaces/AmanKhokhar/safe-roads-catboost  (name retained from early experiments; model is XGBoost)

Infra and packaging
- Compute: AWS EC2 free tier for batch and maintenance jobs.
- Database: AWS RDS PostgreSQL stores raw data, features, and predictions.
- Containerization: Docker images for the API and jobs. Images are pushed to GitHub Container Registry.
- CI: GitHub Actions run scheduled workflows to refresh weather so predictions stay current.
- Deployment folder: Dockerfiles and scripts live under deployment/.

![Infrastructure](https://i.ibb.co/qMXN923M/Intrastructure.png)

Online prediction loop
1) Load live inputs from PostgreSQL.  
2) Send features to the HF Spaces FastAPI endpoint for inference.  
3) Write predictions back to PostgreSQL.  
4) The web app reads predictions and renders them with Leaflet and JavaScript. The site is deployed on Render.com as a static app.

---

## Reports and figures

Place figures in docs/ and reference them here. MLflow also logs these as artifacts during training.
- ROC curve: docs/roc_curve.png  
- Confusion matrix: docs/confusion_matrix.png  
- Infrastructure overview: docs/infra_overview.png

---

## Cost choices

Goal is to keep hosting cost minimal.
- EC2 and RDS on free tier where possible
- Hugging Face Spaces free tier for the API app
- Render.com free tier for static web hosting
- GitHub Actions for scheduled data refresh and GHCR for image storage

---

## Limitations

- OSM data completeness  
  OpenStreetMap can be incomplete or inconsistent in some areas and times. Missing tags, sparse mapping activity, and delayed edits can reduce feature quality. The pipeline uses historical snapshots to stay time correct, but gaps in OSM coverage remain a source of noise.
- Reporting effects  
  Collision data depends on reporting and may miss minor incidents.
- Exposure not explicit  
  Traffic volumes and pedestrian or cyclist flows are not modeled directly. This can bias risk in locations with high exposure.
- Temporal drift  
  Network changes and policy interventions can shift risk over time. Regular retraining and drift monitoring are recommended.

---

## Future work and use cases

- Routing for safety  
  Build a routing app that selects the safest path given origin, destination, and time.
- Infrastructure planning  
  Support councils and planners with risk maps for targeted interventions such as crossings, speed cameras, and traffic calming.
- Traffic monitoring  
  Combine live feeds and streaming weather to produce updated risk scores for operational dashboards.
- Better exposure modeling  
  Integrate traffic counts, mobile location data, and pedestrian or cycling volumes.
- Calibration at scale  
  Apply isotonic or Platt scaling per borough or road class and monitor drift.
- Uncertainty estimates  
  Add conformal prediction intervals to report risk bounds.
- Spatial validation  
  Formalize spatial cross validation folds by borough or by H3 partitions.
- Interpretability  
  Provide SHAP summaries and partial dependence plots per borough and road class.

---

## Getting started

Prerequisites
- Python 3.10 or newer
- PostgreSQL 14 or newer
- Docker for building images if you want to run the API locally
- An AWS account if you plan to mirror the infra
- Optional CUDA for GPU training

Environment variables
```yml  
Create a .env file with at least:
- POSTGRES_HOST
- POSTGRES_PORT
- POSTGRES_DB
- POSTGRES_USER
- POSTGRES_PASSWORD
- HF_TOKEN (optional if any private repos or endpoints are used)
```

Install
```bash
- git clone https://github.com/Aman-Khokhar18/safe-roads.git  
- cd safe-roads  
- python -m venv .venv && source .venv/bin/activate  
- pip install -r requirements.txt
```

Data load outline
```bash
- Ingest TfL CSVs to Postgres via Prefect flows  
  python src/safe_roads/flows/ingest_collisiondata.py --years 2015 2024 --h3-res 11  
  python src/safe_roads/flows/ingest_negativedata.py --years 2015 2024 --h3-res 11
- Pull OSM history via ohsome and aggregate per H3  
  python etl/ohsome_features.py --resolution 11
- Join Meteostat weather  
  python etl/weather_features.py
- Materialize the training dataset  
  python etl/build_dataset.py
```

Serve locally with Docker
```bash
- Build  
  docker build -f deployment/Dockerfile.api -t saferoads-api .
- Run  
  docker run -p 7860:7860 --env-file .env saferoads-api
- Example request  
  curl -X POST http://localhost:7860/predict -H "Content-Type: application/json" -d '{"h3":"8928308280fffff","timestamp":"2024-06-01T08:00:00Z"}'  
  The exact payload shape depends on the API app. See the FastAPI code for the latest schema.
```
---

## Architecture

High level flow
- TfL collisions 2015 to 2024
  - convert easting and northing to lat and lon
  - index to H3 cells
- ohsome OSM history
  - aggregate features per H3 and per time
- Meteostat weather
  - join hourly conditions
- Feature join on H3 plus time
- Train XGBoost
- Push model artifacts to Hugging Face
- Serve with FastAPI on Hugging Face Spaces
- Batch jobs on EC2 build live features and call the API
- Predictions stored in RDS PostgreSQL
- Leaflet map on Render.com reads predictions for the public site

---

## Acknowledgements

- Transport for London for collision data
- OpenStreetMap contributors and the ohsome API
- Meteostat for historical weather
- H3 by Uber Engineering
- XGBoost, MLflow, Hyperopt, Prefect, Leaflet, FastAPI, Docker, and the broader open source community
- kim <3
