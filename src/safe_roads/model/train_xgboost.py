import pandas as pd
import json
from tqdm.auto import tqdm
import xgboost as xgb
from xgboost import callback

import os
os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"
os.environ["PREFECT_API_ENABLE_EPHEMERAL_SERVER"] = "false"
from prefect import task

import mlflow
import mlflow.xgboost
mlflow.set_tracking_uri("http://localhost:5000")

from safe_roads.utils.mlutil import data_loader,  prepare_data
from safe_roads.utils.config import load_config
    

@task(name="Train XGBOOST model")
def train():

    config = load_config("configs/train.yml")
    EXPERIMENT_NAME = config['EXPERIMENT_NAME']
    RANDOM_STATE = config['RANDOM_STATE']
    TARGET = config["TARGET"]
    TEST_SIZE = config["TEST_SIZE"]
    VAL_SIZE = config["VAL_SIZE"]

    with tqdm(total=2, desc="Loading data") as p:
        pos = data_loader("roads_features_collision")
        p.update(1)
        neg = data_loader("roads_features_negatives")
        p.update(1)



    X_train, y_train, X_val, y_val, _, _  = prepare_data(config, pos, neg)

    pos_label = config["POS_LABEL"]
    pos_cnt = (y_train == pos_label).sum()
    neg_cnt = (y_train != pos_label).sum()
    scale_pos_weight = (neg_cnt / max(1, pos_cnt))

    early_stop = xgb.callback.EarlyStopping(
    rounds=5, metric_name='aucpr', maximize=True, save_best=True,  min_delta=1e-4)

    lr_sched = xgb.callback.LearningRateScheduler(
    lambda t: 0.05 if t < 50 else (0.1 if t < 200 else (0.05 if t < 1000 else 0.01))
)                    
    model = xgb.XGBClassifier( 
        objective="binary:logistic",  
        tree_method="hist", 
        enable_categorical = True,
        n_estimators=3000,
        learning_rate=0.1,
        min_child_weight=5,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda =1.0,
        reg_alpha=0.5,
        eval_metric="aucpr",
        scale_pos_weight= float(scale_pos_weight),
        random_state=RANDOM_STATE,
        callbacks = [early_stop, lr_sched]
    )


    # ---- MLflow ----
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.xgboost.autolog(log_input_examples=False, log_model_signatures=False, log_models=True)

    with mlflow.start_run(run_name="xgb_categorical_hist"):
        # Always log features used
        mlflow.log_text(json.dumps(list(X_train.columns), indent=2), "features.json")
        mlflow.log_params({
            "n_rows_train": X_train.shape[0],
            "n_features": X_train.shape[1],
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "early_stopping": True,
            "learning_rate": True, 
            "scale_pos_weight": float(scale_pos_weight),
        })

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        mlflow.log_metrics({
        "best_iteration": int(model.best_iteration),
        "best_score": float(model.best_score),
        "n_estimators_effective": int(model.best_iteration + 1),
        })


    print(model.best_iteration, model.best_score)
    print(model.evals_result())
    print("Training complete.")



if __name__ == "__main__":
    train()