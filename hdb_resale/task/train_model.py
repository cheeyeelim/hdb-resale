"""DAG task to train HDB resale price prediction model."""
import logging
import os

import mlflow
from airflow_submodule.hdb_resale import data, model, sql
from dotenv import load_dotenv
from mlflow.models import infer_signature

# Setup logger
logger = logging.getLogger(__name__)

# Retrieve environment variables
load_dotenv()

POSTGRESQL_DASH_USER = os.environ.get("POSTGRESQL_DASH_USER")
POSTGRESQL_DASH_PASSWORD = os.environ.get("POSTGRESQL_DASH_PASSWORD")
POSTGRESQL_DASH_DATABASE = os.environ.get("POSTGRESQL_DASH_DATABASE")
POSTGRESQL_HOST = os.environ.get("POSTGRESQL_HOST")
POSTGRESQL_PORT = os.environ.get("POSTGRESQL_PORT")


def run(cfg):
    """Main DAG task to train HDB resale price prediction model.

    Stores all models locally as pickled joblib files.

    TODO - store model not locally or inside docker, but in the cloud (e.g. DO Spaces).

    Parameters
    ----------
    cfg : DictConfig
        Hydra configs in OmegaConf format.

    Returns
    -------
    None
    """
    engine, metadata = sql.setup_database(
        postgresql_dash_user=POSTGRESQL_DASH_USER,
        postgresql_dash_password=POSTGRESQL_DASH_PASSWORD,
        postgresql_dash_database=POSTGRESQL_DASH_DATABASE,
        postgresql_host=POSTGRESQL_HOST,
        postgresql_port=POSTGRESQL_PORT,
    )

    # Get training data
    X, y = data.get_training_data(cfg=cfg, engine=engine, metadata=metadata)

    # Define model hyperparameters
    params = {
        "random_state": cfg.model.random_state,
    }

    # Setup ML model
    # Define ensemble model
    resale_model = model.MultiTreeEnsembleRegressor(**params)

    # Run model training
    logger.info("Training model......")
    resale_model = model.train_model(X=X, y=y, model=resale_model)

    # Run model diagnosis
    logger.info("Diagnosing model......")
    cv_res, size, diag_fig = model.diagnose_model(
        X=X,
        y=y,
        model=resale_model,
        random_state=cfg.model.random_state,
    )

    # Setup the MLflow Experiment
    # It will create a new experiment if not exist
    tags = {
        "description": "Model that predicts the price of HDB resale flats",
        "data_source": engine.url.render_as_string(hide_password=True),
    }
    mlflow.set_experiment("HDB Resale Price")
    mlflow.set_experiment_tags(tags)

    model_name = "hdb-resale-price"

    # Start an MLflow run
    with mlflow.start_run():
        # Infer the model signature
        signature = infer_signature(X, resale_model.predict(X))

        # Log the model
        # Do not register model here, register later for more flexibility
        _ = mlflow.sklearn.log_model(
            sk_model=resale_model,
            artifact_path="hdb_resale",
            signature=signature,
            input_example=X.head(20),
            registered_model_name=model_name,
        )

        # Log the hyperparameters
        mlflow.log_params(params)

        # Prepare metrics, in dictionary format
        loss_metrics = cv_res.mean().to_dict()
        size["name"] = size["name"] + "_size"
        size_metrics = size.set_index("name")["act_size"].to_dict()

        # Log metrics
        mlflow.log_metrics(loss_metrics)
        mlflow.log_metrics(size_metrics)

        # Log the diagnostic plots
        mlflow.log_figure(diag_fig, "model_diagnostics.png")
