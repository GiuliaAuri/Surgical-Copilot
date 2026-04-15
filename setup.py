import os
import wandb
import mlflow
import dagshub
from dotenv import load_dotenv

def initialize_mlops(project_name, run_name, config=None):
   
    load_dotenv() 

    dagshub.init(
        repo_owner='minila03', 
        repo_name='surgical-copilot', 
        mlflow=True
    )
    
    mlflow.set_experiment(project_name)

    #with mlflow.start_run():
    #    mlflow.log_param('parameter name', 'value')
    #    mlflow.log_metric('metric name', 1)

    wandb.init(
        project=project_name,
        name=run_name,
        config=config
    )
    
    print(f"🚀 MLOps Initialized: Tracking on DAGsHub (MLflow) and W&B.")