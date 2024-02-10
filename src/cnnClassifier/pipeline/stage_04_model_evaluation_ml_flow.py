import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))  # Adjust the number of '..' as needed
sys.path.append(project_dir)
from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.model_evaluation_ml_flow import Evaluation
from src.cnnClassifier import logger

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/Kamesk/kidney_tumor.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="Kamesk"
os.environ["MLFLOW_TRACKING_PASSWORD"]="8a9f1f9042667d73da737556a992051abe77ba15"

STAGE_NAME = "model evaluation stages"

class EvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        # evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


