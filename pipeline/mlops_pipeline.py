import kfp
import kfp.dsl as dsl
from kfp.components import create_component_from_func
from data_preprocessing import data_preprocessing
from train_model import train_model
from evaluate_model import evaluate_model
from deploy_model import deploy_model

# Create pipeline components
data_preprocessing_op = create_component_from_func(data_preprocessing)
train_model_op = create_component_from_func(train_model)
evaluate_model_op = create_component_from_func(evaluate_model)
deploy_model_op = create_component_from_func(deploy_model)

@dsl.pipeline(
    name="MLOps Pipeline",
    description="An end-to-end MLOps pipeline with Kubeflow"
)
def mlops_pipeline():
    preprocess_task = data_preprocessing_op()
    train_task = train_model_op(preprocess_task.outputs["train_data"], preprocess_task.outputs["train_labels"])
    evaluate_task = evaluate_model_op(train_task.output, preprocess_task.outputs["test_data"], preprocess_task.outputs["test_labels"])
    deploy_task = deploy_model_op(train_task.output)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(mlops_pipeline, "pipeline/mlops_pipeline.yaml")
