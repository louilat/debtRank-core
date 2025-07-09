import boto3
from datetime import datetime
import io
import warnings
import numpy as np
import os

warnings.filterwarnings("ignore")

from src.inputs.extract_data import extract_data
from src.inputs.parse_inputs import (
    preprocess_debtrank_inputs,
    create_debtrank_inputs,
    create_initial_shock,
)
from src.core.debt_rank import DebtRank
from src.outputs.process_outputs import post_process_outputs

access_key_id = os.environ["ACCESS_KEY_ID"]
secret_access_key = os.environ["SECRET_ACCESS_KEY"]

client_s3 = boto3.client(
    "s3",
    endpoint_url="https://" + "minio-simple.lab.groupe-genes.fr",
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    verify=False,
)

alpha = 2
n_iter = 20
day = datetime(2024, 1, 14)

print("STEP 1 - Extracting input data...")

nodes_info, nodes_impacts, prices_shocks = extract_data(client_s3=client_s3, day=day)

print("STEP 2 - Processing intputs...")

processed_nodes_info, processed_impact_weights = preprocess_debtrank_inputs(
    nodes_info=nodes_info, impact_weights=nodes_impacts
)

W_reserve_user, W_user_reserve, nodes_weights = create_debtrank_inputs(
    processed_nodes_info=processed_nodes_info,
    processed_impact_weights=processed_impact_weights,
)

# print(processed_nodes_info.head(5))
# print(W_reserve_user.shape)
# print(W_user_reserve.shape)
# print(nodes_weights.shape)
# print(nodes_weights.sum())

initial_h_reserve = create_initial_shock(
    assets_names=prices_shocks.columns.tolist(),
    prices_shocks=prices_shocks.loc[47].values.tolist(),
    processed_nodes_info=processed_nodes_info,
)

print(initial_h_reserve)

print("STEP 3 - Running debtRank...")

dr = DebtRank(alpha=alpha)

score, grad_ru, grad_ur = dr.run_simulation(
    W_reserve_user=W_reserve_user,
    W_user_reserve=W_user_reserve,
    reserve_values=nodes_weights.reshape(len(nodes_weights), 1),
    initial_h_reserve=initial_h_reserve,
    n_iter=n_iter,
)

print(score)

print("STEP 4 - Processing outputs...")

scores_output, grad_output = post_process_outputs(
    scores=score,
    W_user_reserve=grad_ur,
    W_reserve_user=grad_ru,
    processed_nodes_impacts=processed_impact_weights,
)

print("STEP 5 - Generating and saving outputs...")

day_str = day.strftime("%Y-%m-%d")

buffer = io.StringIO()
scores_output.to_csv(buffer, index=False)
client_s3.put_object(
    Bucket="projet-datalab-group-jprat",
    Key=f"debtrank/debtrank-outputs/debtrank_outputs_snapshot_date={day_str}/scores.csv",
    Body=buffer.getvalue(),
)

buffer = io.StringIO()
grad_output.to_csv(buffer, index=False)
client_s3.put_object(
    Bucket="projet-datalab-group-jprat",
    Key=f"debtrank/debtrank-outputs/debtrank_outputs_snapshot_date={day_str}/connections_gradients.csv",
    Body=buffer.getvalue(),
)

print("Done!")
