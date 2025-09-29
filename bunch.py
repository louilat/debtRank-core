import boto3
from datetime import datetime, timedelta
import io
import warnings
import numpy as np
import os

warnings.filterwarnings("ignore")

from src.inputs.extract_data import extract_data
from src.inputs.parse_inputs import (
    preprocess_debtrank_inputs,
    create_debtrank_inputs,
    create_all_initial_shocks,
    check_simulation_stability,
)
from src.core.debt_rank_bunch import DebtRankBunch
from src.outputs.process_outputs import post_process_bunch_outputs

access_key_id = os.environ["ACCESS_KEY_ID"]
secret_access_key = os.environ["SECRET_ACCESS_KEY"]

client_s3 = boto3.client(
    "s3",
    region_name="eu-north-1",
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
)

alpha = 2
n_iter = 20

start = datetime(2024, 7, 1)
stop = datetime(2024, 7, 1)
day = start

while day <= stop:
    print("Starting DebtRank run for day", day)

    print("STEP 1 - Extracting input data...")

    nodes_info, nodes_impacts, prices_shocks = extract_data(
        client_s3=client_s3, day=day
    )

    print("STEP 2 - Processing intputs...")

    processed_nodes_info, processed_impact_weights = preprocess_debtrank_inputs(
        nodes_info=nodes_info, impact_weights=nodes_impacts
    )

    W_reserve_user, W_user_reserve, nodes_weights = create_debtrank_inputs(
        processed_nodes_info=processed_nodes_info,
        processed_impact_weights=processed_impact_weights,
    )

    check_simulation_stability(
        alpha=alpha, W_reserve_user=W_reserve_user, W_user_reserve=W_user_reserve
    )

    all_shocks = create_all_initial_shocks(
        assets_names=prices_shocks.columns.tolist(),
        prices_shocks=prices_shocks.values,
        processed_nodes_info=processed_nodes_info,
    )

    # print(initial_h_reserve)

    print("STEP 3 - Running debtRank...")

    dr = DebtRankBunch(alpha=alpha, shocks=all_shocks)

    dr.run_bunch_simulations(
        W_reserve_user=W_reserve_user,
        W_user_reserve=W_user_reserve,
        reserve_values=nodes_weights.reshape(len(nodes_weights), 1),
        n_iter=n_iter,
    )

    print("STEP 4 - Processing outputs...")

    scores_output, grad_output = post_process_bunch_outputs(
        scores=dr.all_scores,
        W_user_reserve_list=dr.grad_W_user_reserve_list,
        W_reserve_user_list=dr.grad_W_reserve_user_list,
        processed_nodes_impacts=processed_impact_weights,
    )

    print("STEP 5 - Generating and saving outputs...")

    day_str = day.strftime("%Y-%m-%d")

    # buffer = io.StringIO()
    # scores_output.to_csv(buffer, index=False)
    # client_s3.put_object(
    #     Bucket="projet-datalab-group-jprat",
    #     Key=f"debtrank/debtrank-outputs-dev/debtrank_outputs_snapshot_date={day_str}/scores.csv",
    #     Body=buffer.getvalue(),
    # )

    # buffer = io.StringIO()
    # grad_output.to_csv(buffer, index=False)
    # client_s3.put_object(
    #     Bucket="projet-datalab-group-jprat",
    #     Key=f"debtrank/debtrank-outputs-dev/debtrank_outputs_snapshot_date={day_str}/connections_gradients.csv",
    #     Body=buffer.getvalue(),
    # )
    scores_output.to_csv("data/scores.csv", index=False)
    grad_output.to_csv("data/gradients.csv", index=False)


    print("Done!")
    day += timedelta(days=1)
