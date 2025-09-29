import numpy as np
import pandas as pd
from pandas import DataFrame


def post_process_outputs(
    scores, W_user_reserve, W_reserve_user, processed_nodes_impacts
):
    processed_nodes_impacts["grad"] = 0
    for i, row in processed_nodes_impacts.iterrows():
        if row.source_type == 0:
            processed_nodes_impacts.loc[i, "grad"] = W_reserve_user[
                row["source_id"], row["target_id"]
            ]
        else:
            processed_nodes_impacts.loc[i, "grad"] = W_user_reserve[
                row["source_id"], row["target_id"]
            ]

    processed_nodes_impacts["user"] = np.where(
        processed_nodes_impacts.source_type == 0,
        processed_nodes_impacts.target,
        processed_nodes_impacts.source,
    )

    scores_output = DataFrame(
        {
            "iteration": [k for k in range(1, len(scores) + 1)],
            "score": scores,
        }
    )

    return scores_output, processed_nodes_impacts


# def post_process_bunch_outputs(
#     scores, W_user_reserve_list, W_reserve_user_list, processed_nodes_impacts
# ):
#     n_users = W_user_reserve_list[0].shape[0]
#     n_reserves = W_user_reserve_list[0].shape[1]
#     n_simulations = len(W_user_reserve_list)

#     W_user_reserve_tensor = np.zeros((n_simulations, n_users, n_reserves))
#     W_reserve_user_tensor = np.zeros((n_simulations, n_reserves, n_users))
#     for simulation_id in range(n_simulations):
#         W_user_reserve_tensor[simulation_id, :, :] = W_user_reserve_list[simulation_id]
#         W_reserve_user_tensor[simulation_id, :, :] = W_reserve_user_list[simulation_id]

#     processed_nodes_impacts["grad"] = 0
#     for i, row in processed_nodes_impacts.iterrows():
#         if row.source_type == 0:
#             processed_nodes_impacts.loc[i, "grad"] = str(
#                 W_reserve_user_tensor[:, row["source_id"], row["target_id"]].tolist()
#             )
#         else:
#             processed_nodes_impacts.loc[i, "grad"] = str(
#                 W_user_reserve_tensor[:, row["source_id"], row["target_id"]].tolist()
#             )

#     processed_nodes_impacts["user"] = np.where(
#         processed_nodes_impacts.source_type == 0,
#         processed_nodes_impacts.target,
#         processed_nodes_impacts.source,
#     )

#     scores_output = DataFrame(
#         {
#             "iteration": [k for k in range(1, len(scores) + 1)],
#             "score": [str(score) for score in scores],
#         }
#     )

#     return scores_output, processed_nodes_impacts


def post_process_bunch_outputs(
    scores, W_user_reserve_list, W_reserve_user_list, processed_nodes_impacts
):
    n_users = W_user_reserve_list[0].shape[0]
    n_reserves = W_user_reserve_list[0].shape[1]
    n_simulations = len(W_user_reserve_list)

    W_user_reserve_tensor = np.zeros((n_simulations, n_users, n_reserves))
    W_reserve_user_tensor = np.zeros((n_simulations, n_reserves, n_users))
    for simulation_id in range(n_simulations):
        W_user_reserve_tensor[simulation_id, :, :] = W_user_reserve_list[simulation_id]
        W_reserve_user_tensor[simulation_id, :, :] = W_reserve_user_list[simulation_id]

    # processed_nodes_impacts["grad"] = 0
    all_users_output = DataFrame()
    for i, row in processed_nodes_impacts.iterrows():
        if row.source_type == 0:
            grad_column = str(
                W_reserve_user_tensor[:, row["source_id"], row["target_id"]].tolist()
            )
        else:
            grad_column = str(
                W_user_reserve_tensor[:, row["source_id"], row["target_id"]].tolist()
            )

        user_output = DataFrame({"grad": grad_column})
        # user_output["connection_id"] = i
        user_output["simulation_id"] = np.arange(len(user_output))
        # user_output["source"] = row["source"]
        # user_output["target"] = row["target"]
        # user_output["weight"] = row["weight"]
        # user_output["source_type"] = row["source_type"]
        # user_output["target_type"] = row["target_type"]
        # user_output["source_id"] = row["source_id"]
        # user_output["target_id"] = row["target_id"]
        user_output["user"] = row["target"] if row["source_type"] == 0 else row["source"]

        all_users_output = pd.concat((all_users_output, user_output))

    # colnames = [
    #     "connection_id",
    #     "source",
    #     "target",
    #     "weight",
    #     "source_type",
    #     "target_type",
    #     "source_id",
    #     "target_id",
    #     "grad",
    # ]
    # all_users_output = all_users_output[colnames]
    all_users_output = all_users_output.groupby(by=["user", "simulation_id"]).agg({"grad": "sum"})

    all_users_output = (
        all_users_output.groupby(by=["user", "grad"], as_index=False)
        .simulation_id.agg("count")
        .to_frame("count")
        .reset_index()
    )

    scores_output = DataFrame(
        {
            "iteration": [k for k in range(1, len(scores) + 1)],
            "score": [str(score) for score in scores],
        }
    )

    return scores_output, all_users_output
