from pandas import DataFrame
from numpy import ndarray
import tensorflow as tf
from tensorflow import SparseTensor
import numpy as np


def preprocess_debtrank_inputs(nodes_info: DataFrame, impact_weights: DataFrame):
    """
    Encode nodes names to unique id per category.
    """
    assert len(nodes_info) == len(nodes_info.drop_duplicates(subset="name")), (
        "Same name found for different nodes in nodes_info, please check your input"
    )
    assert len(nodes_info.type.unique()) <= 2, (
        "You can provide at most two different types of nodes, please check your input"
    )
    nodes_info_ = nodes_info.copy()
    nodes_info_["id"] = nodes_info_.groupby("type").cumcount()
    nodes_info_["type"] = nodes_info_.groupby("type").ngroup()

    impact_weights_ = (
        impact_weights.merge(
            nodes_info_[["name", "type"]], how="left", left_on="source", right_on="name"
        )
        .rename(columns={"type": "source_type"})
        .drop(columns="name")
    )
    impact_weights_ = (
        impact_weights_.merge(
            nodes_info_[["name", "type"]], how="left", left_on="target", right_on="name"
        )
        .rename(columns={"type": "target_type"})
        .drop(columns="name")
    )

    assert np.sum((impact_weights_.source_type == impact_weights_.target_type)) == 0, (
        "The network topology is not bipartite, please check your input"
    )

    impact_weights_ = (
        impact_weights_.merge(
            nodes_info_[["name", "id"]], how="left", left_on="source", right_on="name"
        )
        .rename(columns={"id": "source_id"})
        .drop(columns="name")
    )
    impact_weights_ = (
        impact_weights_.merge(
            nodes_info_[["name", "id"]], how="left", left_on="target", right_on="name"
        )
        .rename(columns={"id": "target_id"})
        .drop(columns="name")
    )
    return nodes_info_, impact_weights_


def create_debtrank_inputs(
    processed_nodes_info: DataFrame, processed_impact_weights: DataFrame
) -> tuple[SparseTensor, SparseTensor, ndarray]:
    reserve_user_weights = processed_impact_weights[
        processed_impact_weights.source_type == 0
    ]
    user_reserve_weights = processed_impact_weights[
        processed_impact_weights.source_type == 1
    ]

    reserve_user_indices = list(
        zip(reserve_user_weights.source_id, reserve_user_weights.target_id)
    )
    user_reserve_indices = list(
        zip(user_reserve_weights.source_id, user_reserve_weights.target_id)
    )

    r = len(processed_nodes_info[processed_nodes_info.type == 0])
    u = len(processed_nodes_info[processed_nodes_info.type == 1])
    nodes_weights = (
        processed_nodes_info[processed_nodes_info.type == 0]
        .sort_values(by="id")
        .reset_index(drop=True)
        .weight.values
    )

    W_reserve_user = SparseTensor(
        indices=reserve_user_indices,
        values=reserve_user_weights.weight,
        dense_shape=(r, u),
    )
    W_user_reserve = SparseTensor(
        indices=user_reserve_indices,
        values=user_reserve_weights.weight,
        dense_shape=(u, r),
    )

    W_reserve_user = tf.sparse.reorder(W_reserve_user)
    W_reserve_user = tf.sparse.to_dense(W_reserve_user)
    W_reserve_user = tf.cast(W_reserve_user, dtype=np.float32)

    W_user_reserve = tf.sparse.reorder(W_user_reserve)
    W_user_reserve = tf.sparse.to_dense(W_user_reserve)
    W_user_reserve = tf.cast(W_user_reserve, dtype=np.float32)

    return W_reserve_user, W_user_reserve, nodes_weights


def create_initial_shock(
    assets_names: list, prices_shocks: list, processed_nodes_info: DataFrame
):
    shocks_data = DataFrame({"name": assets_names, "shock": -np.array(prices_shocks)})
    reserves_info = processed_nodes_info.loc[
        processed_nodes_info.type == 0, ["name", "id"]
    ].copy()
    reserves_info = reserves_info.merge(shocks_data, how="left", on="name")
    reserves_info.shock = reserves_info.shock.fillna(0)
    reserves_info = reserves_info.sort_values("id").reset_index(drop=True)
    shocks = reserves_info.shock.values.reshape(len(reserves_info), 1)
    # print("Initial shock is:", shocks)
    return np.maximum(np.minimum(shocks, 0.5), -0.5)


def create_all_initial_shocks(
    assets_names: list, prices_shocks: np.ndarray, processed_nodes_info: DataFrame
):
    all_initial_h = []
    for shock_id in range(prices_shocks.shape[0]):
        shock = prices_shocks[shock_id].tolist()
        h = create_initial_shock(
            assets_names=assets_names,
            prices_shocks=shock,
            processed_nodes_info=processed_nodes_info,
        )
        all_initial_h.append(h)
    return all_initial_h


def check_simulation_stability(alpha, W_reserve_user, W_user_reserve):
    n_reserves = W_reserve_user.shape[0]
    n_users = W_reserve_user.shape[1]
    W = np.block(
        [
            [W_reserve_user, np.zeros((n_reserves, n_reserves))],
            [np.zeros((n_users, n_users)), W_user_reserve],
        ]
    )
    eigenvalues, _ = np.linalg.eig(W)
    max_eig = np.max(eigenvalues)
    print("   --> INFO: Max eigen value for W is: ", max_eig)
    # assert np.log(max_eig) < alpha, f"Simulation is not stable!"
