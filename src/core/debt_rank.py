import tensorflow as tf
from tensorflow import Tensor
import numpy as np
from numpy import ndarray


class DebtRank:
    def __init__(self, alpha):
        self.alpha = alpha
        self.iter = 0

    def debtRankLayer(
        self,
        W_reserve_user: Tensor,
        W_user_reserve: Tensor,
        h_reserve: ndarray,
        h_user: ndarray,
        p_reserve: ndarray,
        prev_p_reserve: ndarray,
        p_user: ndarray,
        prev_p_user: ndarray,
    ):
        assert np.max(np.abs(h_reserve)) <= 1, "h_reserve out of [-1, 1]"
        assert np.max(np.abs(h_user)) <= 1, "h_user out of [-1, 1]"
        assert np.min(p_reserve - prev_p_reserve) >= 0, (
            "Unexpected decrease of p_reserve"
        )
        assert np.min(p_user - prev_p_user) >= 0, "Unexpected decrease of p_user"

        new_h_reserve = tf.math.minimum(
            1,
            h_reserve
            + tf.linalg.matmul(W_user_reserve, p_user - prev_p_user, transpose_a=True),
        )
        new_h_user = tf.math.minimum(
            1,
            h_user
            + tf.linalg.matmul(
                W_reserve_user, p_reserve - prev_p_reserve, transpose_a=True
            ),
        )
        new_p_reserve = tf.math.maximum(
            tf.math.multiply(
                new_h_reserve, tf.math.exp(self.alpha * (new_h_reserve - 1))
            ),
            0,
        )
        new_p_user = tf.math.maximum(
            tf.math.multiply(new_h_user, tf.math.exp(self.alpha * (new_h_user - 1))), 0
        )

        assert np.min(new_h_reserve - h_reserve) >= 0, (
            "Unexpected decrease of h_reserve"
        )
        assert np.min(new_h_user - h_user) >= 0, "Unexpected decrease of h_user"

        return new_h_reserve, new_h_user, new_p_reserve, p_reserve, new_p_user, p_user

    def run_simulation(
        self,
        W_reserve_user,
        W_user_reserve,
        reserve_values,
        initial_h_reserve,
        n_iter,
        verbose: bool = True,
    ):
        # W_ru = np.asarray(W_reserve_user, np.float32)
        # W_ru = tf.convert_to_tensor(W_ru, np.float32)
        # W_ru = tf.Variable(W_ru)
        W_ru = tf.Variable(W_reserve_user)

        # W_ur = np.asarray(W_user_reserve, np.float32)
        # W_ur = tf.convert_to_tensor(W_ur, np.float32)
        # W_ur = tf.Variable(W_ur)
        W_ur = tf.Variable(W_user_reserve)

        n_reserves = W_user_reserve.shape[1]
        n_users = W_user_reserve.shape[0]

        h_reserve = initial_h_reserve.astype(np.float32)
        h_user = np.zeros((n_users, 1)).astype(np.float32)

        prev_p_reserve = np.zeros((n_reserves, 1))
        prev_p_user = np.zeros((n_users, 1))
        p_reserve = tf.math.maximum(
            tf.math.multiply(h_reserve, tf.math.exp(self.alpha * (h_reserve - 1))), 0
        )

        p_user = np.zeros((n_users, 1)).astype(np.float32)

        scores_history = []
        with tf.GradientTape() as tape:
            for _ in range(n_iter):
                self.iter += 1
                if verbose:
                    print(f"Iteration {self.iter}")
                h_reserve, h_user, p_reserve, prev_p_reserve, p_user, prev_p_user = (
                    self.debtRankLayer(
                        W_reserve_user=W_ru,
                        W_user_reserve=W_ur,
                        h_reserve=h_reserve,
                        h_user=h_user,
                        p_reserve=p_reserve,
                        prev_p_reserve=prev_p_reserve,
                        p_user=p_user,
                        prev_p_user=prev_p_user,
                    )
                )
                if verbose:
                    print("   --> Mean h_reserve: ", tf.reduce_mean(h_reserve).numpy())
                    print("   --> Mean h_user: ", tf.reduce_mean(h_user).numpy())
                scores_history.append(
                    tf.reduce_sum(
                        tf.math.multiply(h_reserve, reserve_values)
                        - tf.math.multiply(
                            initial_h_reserve.astype(np.float32), reserve_values
                        )
                    ).numpy()
                )
            y = tf.reduce_sum(h_reserve * reserve_values)
            # print(h_reserve)

        weights = {"W_ru": W_ru, "W_ur": W_ur}
        grad = tape.gradient(y, weights)
        grad_reserve_user = grad["W_ru"].numpy()
        grad_user_reserve = grad["W_ur"].numpy()

        assert grad_reserve_user is not None, "No!"
        assert grad_user_reserve is not None, "No!"

        return scores_history, grad_reserve_user, grad_user_reserve
