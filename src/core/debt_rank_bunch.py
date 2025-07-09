from src.core.debt_rank import DebtRank


class DebtRankBunch:
    def __init__(self, alpha, shocks):
        self.alpha = alpha
        self.shocks = shocks
        self.grad_W_user_reserve_list = list()
        self.grad_W_reserve_user_list = list()
        self.all_scores = list()

    def run_bunch_simulations(
        self, W_reserve_user, W_user_reserve, reserve_values, n_iter
    ):
        n = len(self.shocks)
        for idx, initial_h in enumerate(self.shocks):
            print(f"Treating shock {idx} / {n}")
            debtrank = DebtRank(alpha=self.alpha)
            score, grad_ru, grad_ur = debtrank.run_simulation(
                W_reserve_user=W_reserve_user,
                W_user_reserve=W_user_reserve,
                reserve_values=reserve_values,
                initial_h_reserve=initial_h,
                n_iter=n_iter,
                verbose=False,
            )
            self.all_scores.append(score)
            self.grad_W_user_reserve_list.append(grad_ur)
            self.grad_W_reserve_user_list.append(grad_ru)
