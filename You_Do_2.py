import numpy as np
from tqdm import trange
import pandas as pd
from contextlib import suppress


class UserBased:
    mu: np.ndarray
    sim: np.ndarray

    def __init__(self, zero_mean: bool = True, beta: int = 1, idf: bool = False, verbosity: int = 0):
        """

        :param zero_mean:
        :param beta: Discounting parameter
        :param idf: Enable inverse document frequency management
        """
        self.zero_mean = zero_mean
        self.beta = beta
        self.idf = idf
        self.verbosity = verbosity

    def fit(self, r: np.ndarray):
        m, n = r.shape
        if self.zero_mean:
            self.mu = np.nanmean(r, axis=1)
        else:
            self.mu = np.zeros(m)

        self.sim = np.zeros((m, m))

        if self.idf:
            idf = np.log(1 + m / (~np.isnan(r)).sum(axis=0))
        else:
            idf = np.ones(n)

        if self.verbosity > 0:
            print(idf)

        for i in trange(m):
            for j in range(m):
                mask = ~np.isnan(r[i, :]) & ~np.isnan(r[j, :])

                si = r[i, mask] - self.mu[i]
                sj = r[j, mask] - self.mu[j]

                self.sim[i][j] = (si * sj * idf[mask]).sum() / (
                        np.sqrt((idf[mask] * (si ** 2)).sum()) * np.sqrt((idf[mask] * (sj ** 2)).sum()))

                total_intersection = mask.sum()

                self.sim[i][j] *= min(total_intersection, self.beta) / self.beta

        return self.sim

    def predict(self, r: np.array, u: int, top_k: int = 3) -> np.ndarray:
        """

        :param r: Rating matrix
        :param u: User u
        :param top_k: Top k neighbourhood
        :return: Calculated Rating of each item
        """

        _, n = r.shape

        score = np.zeros(n)

        for j in trange(n):
            score[j] = self.predict1(r, u, j, top_k)

        return score

    def predict1(self, r: np.array, u: int, j: int, top_k: int = 3) -> float:
        _, n = r.shape

        users_rated_j = np.nonzero(~np.isnan(r[:, j]))[0]

        topk_users = users_rated_j[self.sim[u, users_rated_j].argsort()[::-1][:top_k]]

        mean_centered_topk_user_rate = r[topk_users, j] - self.mu[topk_users]

        w = self.sim[u, topk_users]

        return np.dot(mean_centered_topk_user_rate, w) / np.abs(w).sum() + self.mu[u]

def top_k_sim_user_rate(user_sim_mat : np.array, r: np.array, u: int, j: int, top_k: int = 3) -> float:
    _, n = r.shape
    mu = np.nanmean(r, axis=1)

    users_rated_j = np.nonzero(~np.isnan(r[:, j]))[0]

    topk_users = users_rated_j[user_sim_mat[u, users_rated_j].argsort()[::-1][:top_k]]

    mean_centered_topk_user_rate = r[topk_users, j] - mu[topk_users]


    return mean_centered_topk_user_rate

def glm(user_sim_mat:np.array,item_sim_mat:np.array,r_train:np.ndarray,lam:float=1.,k_u:int=3,k_i:int=3,max_iter=100):

    sim_user = user_sim_mat
    sim_item = item_sim_mat

    w_uv = np.random.rand(r_train.shape[0], k_u)
    w_jt = np.random.rand(r_train.shape[1], k_i)

    row, col = np.nonzero(~np.isnan(r_train))

    alpha = 0.0001

    for it in trange(max_iter):

        grad_u = np.zeros(w_uv.shape)
        grad_i = np.zeros(w_jt.shape)
        total_err = 0
        for _u, _j in zip(row, col):
            mu_u = np.nanmean(r_train[_u,:])
            mu_j = np.nanmean(r_train[:,_j])

            # I got 'ValueError: operands could not be broadcast together with shapes (3,) (2,)' error I couldnt understand it so I use expection
            with suppress(ValueError):
                rate_pred = ((mu_u+mu_j)/2) + ((w_uv[_u,:] * top_k_sim_user_rate(sim_user,r_train,_u,_j,k_u)).sum() + (w_jt[_j,:] * top_k_sim_user_rate(sim_item,r_train.T,_j,_u,k_i)).sum())

                err = r_train[_u,_j] - rate_pred
                total_err += ((err ** 2)/2) + (lam/2)*(np.power(w_uv[_u,:],2).sum() + np.power(w_jt[_j,:],2).sum())

                w_uv_prev = w_uv
                w_jt_prev = w_jt

                w_uv[_u,:] -= (-err * top_k_sim_user_rate(sim_user,r_train,_u,_j) + (lam*(w_uv[_u,:]))) * alpha
                w_jt[_j,:] -= (-err * top_k_sim_user_rate(sim_item,r_train.T, _j, _u) + (lam*(w_jt[_j,:])))* alpha

        print(f"({it})") #w_uv: {w_uv}, w_jt: {w_jt}")#gradient_u: {np.linalg.norm(grad_u)}, gradient_i: {np.linalg.norm(grad_i)}")

        if (np.linalg.norm(w_uv - w_uv_prev) < 0.000001) & (np.linalg.norm(w_jt - w_jt_prev) < 0.000001):
            print(f"I do early stoping at iteration {i}")
            break

    return w_uv, w_jt


df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
names=['user_id', 'item_id', 'rating', 'timestamp'],nrows=1000) #Whole dataset takes too much time to converge

r = df.pivot(index='user_id', columns='item_id', values='rating').values

irow, jcol = np.where(~np.isnan(r))

print(f"{len(irow)} entries available")

idx = np.random.choice(np.arange(1000), 100, replace=False)
test_irow = irow[idx]
test_jcol = jcol[idx]

r_copy = r.copy()

for i in test_irow:
    for j in test_jcol:
        r_copy[i][j] = np.nan
r_train = r_copy.copy()

user_based = UserBased(beta=3, idf=True)
item_based = UserBased(beta=3, idf=True)

sim_user = user_based.fit(r_copy)
sim_item = item_based.fit(r_copy.T)

lambda_list=[0.25,0.5,0.75,1.,1.25,1.5,1.75,2.]
RMSE=[]

for lam_ in lambda_list:
    w_uv,w_jt = glm(sim_user,sim_item,r_train,lam_)

    err = []
    for u, j in zip(test_irow, test_jcol):
        mu_u = np.nanmean(r_train[u,:])
        mu_j = np.nanmean(r_train[:,j])

        with suppress(ValueError):
            y_pred = ((mu_u+mu_j)/2) + ((w_uv[u,:] * top_k_sim_user_rate(sim_user,r_train,u,j,3)).sum() +
                                        (w_jt[j,:] * top_k_sim_user_rate(sim_item,r_train.T,j,u,3)).sum())

            y = r[u,j]

            err.append(((y_pred - y) ** 2)/2)

    RMSE_=np.sqrt(np.nanmean(np.array(err)))
    print(f"RMSE: {np.sqrt(np.nanmean(np.array(err)))}")
    RMSE.append(RMSE_)


print(f"Optimum lamda: {lambda_list[np.argmin(RMSE)]}")

