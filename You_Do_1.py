import random

import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing


def _model2(x, y, lam, alpha=0.0001,teta:float =3.) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)
    loss = []

    for i in range(100):
        g_b0 = 0
        g_b1 = 0
        _loss = 0

        for __x,__y in zip(x,y):
            __y_pred = beta[0] + beta[1] * __x
            __e = __y - __y_pred

            if abs(__e) <= teta:
                _loss += np.power(__e, 2) + lam*(np.power(beta[0],2) + np.power(beta[1],2))
                g_b0 += -2 * (__y - __y_pred) + 2 * lam * beta[0]
                g_b1 += -2 * (__x * (__y - __y_pred)) + 2 * lam * beta[1]

            else:
                if __e > 0:
                    _loss += np.log(__e - (teta - 1)) + np.power(teta, 2) + lam*(np.power(beta[0],2) + np.power(beta[1],2))
                    g_b0 += -1/( (__y - __y_pred) +1 -teta ) + 2 * lam * beta[0]
                    g_b1 += -__x/( (__y - __y_pred) +1 -teta )  + 2 * lam * beta[1]
                else:
                    _loss += np.log(-__e - (teta - 1)) + np.power(teta, 2) + lam*(np.power(beta[0],2) + np.power(beta[1],2))
                    g_b0 += +1/( -(__y - __y_pred) +1 -teta ) + 2 * lam * beta[0]
                    g_b1 += +__x/( -(__y - __y_pred) +1 -teta ) + 2 * lam * beta[1]

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)
        loss.append(_loss)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta , loss


def main(verbose: bool = True):

    st.header("Let's Generate a Blob")

    st.header("Dataset")
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    #x=X[MedInc]

    st.write(X)
    x=X.iloc[:,0]
    st.write(x)
    st.write(y)

    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=cal_housing.target))

    st.dataframe(df)

    teta=5.
    lam=0.5

    st.write("""I could not find convex error function such that sustain conditions. I use function below but
    it isn't convex so my gradients don't converge but I write the script anyway as a practice
        """)

    st.latex(
        r"L(y={ x^{2} + \lambda (\beta_0^2 + \beta_1^2): x^{2}\le t^{2}})")
    st.latex(
        r"L(y={\ln\left(x-\left(t-1\right)\right)+\left(t^{2}\right) + \lambda (\beta_0^2 + \beta_1^2) : x^{2}>t^{2}\ , x>0})")
    st.latex(
        r"L(y={\ln\left(-x-\left(t-1\right)\right)+\left(t^{2}\right) + \lambda (\beta_0^2 + \beta_1^2) : x^{2}>t^{2}\ , x<0})")


    beta1, loss = _model2(x, y, lam, alpha=0.0001, teta=3.)
    print(beta1)
    print(loss)

    if verbose:
        loss, b0, b1 = [], [], []
        for i, _b0 in enumerate(np.linspace(-100, 100, 50)):

            for _b1 in np.linspace(-100, 100, 50):
                b0.append(_b0)
                b1.append(_b1)
                _loss=0

                for __x,__y in zip(x,y):
                    __y_pred=_b0 + _b1*__x
                    __e=__y-__y_pred

                    if abs(__e) <= teta :
                        _loss += np.power(__e,2) + lam*(np.power(_b0,2) + np.power(_b1,2))
                    else:
                        if __e > 0:
                            _loss += np.log(__e - (teta -1 )) + np.power(teta,2) + lam*(np.power(_b0,2) + np.power(_b1,2))
                        else:
                            _loss += np.log(-__e - (teta - 1)) + np.power(teta, 2) + lam*(np.power(_b0,2) + np.power(_b1,2))

                loss.append(_loss)

        l = pd.DataFrame(dict(b0=b0, b1=b1, loss=loss))
        st.dataframe(l)

        fig = px.scatter(l, x="b1", y="loss")

        st.plotly_chart(fig, use_container_width=True)







if __name__ == '__main__':
    main(verbose=st.sidebar.checkbox("Verbose"))

