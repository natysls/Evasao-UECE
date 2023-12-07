import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
import operacoes

ano = 2022
df = operacoes.filtrando_df(ano)

X = df.iloc[:,1:]
y = df.iloc[:,:1]

y = operacoes.formatar_em_string(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

g_nb, m_nb, b_nb = GaussianNB(), MultinomialNB(alpha=1.0, fit_prior=True), BernoulliNB(alpha=1.0, fit_prior=True) 

def naive_bayes(model):
    model.fit(X_train, y_train)
    previsoes = model.predict(X_test)

    operacoes.validacao(X_test, y_test, previsoes, "Naive Bayes")
    operacoes.validacao_cruzada(model, X, y)
    operacoes.plot_matrix_confusao(y_test, previsoes)
    operacoes.curva_roc(model, X_test, y_test)
    operacoes.matriz_correlacao(X_train)

naive_bayes(g_nb)
#naive_bayes(m_nb)
#naive_bayes(b_nb)
