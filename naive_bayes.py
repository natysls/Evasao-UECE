import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
import operacoes

ano = 2022
df = operacoes.filtrando_df(ano)

#operacoes.aluno_evadidos(df)

X = df.iloc[:,1:]
y = df.iloc[:,:1]

y = operacoes.formatar_em_string(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

g_nb, m_nb, b_nb = GaussianNB(), MultinomialNB(alpha=1.0, fit_prior=True), BernoulliNB(alpha=1.0, fit_prior=True) 
m_nb.fit(X_train, y_train)

def naive_bayes():
    previsoes = m_nb.predict(X_test)

    operacoes.validacao(X_test, y_test, previsoes, "Naive Bayes Multinomial")
    operacoes.mais_validacao(y_test, previsoes)
    operacoes.validacao_cruzada(m_nb, X, y)
    operacoes.matriz_correlacao(df)
    operacoes.plot_matrix_confusao(y_test, previsoes)
    operacoes.curva_roc(m_nb, X_test, y_test)

naive_bayes()

def segundo_naive_bayes():
    ano = 2023
    df = operacoes.filtrando_df(ano)

    X = df.iloc[:,1:]
    y = df.iloc[:,:1]

    y = operacoes.formatar_em_string(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

    previsoes = m_nb.predict(X_test)

    operacoes.validacao(X_test, y_test, previsoes, "Naive Bayes Multinomial")
    operacoes.mais_validacao(y_test, previsoes)
    operacoes.validacao_cruzada(m_nb, X, y)
    operacoes.matriz_correlacao(df)
    operacoes.plot_matrix_confusao(y_test, previsoes)
    operacoes.curva_roc(m_nb, X_test, y_test)

#segundo_naive_bayes()