from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import operacoes

ano = 2022
df = operacoes.filtrando_df(ano)

X = df.iloc[:,1:]
y = df.iloc[:,:1]

y = operacoes.formatar_em_string(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

def arvore_decisao():
    arvore = DecisionTreeClassifier(random_state=42, criterion='entropy')
    arvore.fit(X_train, y_train)
    predictions = arvore.predict(X_test)

    operacoes.validacao(X_test, y_test, predictions, "Árvore de Decisão")
    operacoes.indices_ordenados(arvore, X_test)
    operacoes.plot_arvore(arvore, X)
    operacoes.plot_matrix_confusao(y_test, predictions)
    operacoes.curva_roc(arvore, X_test, y_test)
    operacoes.validacao_cruzada(arvore, X_train, y_train)
    operacoes.matriz_correlacao(X_train)


arvore_decisao()

