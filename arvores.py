from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import operacoes

df = operacoes.filtrando_df()

X = df.iloc[:,1:]
y = df.iloc[:,:1]

y = operacoes.formatar_em_string(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

def arvore_decisao():
    arvore = DecisionTreeClassifier(random_state=42, criterion='entropy')
    arvore.fit(X_train, y_train)
    predictions = arvore.predict(X_test)

    operacoes.validacao(X_test, y_test, predictions, "Árvore de Decisão")
    #operacoes.indices_ordenados(arvore, X_test)
    #operacoes.plot_arvore(arvore, X)
    #operacoes.plot_matrix_confusao(y_test, predictions)
    #operacoes.curva_roc(arvore, X_test, y_test)
    operacoes.validacao_cruzada(arvore, X_train, y_train)
    #operacoes.matriz_correlacao(X_train)


def random_forest(num_arvores):
    rf = RandomForestClassifier(n_estimators=num_arvores, max_features='sqrt', criterion='entropy', max_depth=5, random_state=18)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    operacoes.validacao(X_test, y_test, y_pred, "Random Forrest")
    operacoes.plot_random_forest(rf, X, num_arvores)
    operacoes.plot_matrix_confusao(y_test, y_pred)
    operacoes.curva_roc(rf, X_test, y_test)
    operacoes.validacao_cruzada(rf, X_train, y_train)
    operacoes.matriz_correlacao(X)

arvore_decisao()

#random_forest(10)
