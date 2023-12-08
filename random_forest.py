from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import operacoes

ano = 2022
df = operacoes.filtrando_df(ano)

#operacoes.aluno_evadidos(df)

X = df.iloc[:,1:]
y = df.iloc[:,:1]

y = operacoes.formatar_em_string(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

num_arvores = 5

rf = RandomForestClassifier(n_estimators=num_arvores, max_features='sqrt', criterion='entropy', max_depth=5, random_state=18)
rf.fit(X_train, y_train)

def random_forest():
    y_pred = rf.predict(X_test)

    operacoes.validacao(X_test, y_test, y_pred, f"Random Forrest com {num_arvores} árvores")
    operacoes.mais_validacao(y_test, y_pred)
    operacoes.validacao_cruzada(rf, X, y)
    operacoes.matriz_correlacao(df)
    #operacoes.plot_random_forest(rf, X, num_arvores)
    operacoes.plot_matrix_confusao(y_test, y_pred)
    operacoes.curva_roc(rf, X_test, y_test)
    
random_forest()

def segundo_random_forest():
    ano = 2023
    df = operacoes.filtrando_df(ano)

    X = df.iloc[:,1:]
    y = df.iloc[:,:1]

    y = operacoes.formatar_em_string(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    y_pred = rf.predict(X_test)

    operacoes.validacao(X_test, y_test, y_pred, f"Random Forrest com {num_arvores} árvores")
    operacoes.mais_validacao(y_test, y_pred)
    operacoes.validacao_cruzada(rf, X, y)
    operacoes.matriz_correlacao(df)
    operacoes.plot_random_forest(rf, X, num_arvores)
    operacoes.plot_matrix_confusao(y_test, y_pred)
    operacoes.curva_roc(rf, X_test, y_test)

#segundo_random_forest()
