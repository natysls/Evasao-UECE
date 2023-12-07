from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import operacoes

ano = 2022
df = operacoes.filtrando_df(ano)

operacoes.aluno_evadidos(df)

X = df.iloc[:,1:]
y = df.iloc[:,:1]

y = operacoes.formatar_em_string(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

arvore = DecisionTreeClassifier(random_state=42, criterion='entropy')
arvore.fit(X_train, y_train)

def arvore_decisao():
    predictions = arvore.predict(X_test)

    operacoes.validacao(X_test, y_test, predictions, "Árvore de Decisão")
    operacoes.mais_validacao(y_test, predictions)
    operacoes.validacao_cruzada(arvore, X_train, y_train)
    operacoes.matriz_correlacao(df)
    operacoes.indices_ordenados(arvore, X_test)
    operacoes.plot_arvore(arvore, X)
    operacoes.plot_matrix_confusao(y_test, predictions)
    operacoes.curva_roc(arvore, X_test, y_test)


arvore_decisao()

def segunda_arvore_decisao():
    ano = 2023
    df = operacoes.filtrando_df(ano)

    alunos_evadidos = df[df['DS_SIT_ALU'] == 0]
    print("Alunos que evadiram:\n", alunos_evadidos)
    print("\n")

    X = df.iloc[:,1:]
    y = df.iloc[:,:1]

    y = operacoes.formatar_em_string(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    previsoes = arvore.predict(X_test)
    operacoes.validacao(X_test, y_test, previsoes, "Árvore de Decisão")
    operacoes.mais_validacao(y_test, previsoes)
    operacoes.validacao_cruzada(arvore, X_train, y_train)
    operacoes.matriz_correlacao(df)
    operacoes.indices_ordenados(arvore, X_test)
    operacoes.plot_arvore(arvore, X)
    operacoes.plot_matrix_confusao(y_test, previsoes)
    operacoes.curva_roc(arvore, X_test, y_test)


#segunda_arvore_decisao()