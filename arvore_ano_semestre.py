import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns

df = pd.read_csv('situacao_aluno_2021-2023.csv')
df = df.drop(['aluno', 'DT_SIT_ALU', 'DS_BAIRRO'], axis=1)
df = df.fillna(0)

# Preenchendo dados faltosos com a media
colunas_faltosas = ['N_COLOCA', 'N_TOTESC', 'N_NOTRED']
for coluna in colunas_faltosas:
    media_sem_zeros = df[df[coluna] != 0][coluna].mean()
    df[coluna] = df[coluna].replace(0, round(media_sem_zeros))

evasao = {'CURSANDO': 'NAO EVADIU', 'ABANDONO': 'EVADIU',  'DESISTENTE': 'EVADIU'}
df['DS_SIT_ALU'] = df['DS_SIT_ALU'].map(evasao)
mapeamento = {'NAO EVADIU': 1, 'EVADIU': 2}
df['DS_SIT_ALU'] = df['DS_SIT_ALU'].replace(mapeamento)

columns_mapping = ['DS_CIDADE', 'DS_ESTADO']
for coluna in columns_mapping:
    label_encoder = preprocessing.LabelEncoder()
    df[coluna] = label_encoder.fit_transform(df[coluna])
    mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

df = df.loc[df['CD_ANO_INGRESSO'] == 2022]
#df = df.loc[df['CD_SEM_INGRESSO'] == 1]

X = df.iloc[:,1:]
y = df.iloc[:,:1]

def formatar_em_string(y):
    y_string = []
    for valor in y.values:
        palavra = str(valor)
        y_string.append(palavra)
    y_string = np.array(y_string) 
    return y_string
y = formatar_em_string(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

arvore_decisao = DecisionTreeClassifier(random_state=42, criterion='entropy')
arvore_decisao.fit(X_train, y_train)
predictions = arvore_decisao.predict(X_test)

def validacao(df_X, df_y, arvore, df_X_test, df_y_test, df_y_pred, text):
    print("Modelo de", text)
    df_real_pred = pd.DataFrame({'Valores reais':df_y_test, 'Valores previstos':df_y_pred})
    print(df_real_pred)

    accuracy = accuracy_score(df_y_test, df_y_pred)
    cross_val_results = cross_val_score(arvore, df_X, df_y, cv=5, scoring='accuracy')

    print(f"Quantidade de ocorrências de 1-NÃO EVASÃO: {np.count_nonzero(df_y_pred == '[1]')}")
    print(f"Quantidade de ocorrências de 2-EVASÃO: {np.count_nonzero(df_y_pred == '[2]')}")
    print("Acurácia do Modelo:", accuracy)
    print("Precisão Média da Validacao Cruzada: {:.2f}%".format(cross_val_results.mean() * 100))

    indices_ordenados = np.argsort(arvore.feature_importances_)[::-1]
    print("Características mais importantes:")
    for indice in indices_ordenados:
        print(f"C {df_X_test.columns[indice]}: Importância = {arvore.feature_importances_[indice]}")

    print("Valores que o modelo errou:\n", df_X_test[df_y_test != df_y_pred])
    print("\n")

validacao(X, y, arvore_decisao, X_test, y_test, predictions, "Árvore de Decisão")

def plot_arvore(arvore):
    plt.figure(figsize=(15, 12))
    tree.plot_tree(arvore, filled=True, feature_names=X.columns, class_names=arvore.classes_, fontsize=8)
    plt.show()

#plot_arvore(arvore_decisao)

def plot_confusao(df_y_test, df_y_pred):
    matriz_confusao = confusion_matrix(df_y_test, df_y_pred)
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predições')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.text(0.5, 0.3, f"Verdadeiro Positivo", horizontalalignment='center', verticalalignment='center')
    plt.text(1.5, 0.3, f"Falso Negativo", horizontalalignment='center', verticalalignment='center')
    plt.text(0.5, 1.3, f"Falso Positivo", horizontalalignment='center', verticalalignment='center')
    plt.text(1.5, 1.3, f"Verdadeiro Negativo", horizontalalignment='center', verticalalignment='center')
    plt.show()

#plot_confusao(y_test, predictions)

rf = RandomForestClassifier(n_estimators=300, max_features='sqrt', max_depth=5, random_state=18)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
validacao(X, y, rf, X_test, y_test, y_pred, "Random Forrest")

