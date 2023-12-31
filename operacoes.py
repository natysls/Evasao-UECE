import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn import tree
import seaborn as sns

def filtrando_df(ano):
    df = pd.read_csv('situacao_aluno_2021-2023.csv')

    columns_mapping = ['DS_BAIRRO', 'DS_CIDADE']
    for coluna in columns_mapping:
        label_encoder = preprocessing.LabelEncoder()
        df[coluna] = label_encoder.fit_transform(df[coluna])
        mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    df = df.fillna(0)

    # Preenchendo dados faltosos com a media
    colunas_faltosas = ['N_COLOCA', 'N_TOTESC', 'N_NOTRED']
    for coluna in colunas_faltosas:
        media_sem_zeros = df[df[coluna] != 0][coluna].mean()
        df[coluna] = df[coluna].replace(0, round(media_sem_zeros))

    df = df.loc[df['CD_ANO_INGRESSO'] == ano]

    df = df.drop(['aluno', 'CD_ANO_INGRESSO', 'DT_SIT_ALU', 'DS_ESTADO'], axis=1)

    evasao = {'CURSANDO': 'NAO EVADIU', 'ABANDONO': 'EVADIU',  'DESISTENTE': 'EVADIU'}
    df['DS_SIT_ALU'] = df['DS_SIT_ALU'].map(evasao)
    mapeamento = {'EVADIU': 0, 'NAO EVADIU': 1}
    df['DS_SIT_ALU'] = df['DS_SIT_ALU'].replace(mapeamento)

    df['BOLSAS_COTA_PRIORIDADE'] = df['BOLSAS_COTA_PRIORIDADE'].replace(99, 4)

    return df

def aluno_evadidos(df):
    alunos_evadidos = df[df['DS_SIT_ALU'] == 0]
    print("Alunos que evadiram:\n", alunos_evadidos)
    print("\n")

def formatar_em_string(y):
    y_string = []
    for valor in y.values:
        palavra = str(valor)
        y_string.append(palavra)
    y_string = np.array(y_string) 
    return y_string

def validacao(df_X_test, df_y_test, df_y_pred, text):
    print("Modelo de", text)
    df_real_pred = pd.DataFrame({'Valores reais':df_y_test, 'Valores previstos':df_y_pred})
    print(df_real_pred)

    print(f"Quantidade de ocorrências de 0 - EVASÃO: {np.count_nonzero(df_y_pred == '[0]')}")
    print(f"Quantidade de ocorrências de 1 - NÃO EVASÃO: {np.count_nonzero(df_y_pred == '[1]')}")

    print("Valores que o modelo errou:\n", df_X_test[df_y_test != df_y_pred])
    print("\n")

    df_X_acertos = df_X_test[df_y_test == df_y_pred]
    df_y_acertos = df_y_test[df_y_test == df_y_pred]

    X_test_reset = df_X_acertos.reset_index(drop=True)
    y_test_reset = pd.Series(df_y_acertos, name='DS_SIT_ALU').reset_index(drop=True)

    df_combinado = pd.concat([X_test_reset, pd.Series(y_test_reset, name='DS_SIT_ALU')], axis=1)
    print("Alunos que o modelo acertou:\n", df_combinado)
    print("\n")

    alunos_evadidos = df_combinado[df_combinado['DS_SIT_ALU'].apply(lambda x: '[0]' in x)]
    print("Alunos que o modelo acertou que evadiram:\n", alunos_evadidos)
    print("\n")

    accuracy = accuracy_score(df_y_test, df_y_pred)
    print("Acurácia do Modelo:", accuracy)
    print("\n")

def mais_validacao(df_y_test, df_y_pred):
    precisao = precision_score(df_y_test, df_y_pred, average='weighted', zero_division=0)
    recall = recall_score(df_y_test, df_y_pred, average='weighted')
    f1 = f1_score(df_y_test, df_y_pred, average='weighted')

    print(f'Precisão: {precisao:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F-Score: {f1:.4f}\n')

def indices_ordenados(arvore, df_X_test):
    indices_ordenados = np.argsort(arvore.feature_importances_)[::-1]
    print("Características mais importantes:")
    for indice in indices_ordenados:
        print(f"C {df_X_test.columns[indice]}: Importância = {arvore.feature_importances_[indice]}")


def plot_arvore(arvore, df_X):
    plt.figure(figsize=(15, 12))
    tree.plot_tree(arvore, filled=True, feature_names=df_X.columns, class_names=arvore.classes_, fontsize=8)
    plt.show()

def plot_random_forest(arvore, df_X, numero_arvore):
    #for i in range(numero_arvore):
        plt.figure(figsize=(15, 12))
        tree.plot_tree(arvore.estimators_[numero_arvore-1], feature_names=df_X.columns, class_names=arvore.classes_, filled=True, rounded=True, fontsize=8)
        plt.show()

def plot_matrix_confusao(df_y_test, df_y_pred):
    matriz_confusao = confusion_matrix(df_y_test, df_y_pred)
    print("Matriz de Confusão\n", matriz_confusao, "\n")
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predições')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.text(0.5, 0.3, f"Evasores evadiram", horizontalalignment='center', verticalalignment='center')
    plt.text(1.5, 0.3, f"Evasores não evadiram", horizontalalignment='center', verticalalignment='center')
    plt.text(0.5, 1.3, f"Não evasores evadiram", horizontalalignment='center', verticalalignment='center')
    plt.text(1.5, 1.3, f"Não evasores não evadiram", horizontalalignment='center', verticalalignment='center')
    plt.show()

def curva_roc(model, df_X_test, df_y_test):
    probabilidade_positivo_classe = model.predict_proba(df_X_test)
    y_test_binarized = preprocessing.label_binarize(df_y_test, classes=np.unique(df_y_test))
    taxa_fpr, taxa_tpr, thresholds = roc_curve(y_test_binarized, probabilidade_positivo_classe[:,1])
    auc_score = auc(taxa_fpr, taxa_tpr)
    print("Acurácia da área sob a curva (ROC):", auc_score)
    print("\n")

    plt.plot(taxa_fpr, taxa_tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()

def validacao_cruzada(model, df_X, df_y):
    cross_val_results = cross_val_score(model, df_X, df_y, cv=5, scoring='accuracy')
    print("Resultados da Validacao Cruzada:", cross_val_results)
    print("Precisão Média: {:.2f}%".format(cross_val_results.mean() * 100))
    print("\n")

def matriz_correlacao(df):
    matriz_correlacao = df.corr()
    correlacoes = matriz_correlacao['DS_SIT_ALU'].sort_values(ascending=False)
    print("Características mais se correlacionam com DS_SIT_ALU:\n", correlacoes)

    sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt='.2f')
    plt.show()

def matriz_correlacao_aprimorada(df_X_test, df_y_test, df_y_pred):
    df_X_acertos = df_X_test[df_y_test == df_y_pred]
    df_y_acertos = df_y_test[df_y_test == df_y_pred]

    X_test_reset = df_X_acertos.reset_index(drop=True)
    y_test_reset = pd.Series(df_y_acertos, name='DS_SIT_ALU').reset_index(drop=True)

    df_combinado = pd.concat([X_test_reset, pd.Series(y_test_reset, name='DS_SIT_ALU')], axis=1)
    print("Alunos que o modelo acertou:\n", df_combinado)
    print("\n")

    mapeamento = {'[0]': 0, '[1]': 1}
    df_combinado['DS_SIT_ALU'] = df_combinado['DS_SIT_ALU'].replace(mapeamento)

    matriz_correlacao = df_combinado.corr()
    sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt='.2f')
    plt.show()

def scatter(df, col1, col2):
    plt.scatter(df[col1], df[col2], linestyle='--')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Scatter Plot entre ' + col1 + ' e ' + col2)
    plt.show()
