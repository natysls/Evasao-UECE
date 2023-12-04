import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

df = pd.read_csv('situacao_aluno_2021-2023.csv')
df = df.drop(['aluno', 'DT_SIT_ALU', 'DS_BAIRRO'], axis=1)
df = df.fillna(0)
''''
plt.hist(df['DS_SIT_ALU'])
plt.xlabel('Situações Aluno')
plt.ylabel('Frequência')
plt.show()
'''

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

g_nb, m_nb, b_nb = GaussianNB(), MultinomialNB(alpha=1.0, fit_prior=True), BernoulliNB(alpha=1.0, fit_prior=True) 
m_nb.fit(X_train, y_train)
previsoes = m_nb.predict(X_test)

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

    print("Valores que o modelo errou:\n", df_X_test[df_y_test != df_y_pred])
    print("\n")

validacao(X, y, m_nb, X_test, y_test, previsoes, "Naive Bayes Multinomial")


matriz_confusao = confusion_matrix(y_test, previsoes)
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de Confusão')
plt.show()


probabilidade_positivo_classe = m_nb.predict_proba(X_test)
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

taxa_fpr, taxa_tpr, thresholds = roc_curve(y_test_binarized, probabilidade_positivo_classe[:,1])
auc_score = auc(taxa_fpr, taxa_tpr)

cross_val_results = cross_val_score(m_nb, X, y, cv=5, scoring='accuracy')
print("Resultados da Validacao Cruzada:", cross_val_results)
print("Precisão Média: {:.2f}%".format(cross_val_results.mean() * 100))
print("Acurácia:", auc_score)

plt.plot(taxa_fpr, taxa_tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()

