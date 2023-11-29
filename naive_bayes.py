import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

df = pd.read_csv('situacao_aluno_2021-2023.csv')
df = df.drop(['DT_SIT_ALU', 'DS_BAIRRO', 'DS_CIDADE', 'DS_ESTADO', 'N_COLOCA', 'N_TOTESC', 'N_NOTRED'], axis=1)
df = df.fillna(0)
''''
plt.hist(df['DS_SIT_ALU'])
plt.xlabel('Situações Aluno')
plt.ylabel('Frequência')
plt.show()
'''
evasao = {'CURSANDO': 'NAO EVADIU', 'ABANDONO': 'EVADIU',  'DESISTENTE': 'EVADIU'}
df['DS_SIT_ALU'] = df['DS_SIT_ALU'].map(evasao)

mapeamento = {'NAO EVADIU': 1, 'EVADIU': 2}
df['DS_SIT_ALU'] = df['DS_SIT_ALU'].replace(mapeamento)
''''
mapeamento = {'TRANSFERIDO': 1, 'GRADUADO': 2, 'CURSANDO': 3, 'ABANDONO': 4,  'DESISTENTE': 5, 'CANCELADO': 6}
df['DS_SIT_ALU'] = df['DS_SIT_ALU'].replace(mapeamento)
'''
data = df.to_numpy()
X = data[:,1:]
y = data[:,:1]
y = y.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

g_nb, m_nb, b_nb = GaussianNB(), MultinomialNB(alpha=1.0, fit_prior=True), BernoulliNB(alpha=1.0, fit_prior=True) 
m_nb.fit(X_train, y_train)
previsoes = m_nb.predict(X_test)

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

