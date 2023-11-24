import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_csv('situacoes_alunos_2015-2020.csv')
df = df.drop(['DT_SIT_ALU', 'DS_BAIRRO', 'DS_CIDADE', 'DS_ESTADO', 'N_COLOCA', 'N_TOTESC', 'N_NOTRED'], axis=1)

df = df.fillna(0)

plt.hist(df['DS_SIT_ALU'])
plt.xlabel('Situações Aluno')
plt.ylabel('Frequência')
plt.show()

mapeamento = {'TRANSFERIDO': 1, 'GRADUADO': 2, 'CURSANDO': 3, 'ABANDONO': 4,  'DESISTENTE': 5, 'CANCELADO': 6}
df['DS_SIT_ALU'] = df['DS_SIT_ALU'].replace(mapeamento)

data = df.to_numpy()
X = data[:,1:]
y = data[:,:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

modelo_NB = MultinomialNB()
modelo_NB.fit(X_train, y_train)
previsoes = modelo_NB.predict(X_test)

matriz_confusao = confusion_matrix(y_test, previsoes)
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de Confusão')
plt.show()


''''
arvore = DecisionTreeClassifier(criterion='entropy')
arvore = arvore.fit(X_train, y_train)
tree.plot_tree(arvore)
plt.show()
'''