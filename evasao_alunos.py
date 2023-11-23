import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('situacoes_alunos_2015-2020.csv')
df = df.drop(['DT_SIT_ALU', 'DS_BAIRRO', 'DS_CIDADE', 'DS_ESTADO'], axis=1)
""""
mapeamento = {'TRANSFERIDO': 1, 'GRADUADO': 2, 'CURSANDO': 3, 'ABANDONO': 4,  'DESISTENTE': 5, 'CANCELADO': 6}
df['DS_SIT_ALU'] = df['DS_SIT_ALU'].replace(mapeamento)
"""""
plt.hist(df['DS_SIT_ALU'])
plt.xlabel('Situações Aluno')
plt.ylabel('Frequência')
plt.show()

data = df.to_numpy()
X = data[:,1:]
y = data[:,:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
