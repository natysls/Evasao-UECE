import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.read_csv('situacoes_alunos_2015-2020.csv')

df = df.loc[df['CD_ANO_INGRESSO'] == 2017]
df = df.loc[df['CD_SEM_INGRESSO'] == 2]

df = df.drop(['CD_ANO_INGRESSO', 'CD_SEM_INGRESSO', 'DT_SIT_ALU', 'N_COLOCA', 'N_TOTESC', 'N_NOTRED'], axis=1)
df = df.fillna(0)

evasao = {'TRANSFERIDO': 'NAO EVADIU', 'GRADUADO': 'NAO EVADIU', 'CURSANDO': 'NAO EVADIU', 'ABANDONO': 'EVADIU',  'DESISTENTE': 'EVADIU', 'CANCELADO': 'EVADIU'}
df['DS_SIT_ALU'] = df['DS_SIT_ALU'].map(evasao)

columns_mapping = ['DS_BAIRRO', 'DS_CIDADE', 'DS_ESTADO']
for coluna in columns_mapping:
    label_encoder = preprocessing.LabelEncoder()
    df[coluna] = label_encoder.fit_transform(df[coluna])
    mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

X = df.iloc[:,1:]
y = df.iloc[:,:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42, criterion='entropy')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Acurácia do Modelo:", accuracy)

plt.figure(figsize=(15, 12))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=y['DS_SIT_ALU'].unique(), fontsize=8)
plt.show()


coluna1 = 'aluno'
coluna2 = 'DS_BAIRRO'
y_pred = predictions.reshape(-1, 1)

X_df_train = X_train[[coluna1, coluna2]]
X_df_test = X_test[[coluna1, coluna2]]
y_df_train = y_train.values.reshape(-1, 1)
y_df_test = y_test.values.reshape(-1, 1)

print(X_df_test[y_df_test != y_pred])

fig, ax = plt.subplots()
#plt.scatter(x=X_train[coluna1], y=X_train[coluna2], c=y_df_train, alpha=0.9, cmap='viridis')
plt.scatter(x=X_test[coluna1], y=X_test[coluna2], c=y_pred, alpha=0.9, cmap='viridis')
plt.scatter(x=X_test[coluna1], y=X_test[coluna2], c=y_df_test, alpha=0.2, cmap='viridis')
cbar = plt.colorbar()
cbar.set_label('DS_SIT_ALU')

plt.xlabel(coluna1)
plt.ylabel(coluna2)
plt.title('Scatter Plot entre ' + coluna1 + ' e ' + coluna2)
plt.show()
