import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.read_csv('situacao_aluno_2021-2023.csv')
df = df.drop(['DT_SIT_ALU', 'N_COLOCA', 'N_TOTESC', 'N_NOTRED'], axis=1)
df = df.fillna(0)

evasao = {'CURSANDO': 'NAO EVADIU', 'ABANDONO': 'EVADIU',  'DESISTENTE': 'EVADIU'}
df['DS_SIT_ALU'] = df['DS_SIT_ALU'].map(evasao)

mapeamento = {'NAO EVADIU': 1, 'EVADIU': 2}
#df['DS_SIT_ALU'] = df['DS_SIT_ALU'].replace(mapeamento)

columns_mapping = ['DS_BAIRRO', 'DS_CIDADE', 'DS_ESTADO']
for coluna in columns_mapping:
    label_encoder = preprocessing.LabelEncoder()
    df[coluna] = label_encoder.fit_transform(df[coluna])
    mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

#data = df.to_numpy()
X = df.iloc[:,1:]
y = df.iloc[:,:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42, criterion='entropy')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Acurácia do Modelo:", accuracy)
print("Relatório de Classificação:\n", report)

plt.figure(figsize=(15, 10))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=y['DS_SIT_ALU'].unique(), fontsize=8)
plt.show()


def scatter(conjunto, classe, col1, col2):
    fig, ax = plt.subplots()
    fig = ax.scatter(x=conjunto[col1], y=conjunto[col2], c=classe, alpha=0.9, cmap='viridis')
    cbar = plt.colorbar(fig)
    cbar.set_label('DS_SIT_ALU')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Scatter Plot entre ' + col1 + ' e ' + col2)
    plt.show()

coluna1 = 'num_idade'
coluna2 = 'DS_CIDADE'
y_pred = predictions.reshape(-1, 1)
y_df_train = y_train.reshape(-1, 1)
y_df_test = y_test.reshape(-1, 1)
X_df_train = X_train[[coluna1, coluna2]]
X_df_test = X_test[[coluna1, coluna2]]

#scatter(X_train, y_df_train, coluna1, coluna2)

print(X_df_test[y_df_test != y_pred])

#scatter(X_test, y_pred, coluna1, coluna2)
#scatter(X_test, y_df_test, coluna1, coluna2)
