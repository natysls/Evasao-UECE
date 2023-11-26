import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.read_csv('situacoes_alunos_2015-2020.csv')

df = df.loc[df['CD_ANO_INGRESSO'] == 2017]
df = df.loc[df['CD_SEM_INGRESSO'] == 2]

df = df.drop(['CD_ANO_INGRESSO', 'CD_SEM_INGRESSO', 'DT_SIT_ALU', 'N_COLOCA', 'N_TOTESC', 'N_NOTRED'], axis=1)
df = df.fillna(0)

label_encoder = preprocessing.LabelEncoder()
df['DS_BAIRRO'] = label_encoder.fit_transform(df['DS_BAIRRO'])
mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

df['DS_CIDADE'] = label_encoder.fit_transform(df['DS_CIDADE'])
mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

df['DS_ESTADO'] = label_encoder.fit_transform(df['DS_ESTADO'])
mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

data = df.to_numpy()
X = data[:,1:]
y = data[:,:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42, criterion='entropy')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Acurácia do Modelo:", accuracy)
print("Relatório de Classificação:\n", report)

nomes_das_colunas_X = df.iloc[:,1:].columns.tolist()
X_df = pd.DataFrame(X, columns=nomes_das_colunas_X) 

nomes_das_colunas_y = df.iloc[:,:1].columns.tolist()
y_df = pd.DataFrame(y, columns=nomes_das_colunas_y) 

plt.figure(figsize=(15, 12))
tree.plot_tree(model, filled=True, feature_names=X_df.columns, class_names=y_df['DS_SIT_ALU'].unique(), fontsize=8)
plt.show()
