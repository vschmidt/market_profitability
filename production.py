import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Parâmetros
path = 'dados\\leads.xlsx'

#Importação
df_leads = pd.read_excel(path)

#Tratamento
df_leads['Origem do Cliente'] = LabelEncoder().fit_transform(df_leads['Origem do Cliente'])
df_leads['Última atividade'] = LabelEncoder().fit_transform(df_leads['Última atividade'])

entradas = df_leads.drop(['Conseguiu vender', 'ID do Cliente'],axis=1)
saidas = df_leads['Conseguiu vender']
df_final = StandardScaler().fit_transform(df_leads)
x_treino, x_teste, y_treino, y_teste = train_test_split(entradas, 
                                                        saidas, 
                                                        test_size=0.33, 
                                                        random_state=42)


#Treino do modelo
log_reg = LogisticRegression(solver="lbfgs")
log_reg.fit(x_treino, y_treino)
y_pred_proba = log_reg.predict_proba(x_teste)[:,1] #Probabilidades de fechar negócio

#Report
report = df_leads.copy()

report['% de fechar'] = log_reg.predict_proba(entradas)[:,1]
report = report.sort_values(by=['% de fechar'], ascending=False)
report['% de fechar'] = report['% de fechar'].apply(lambda w: "{:.2%}".format(w))

report = report.loc[:, ['ID do Cliente', '% de fechar']]

#Exportação para um arquivo Excel (Opicional)
report.to_excel('report.xlsx', index=False)
