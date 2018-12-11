#pandas
'''https://pandas.pydata.org/pandas-docs/stable/10min.html#min'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Serie e uma tabela com uma coluna indice e uma coluna de dados
s = pd.Series([1,3,5,np.nan,6,8])
print(s)
#datas
dates = pd.date_range('20130101', periods=6)
print(dates)

#cria dataframe
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df)
#cria dataframe
df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })

print(df2)
print('tipos de dados do dataframe')
print(df2.dtypes)
#imprime as 5 primeiras colunas
print(df.head())
#imprime as 3 primeiras colunas
print(df.tail(3))
#imprime index
print(df.index)
#imprime cabecalho das colunas
print(df2.columns)
#imprime estatisticas basicas dos dados
print(df.describe())
#imprime a transposta
print(df.T)
#imprime segundo um eixo
print(df.sort_index(axis=1, ascending=False))
#imprime segundo a columa B
print(df.sort_values(by='B'))
#seleciona uma coluna
print(df['A'])
#seleciona linha 0, 1 e 2
print(df[0:3])
#seleciona as linhas entre as datas (index)
print(df['20130102':'20130104'])
#seleciona os elementos da linha 0
print(df.loc[dates[0]])
#seleciona os elementos da linha 1
print(df.loc[dates[1]])
#seleciona os elementos da linha 2
print(df.loc[dates[2]])
#seleciona todas as linhas das colunas A e B
print(df.loc[:,['A','B']])
#seleciona as linhas entre as datas sendo das colunas A e B
print(df.loc['20130102':'20130104',['A','B']])
#seleciona os elementos das columas A e B da linha especificada
print(df.loc['20130102',['A','B']])
#seleciona o elemento da linha 0 e coluna A
print(df.loc[dates[0],'A'])
print(df.at[dates[0],'A']) #mais rapido!
#todos os elementos da linha 3
print(df.iloc[3])
#elementos da linha 3 e 4 e das colunas A e B ou 0 e 1
#repare que a linha 5 não entra e a coluna 2 nao entra.
print(df.iloc[3:5,0:2])
#linhas 1, 2 e 4 e colunas 0 e 1
print(df.iloc[[1,2,4],[0,2]])
#todas as colunas das linhas 1 e 2
print(df.iloc[1:3,:])
#todas as linhas das colunas 1 e 2
print(df.iloc[:,1:3])
#le um unico valor
print(df.iloc[1,1])
print(df.iat[1,1]) #mais rapido!
#todas as linhas com valor de A maior que 0
print(df[df.A > 0])
#todas as linhas que forem maior que 0
print(df[df > 0])
#copia dataframe
df2 = df.copy()
#adiciona nova coluna E e os respectivos valores
df2['E'] = ['one', 'one','two','three','four','three']
print(df2)
#todas as linhas em que os valores two ou four estejam na coluna E
print(df2[df2['E'].isin(['two','four'])])
#cria series com valores 1 a 6 e com indices da data inicial até 6 periodos
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130101', periods=6))
print(s1)
#insere s1 na coluna F de df
df['F'] = s1
print(df)
#atribui 0 na linha 0 coluna A
df.at[dates[0],'A'] = 0
print(df)
#atribui 0 na linha 0 e coluna 1 ou B
df.iat[0,1] = 0
print(df)
#coloca um vetor de valores 5 na coluna D de df
df.loc[:,'D'] = np.array([5] * len(df))
print(df)
df2 = df.copy()
#onde os valores forem menor que 0, coloca os valores invertidos nos campos correspondentes.
df2[df2 > 0] = -df2
#reindex e especifica o lable das colunas
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
#coloca o valor 1 na coluna E da primeira e segunda linhas
df1.loc[dates[0]:dates[1],'E'] = 1
print(df1)
#apaga todas as linhas com valores NaN
df1 = df1.dropna(how='any')
print(df1)
#coloca o valor 5 onde for NaN
df1 = df1.fillna(value=5)
print(df1)
#testa se existe NaN em df1
print(pd.isna(df1))
#media por coluna
print(df.mean())
#media por linha
print(df.mean(1))

s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
print(s)
#subtrai s de df, elemento a elemento
print(df.sub(s,axis='index'))
#aplica a operacao np.cumsum ao dataframe
print(df.apply(np.cumsum))
#aplica lambda ao df
print(df.apply(lambda x: x.max() - x.min()))
#cria serie
s = pd.Series(np.random.randint(0, 7, size=10))
#conta valores (histograma)
xx = s.value_counts()
print(xx)
#cria series
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
#transforma para minusculo
print(s.str.lower())
#cria dataframe
df = pd.DataFrame(np.random.randn(10, 4))
#quebra em partes
pieces = [df[:3], df[3:7], df[7:]]
print(pieces)
concatenacao = pd.concat(pieces)
print(concatenacao)










