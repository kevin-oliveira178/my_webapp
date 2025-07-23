import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib 
from yaspin import yaspin



######################################## Tratamento dos dados ################################################
local_arquivo = "C:/Users/Oliveira/Desktop/projetosGit/fake_clientes_vendas_com_entrega.csv"

#ler o aqruivo como data frame 
df = pd.read_csv(local_arquivo)

#colunas de interesse 
df = df[['qualidade', 'preco_unitario'  , 'distancia_km','desconto_aplicado',
         'idade_cliente' ,'tempo_site_min' ,'compra_finalizada','preco_total','tempo_entrega_dias']]


#codificar a coluna qualidade em coluna com valores inteiros 
df["qualidade"] = df["qualidade"].map({"baixa": 1, "média": 2, "alta":3})

#limpar linhas com valores nulos e conferir se se está limpo usando o .info()
df = df.dropna()








#################################### construção do modelo ##################################################

#criar uma variável pros valores do "eixo X" retirando do df o valor que vai ser do eixo das ordendas 
X = df.drop('compra_finalizada', axis=1)
#definir o eixo das ordenadas 
y = df['compra_finalizada']



# dividir os dados em grupos de testes e de treino 
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)


# criaçao do modelo não treinado do tipo floresta aleatoria passando parametros com qiantidade de florestas 
# usadas e paramentro que garante a reprodução do modelo

modelo = RandomForestClassifier( n_estimators=400,       # mais árvores (teste com 200, 300…)
    max_depth=10,           # limitar profundidade (evita overfitting)
    min_samples_split=5,    # exige mais dados por divisão
    random_state=42)


#treina o modelo definodo antes atraves do metodo fit() que passa como parametro o eixo x e o eixo y de treino
modelo.fit(X_treino, y_treino)


#fase de predição: toma os dados presentes em testes e fazem uma predição que é armazenada em pred_y
y_pred = modelo.predict(X_teste)

#mede a precisão da previsão e armazena na variável acuracia 
acuracia = accuracy_score(y_teste, y_pred)
print(f"A acuracia do modelo é {acuracia * 100:2f}%")

#armazena o modelo em binário pra não precisar treinar ele de novo
joblib.dump(modelo, "modelo_vendas.pkl")

