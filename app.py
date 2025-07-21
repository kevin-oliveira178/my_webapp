import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

modelo = joblib.load("modelo_vendas.pkl")

st.title("Vende ou não vende ?")
#coletjar os dados do novo cliente e armazenar numa variável 
st.sidebar.header("adicione as informações da venda")
qualidade = st.sidebar.selectbox(label="classifque a qualidade do produto", options=["baixa", "média", "alta"])
preco_unit = st.sidebar.slider("valor do unitário do produto", 0.0, 300.0)
distancia = st.sidebar.slider('distancia em kilometros',0.0, 250.0)
idade = st.sidebar.slider("Idade do comprador", 1, 80, 30)
desconto = st.sidebar.radio("foi aplicado desconto ?", ["sim", "não"])
tempo = st.sidebar.slider("quanto tempo o ciente permaneceu na página e minutos", 0, 90, 15)
entrega = st.sidebar.slider("tempo para entrega en dias", 0, 30, 15)
total = st.sidebar.slider("Preço total pago em R$", 0.0, 500.0, 20.0)

#armazenar informações num dataframe 
dados_venda = pd.DataFrame(
    {
        "qualidade": [qualidade],
        "preco_unitario": [preco_unit],
        "distancia_km": [distancia],
        "desconto_aplicado": [desconto],
        "idade_cliente": [idade],
        "tempo_site_min": [tempo],
        "preco_total":[total],
        "tempo_entrega_dias":[entrega],
        
    }
)

st.subheader("dados informações da venda")
st.write(dados_venda)
#mapear o valor da qualidade e dos descontos aplicados antes de jogar no modelo
dados_venda["desconto_aplicado"] = dados_venda["desconto_aplicado"].map({"sim": 1, "não": 0})
dados_venda["qualidade"] = dados_venda["qualidade"].map({"baixa": 1, "média": 2, "alta":3})

#decisão tomada no botão -  la ele
if st.button("a venda foi finalizada ?"):
    previsao = modelo.predict(dados_venda)[0]
    probabilidades = modelo.predict_proba(dados_venda)[0]
    
    st.subheader("Resultado da previsão")
    if previsao == 1:
        st.success("🎉 venda efetuada com sucesso!")
    else:
        st.error("venda não concluida!")

    st.subheader("Probabilidade da venda ser finalizada ")
    fig, ax = plt.subplots()
    ax.bar(["Não vende", "vende"], probabilidades)
    ax.set_ylabel("Probabilidade")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
