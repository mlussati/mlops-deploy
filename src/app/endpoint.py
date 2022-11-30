import pickle
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob

colunas = ['tamanho', 'ano', 'garagem']
modelo = pickle.load(open('models/modelo.sav', 'rb'))

app = Flask("meu_app")
app.config['BASIC_AUTH_USERNAME'] = ''
app.config['BASIC_AUTH_PASSWORD'] = ''

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Minha aplicação nova"


@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br', to='en')
    polaridade = tb_en.sentiment.polarity
    return f"polaridade: {polaridade}"


@app.route('/cotacao/', methods=["POST"])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]

    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])


app.run(debug=True)
