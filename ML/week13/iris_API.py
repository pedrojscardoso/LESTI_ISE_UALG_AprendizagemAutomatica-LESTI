import Orange
import pickle
import numpy as np
from flask import Flask, jsonify, Response


# carega o modelo gravado no Orange
model_iris = pickle.load(open("iris_tree.pkcls", "rb"))

# carrega o dataset iris do Orange para extrairmos alguma informação a apresentar na API
data = Orange.data.Table("iris")
classes = data.domain.class_var.values  # ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
columns = data.X_df.columns  # ['sepal length', 'sepal width', 'petal length', 'petal width']


# ------------------------------------
# Vamos utilizar o Flask para criar um serviço
app = Flask(__name__)

# Definição de rotas
@app.route("/usage")  # se for pedido o modo de utilizaçao
def usage():
    return jsonify(
        {
            "route_order": "/".join(columns),
            "usage example": "http://<host>:<port>/" + "/".join(map(str, list(data.X_df.values[0])))
        }
    )


@app.route("/<a>/<b>/<c>/<d>", methods=["GET"])
def predict(a, b, c, d):
    # cria uma instancia de valores para ['sepal length', 'sepal width', 'petal length', 'petal width']
    instance = np.array([[a, b, c, d]], dtype=float)
    
    # usa o modelo carregado para calcular as probabilidades de cada classe ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    pred = model_iris.predict(instance).tolist()[0]

    # obtem o indice com maior probabilidade
    idx = np.argmax(pred)

    return jsonify(
        {
            "class": classes[idx],
            "prob": pred[idx],
            "details": {
                "classes": classes,
                "probs": pred
            },
            "code": 200  # para ajudar na verificação de um "200 - OK"
        }
    ), 200


# trata, p.e., de rotas mal definidas
@app.errorhandler(404)
def url_mal_formado(e):
    return jsonify(
        {
            "code": e.code,
            "usage": usage().json
        }
    ), 404


if __name__ == "__main__":
    # coloca o serviço a correr
    app.run(
        host='0.0.0.0',  # Configuração para tratar pedidos remotos (de outras máquinas)
        debug=True
    )
