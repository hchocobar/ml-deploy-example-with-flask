import os
from flask import Flask, request, render_template
from pickle import load


app = Flask(__name__)


model = load(open('iris-decision-tree-classifier-default-18.pkl', 'rb'))
class_dict = {"0": "Iris setosa",
              "1": "Iris versicolor",
              "2": "Iris virginica"}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html', prediction=None)
    if request.method == "POST":
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])
        data = [[val1, val2, val3, val4]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
        return render_template('index.html', prediction=pred_class)
    return None


if __name__ == '__main__':
    # PORT = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=3000, debug=True)