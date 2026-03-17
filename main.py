from flask import Flask, render_template, request
from Feature_Engineering_Selection.Vectoization import LemmaTokenizer
import joblib

app = Flask(__name__)

model = joblib.load("sentiment_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form["review"]
        pred = model.predict([review_text])[0]

        if pred == 1:
            prediction = "Positive 😊"
        else:
            prediction = "Negative 😞"

    return render_template("index.html", prediction=prediction, review_text=review_text)

if __name__ == "__main__":
    app.run(debug=True)