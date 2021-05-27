from flask import Flask, render_template, request

app = Flask(__name__)

model = None


def detectLanguage(text):
    import numpy as np
    import string
    import re
    import pickle

    # replacing puncutations with None or ''
    translate_table = {ord(char): None for char in string.punctuation}

    with open('languagedetection/model/LDModel.pkl', 'rb') as files:
        model = pickle.load(files)

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(translate_table)
    pred = model.predict([text])

    return pred[0]


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():

    if request.method != 'POST':
        return render_template("index.html")

    Text = request.form['text']

    output = detectLanguage(Text)

    return render_template('index.html', Detected_Language=f'Language is {output}')


if __name__ == '__main__':
    app.run(debug=True)
