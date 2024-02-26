import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file

book_pivot = pd.read_csv("data/book_pivot.csv")
book_pivot.set_index(book_pivot.columns[0], inplace=True)
model = pickle.load(open("models/knn_model_brs.sav", 'rb'))


app = Flask(__name__)

@app.route("/")
def home() :
    return render_template("index.html")


@app.route("/about")
def about() :
    return render_template("about.html")

@app.route("/material")
def material() :
    return render_template("material.html")

@app.route("/resume")
def resume() :
    return render_template("resume.html")

@app.route("/predict", methods=["GET", "POST"])
def predict() :
    if request.method == 'POST':
        mess = request.form
        val = []
        for key, value in mess.items():
            val.append(value)
        val[1] = int(val[1])
        
        recommended = []
        try :
            book_id = np.where(book_pivot.index == val[0])[0][0]
            distance, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=val[1]+1)
            for i in range(len(suggestions)) :
                if not i :
                    recommended.append(book_pivot.index[suggestions[i]])
            arr = np.array(recommended[0])
            book_name = arr[0]
            arr = np.delete(arr, 0)
            return render_template("result.html", result=arr, name=book_name)
        except :
            recommended = ["'We could not find any of your book in our system, Hence we cant recommend!, SORRY'"]
            arr = np.array(recommended)
            return render_template("result.html", result=arr, name=val[1])
        
        
    

if __name__ == "__main__" :
    app.run(debug=True)







