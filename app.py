from flask import Flask, render_template, request
import pickle 

app = Flask(__name__)

mnb = open("spam_classifier.pkl","rb")
model = pickle.load(mnb)

cvt = open("cv_transformation.pkl","rb")
transformer = pickle.load(cvt)

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = transformer.transform(data).toarray()
    	my_prediction = model.predict(vect)
    	return render_template('predict.html', prediction=my_prediction)
    
if __name__ == "__main__":
    app.run(debug = True)