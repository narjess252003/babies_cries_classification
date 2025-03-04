from flask import Flask
import numpy 
app = Flask(__name__)
@app.route('/')
def home():
    return "Baby Cry Classification API"
if __name__ == '__main__':
    app.run(debug=True)
