from flask import Flask,render_template,url_for,redirect,request,session
import network

PATH_TO_CITED = "./data/cited.pkl"
PATH_TO_CITED_ENCODE = "./data/cited_encode.pkl"
COLORS = ['red', 'orange', 'green', 'blue', 'purple']
TRIM_NUM = 2

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/demo', methods=['GET', 'POST'])
@app.route('/demo/input', methods=['GET', 'POST'])
def demo():
    if request.method == 'POST':
        text = request.form['text']
        if request.form['submit_button'] == "submit":
            return redirect(url_for("outcome", text=text))
        else:
            return redirect(url_for("outcomeCluster", text=text))
    return render_template('input.html')

@app.route('/demo/outcome/<text>/')
def outcome(text):
    candidates = network.recommend(text)
    return render_template('outcome.j2', text=text, candidates=candidates)

@app.route('/demo/outcomeCluster/<text>/')
def outcomeCluster(text):
    candidateClusters = network.recommendCluster(text)
    return render_template('outcomeCluster.j2', text=text, candidateClusters=candidateClusters)

@app.route('/author')
def author():
    return render_template('author.html')
