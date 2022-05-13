from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
app = Flask(__name__, static_url_path='/static') 

lemmatizer = WordNetLemmatizer()

@app.route("/", methods=['GET'])
def index():
    # if user query in url query string, run query and return results
    query = request.args.get('q')
    if query:
        tokens = [lemmatizer(token) for token in word_tokenize(query) if token not in stopwords]

        # TODO: index
        # TODO: search
        # TODO: suggestions

    # else simply render page
    return render_template('index.html')  # return 'hello world!'

if __name__ == "__main__":
    app.run(host='localhost', port=5000) # debug=True
