from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

import os
# print(os.listdir("."))
from collections import Counter # A counter is a container that stores elements as dictionary keys, and their counts are stored as dictionary values.
app = Flask(__name__, static_url_path='/static') 

lemmatizer = WordNetLemmatizer()
CORPUS = 'corpus'
inv_index = {}

def build_index():
    print('building index...')
    for doc_id, filename in enumerate(os.listdir(CORPUS)): 
        #print(os.listdir(CORPUS)) # ['8496.txt', '8499.txt', '8500.txt', '8530.txt', '8545.txt']
        #print(doc_id) # 0
        #print(filename) # 8496.txt
        # read document
        #print(os.path.join(CORPUS, filename)) # corpus\8496.txt
        with open(os.path.join(CORPUS, filename)) as fp:
            document = fp.read()
        # preprocessing
        tokens = []
        for token in word_tokenize(document.lower()):
            if token not in stopwords.words('english'):
                tokens.append(lemmatizer.lemmatize(token))
        inv_index[doc_id] = Counter(tokens)
    print('done!')

@app.route("/", methods=['GET'])
def index():
    # if user query in url query string, run query and return results
    query = request.args.get('q')
    if query:
        tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(query.lower()) if token not in stopwords.words('english')]

        # TODO: index
        # TODO: search
        # TODO: suggestions

    # else simply render page
    return render_template('index.html')  # return 'hello world!'

if __name__ == "__main__":
    build_index()
    app.run(host='localhost', port=5000) # debug=True
