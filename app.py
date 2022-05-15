from flask import Flask, render_template, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import math
from time import time
# print(os.listdir("."))
from collections import Counter # A counter is a container that stores elements as dictionary keys, and their counts are stored as dictionary values.
import json
from query_suggestions import QuerySuggestions
app = Flask(__name__, static_url_path='/static') 

lemmatizer = WordNetLemmatizer()
CORPUS = 'corpus/wiki_dir'
#print(os.listdir(CORPUS)) # ['8496.txt', '8499.txt', '8500.txt', '8530.txt', '8545.txt']
DOCUMENTS = os.listdir(CORPUS)
STOPWORDS = set(stopwords.words('english'))        
N = len(DOCUMENTS)
#inv_index = {}
TF = {} #Term Frequency
IDF = {} #Inverse Document Frequency
QUERY_LOGFILE = 'data/query_count.csv'
QUERY_SUGGESTIONS = QuerySuggestions(QUERY_LOGFILE)
#print(QUERY_SUGGESTIONS.query_log)
#print(QUERY_SUGGESTIONS.query_log.items()) # dict_items([('a a', 6), ('a a a', 3),...])
#print(QUERY_SUGGESTIONS.get_candidates('one').values()) # dict_values([3, 1])
print(QUERY_SUGGESTIONS.get_candidates('one').items()) 
QUERY_SUGGESTIONS.rank_candidates(QUERY_SUGGESTIONS.get_candidates('one'))
#score_dict {'one book': 0.3, 'twenty one': 0.1, 'one thousands and one hundred': 0.2, 'one thousands and two hundreds': 0.1, 'one thousands and three hundreds': 0.3}

def build_index():
    global TF, IDF
    # load stored index data if possible
    if os.path.exists('index/TF.json') and os.path.exists('index/IDF.json'):
        with open('index/TF.json') as fp:
            TF = json.load(fp)
        with open('index/IDF.json') as fp:
            IDF = json.load(fp)
        return 
    
    # compute index fresh        
    print('building index...')
    start_time = time()
    all_tokens = set()

    # freq(w,d) = frequency of terms in each document
    freq_wd = {}
    print('computing freq(w,d)')
    for doc_id, filename in enumerate(DOCUMENTS):
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
            if token not in STOPWORDS and token.isalpha():
                tokens.append(lemmatizer.lemmatize(token))
                all_tokens.add(lemmatizer.lemmatize(token))
        print(filename, tokens)
        #inv_index[doc_id] = Counter(tokens)
        freq_wd[filename] = Counter(tokens)
    print('all tokens', all_tokens)
    print(freq_wd) #{0:Counter({'oil':3, 'sentence':1}), 1:Counter({'human': 2, 'bloom':1})}

    # term-frequency = freq(w,d) / max_d
    print('Computing term-frequency = freq(w,d) / max_d')
    for d in DOCUMENTS: 
        for w in all_tokens:
            key = w + '-' + d
            # get max freq of w in any doc (or using sum to get the total number of freq)
            # print(freq_wd[d].values()) #dict_values([1, 2])
            max_d = max(freq_wd[d].values())
            print('freq_wd[',key,']', freq_wd[d].get(w,0), 'max:', max_d)
            TF[key] = freq_wd[d].get(w,0) / max_d
            print('TF[',key,']=',TF[key])

    # IDF(w) = log(total documents / num documents where w appears)
    print('Computing IDF(w) = log(total documents / num documents where w appears)')
    for w in all_tokens:
        nw = len([d for d in DOCUMENTS if w in freq_wd[d]])
        print('N=',N,'nw=',nw)
        IDF[w] = math.log2(N /(nw+0.5))
        print('IDF[',w,']=',IDF[w])
    
    duration = round(time() - start_time, 3)
    print(f'done! built index in {duration} seconds')

    # persist index
    INDEX_DIR = 'index'
    TF_FILE = os.path.join(INDEX_DIR, 'TF.json')
    IDF_FILE = os.path.join(INDEX_DIR, 'IDF.json')
    FREQ_WD_FILE= os.path.join(INDEX_DIR, 'FREQ_WD.json')
    # store values to avoid re-computing
    with open(FREQ_WD_FILE, 'w') as fp:
        json.dump(freq_wd, fp)
    with open(TF_FILE, 'w') as fp:
        json.dump(TF, fp)
    with open(IDF_FILE, 'w') as fp:
        json.dump(IDF, fp)

@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')  # return 'hello world!'

@app.route("/search", methods=['GET'])
def search():
    # if user query in url query string, run query and return results
    rankings = []
    if request.args:
        query = request.args.get('q')

        # perform search, get rankings, return most relevant result
        q_tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(query.lower()) if token not in STOPWORDS]
        relevance_scores = {}
        for d in DOCUMENTS:
            score = 0
            for w in q_tokens:
                key = w + '-' + d
                score += TF.get(key) * IDF.get(w)
                print(TF.get(key), IDF.get(w), TF.get(key)*IDF.get(w), score)
            relevance_scores[d] = score

        print(relevance_scores)
        rankings = list(sorted(relevance_scores, key=relevance_scores.get, reverse=True)) # .get() value

    return  render_template('index.html', results=rankings)

if __name__ == "__main__":
    build_index()
    app.run(host='localhost', port=5000) # debug=True
