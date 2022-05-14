from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import math
# print(os.listdir("."))
from collections import Counter # A counter is a container that stores elements as dictionary keys, and their counts are stored as dictionary values.
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

def build_index():
    print('building index...')
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
            # get max freq of w in any doc (or using sum to get the total number of freq)
            # print(freq_wd[d].values()) #dict_values([1, 2])
            max_d = max(freq_wd[d].values())
            print('freq_wd[',d,',', w,']', freq_wd[d].get(w,0), 'max:', max_d)
            TF[w,d] = freq_wd[d].get(w,0) / max_d
            print('TF[w,d]=',TF[w,d])

    # IDF(w) = log(total documents / num documents where w appears)
    print('Computing IDF(w) = log(total documents / num documents where w appears)')
    for w in all_tokens:
        nw = len([d for d in DOCUMENTS if w in freq_wd[d]])
        print('N=',N,'nw=',nw)
        IDF[w] = math.log2(N / (nw + 1))
        print('IDF[',w,']=',IDF[w])

@app.route("/", methods=['GET'])
def index():
    # if user query in url query string, run query and return results
    query = request.args.get('q')
    if query:
        q_tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(query.lower()) if token not in STOPWORDS]
        relevance_scores = {}
        for d in DOCUMENTS:
            score = 0
            for w in q_tokens:
                score += TF[w,d] * IDF[w]
            relevance_scores[d] = score
        rankings = list(sorted(relevance_scores, key=relevance_scores.get, reverse=True)) # .get() value
        

    # else simply render page
    return render_template('index.html')  # return 'hello world!'

if __name__ == "__main__":
    build_index()
    app.run(host='localhost', port=5000) # debug=True
