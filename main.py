import rdflib
import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from gensim.models import Word2Vec
from pprint import pprint
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import random

G=nx.Graph()

rdf = True

oX = []
oY = []

if rdf:
    g = rdflib.Graph()
    result = g.parse("xlendiamphoraesample2.rdf")
    for s, p, o in g:
        G.add_edge(s, o)
        oX.append([s, o])
        oY.append(p)
    # for edge in G.edges:
    #     print(edge)

    # not all the edges from the rdf are in the graph, need to see why

else: # other graph, pretty long to train
    dfedges = pd.read_csv("edges.csv")
    dfnodes = pd.read_csv("nodes.csv")
    dfgroups = pd.read_csv("groups.csv")
    grEdges = pd.read_csv("group-edges.csv")
    for i in range(len(dfedges)):
        G.add_edge(dfedges.iloc[i, 0], dfedges.iloc[i, 1])



G = G.to_undirected()

# Generate walks
node2vec = Node2Vec(G, dimensions=300, walk_length=16, num_walks=100)

# Learn embeddings
model = node2vec.fit(window=10, min_count=1)


X = []
y = []
for i in range(len(oX)):
    X.append((model.wv[str(oX[i][0])] + model.wv[str(oX[i][1])])/2)
    print(oY[i])
    # if str(oY[i]) == 'https://data.hawaii.gov/resource/usep-nua7/fund':
    y.append(True)
    # else:
    #     y.append(False)

# put a random edge that will be false
for i in range(len(oX)):
    r1 = random.randint(0,len(oX)-1)
    r2 = random.randint(0,len(oX)-1)
    biR1 = random.randint(0,1)
    biR2 = random.randint(0,1)
    if [oX[r1][biR1], oX[r2][biR2]] not in oX and [r2, r1] not in oX:
        X.append((model.wv[str(oX[r1][biR1])] + model.wv[str(oX[r2][biR2])]) / 2)
        y.append(False)

# the edge is represented by the sum of the twe vertices vectors


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# I am using a linear svm
clf = svm.SVC(kernel='linear').fit(X_train, y_train)
pred = clf.predict(X_test)

print(accuracy_score(pred, y_test))
print(f1_score(pred, y_test))
print(recall_score(pred, y_test))
print(precision_score(pred, y_test))
# high precision, low recall
# very few false positives, but more false negative


# plot_embeddings(embeddings)