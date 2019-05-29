import rdflib
import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from gensim.models import Word2Vec
from pprint import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

G=nx.Graph()

rdf = True

if rdf:
    g = rdflib.Graph()
    result = g.parse("xlendiamphoraesample2.rdf")
    for s, p, o in g:
        G.add_edge(s, o)
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
node2vec = Node2Vec(G, dimensions=20, walk_length=16, num_walks=100)

# Learn embeddings
model = node2vec.fit(window=10, min_count=1)


X = []
y = []
for s,p,o in g:
    X.append([model.wv[str(s)], model.wv[str(o)]])
    y.append(True)
    # print(X[-1])

print(model.wv.syn0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
pred = clf.predict(X_test)

print(accuracy_score(pred, y_test))


# plot_embeddings(embeddings)