import networkx as nx

# load saved Erdos-Renyi graphs in the same order they were generated

graph_list = []

for i in range(1, 101):
    fname = './data/synthetic-graphs/erdos-renyi-graphs/erdos-renyi-'+str(i)+'.gpickle'
    G = nx.read_gpickle(fname)
    graph_list.append(G)
print('finished loading '+str(i)+' erdos-renyi graphs')

# load saved complete graphs in the same order they were generated above:
for i in range(1,101):
    fname = './data/synthetic-graphs/complete-graphs/complete-'+str(i)+'.gpickle'
    G = nx.read_gpickle(fname)
    graph_list.append(G)
print('finished loading '+str(i)+' complete graphs')

# load saved power-law (Barabasi-Albert) graphs in the same order they were generated above:
for i in range(1,101):
    fname = './data/synthetic-graphs/barabasi-albert-graphs/barabasi-albert-'+str(i)+'.gpickle'
    G = nx.read_gpickle(fname)
    graph_list.append(G)
print('finished loading '+str(i)+' barabasi-albert graphs')

# load saved Newman-Watts-Strogatz graphs in the same order they were generated
for i in range(1,101):
    fname = './data/synthetic-graphs/newman-watts-strogatz-graphs/newman-watts-strogatz-'+str(i)+'.gpickle'
    G = nx.read_gpickle(fname)
    graph_list.append(G)
print('finished loading '+str(i)+' newman-watts-strogatz graphs')

# load saved cycle graphs in the same order they were generated
for i in range(1,101):
    fname = './data/synthetic-graphs/cycle-graphs/cycle-'+str(i)+'.gpickle'
    G = nx.read_gpickle(fname)
    graph_list.append(G)
print('finished loading '+str(i)+' cycle graphs')

# load saved path graphs in the same order they were generated
for i in range(1,101):
    fname = './data/synthetic-graphs/path-graphs/path-'+str(i)+'.gpickle'
    G = nx.read_gpickle(fname)
    graph_list.append(G)
print('finished loading '+str(i) + ' path graphs')

# Use Graph2Vec to generate graph embeddings, 
# 	with differing numbers of dimensionality

import matplotlib.pyplot as plt
print('generating Graph2Vec embeddings and visualizations...')
from karateclub import Graph2Vec
ndims_list = [2,4,8,16,32,64,128,256]
for ndims in ndims_list:
	print('ndims: '+str(ndims))
	model = Graph2Vec(dimensions=ndims)
	model.fit(graph_list)
	embedding = model.get_embedding()

	# get TSNE embedding for visualization of output
	from sklearn.manifold import TSNE
	twoD_embedded_graphs = TSNE(n_components=2).fit_transform(embedding)

	# Plot visualization of Graph2Vec results
	plt.plot(twoD_embedded_graphs[:100,0], twoD_embedded_graphs[:100,1], 'b*', label='Erdos-Renyi')
	plt.plot(twoD_embedded_graphs[100:200,0], twoD_embedded_graphs[100:200,1], 'r*', label='complete')
	plt.plot(twoD_embedded_graphs[200:300,0], twoD_embedded_graphs[200:300, 1], 'g*', label="Barabasi-Albert")
	plt.plot(twoD_embedded_graphs[300:400, 0], twoD_embedded_graphs[300:400,1], 'k*', label='Neumann-Watts-Strogatz')
	plt.plot(twoD_embedded_graphs[400:500, 0], twoD_embedded_graphs[400:500,1], 'c*', label='cycle')
	plt.plot(twoD_embedded_graphs[500:, 0], twoD_embedded_graphs[500:,1], 'm*', label='path')
	plt.legend(loc='upper left')
	#plt.legend(bbox_to_anchor=(1.05, 1))
	plt.title('Graph2Vec ('+str(ndims)+' dims) \n TSNE visualization of input graphs')
	fname='./figs/graph-embedding-visualizations/synthetic-graph-comparison-graph2vec-'+str(ndims)+'-dims.png'
	plt.savefig(fname)
	plt.clf()

# Produce embeddings and visualizations with LDP embeddings
print('generating LDP embeddings and visualizations...')
from karateclub import LDP
nbins_list = [4, 16, 32, 66, 128, 256]
for nbins in nbins_list:
	ldp_model = LDP(bins=nbins)
	ldp_model.fit(graph_list)
	embedding = ldp_model.get_embedding()
	
	# produce visualization using LDP embeddings
	twoD_embedded_graphs = TSNE(n_components=2).fit_transform(embedding)
	plt.plot(twoD_embedded_graphs[:100,0], twoD_embedded_graphs[:100,1], 'b*', label='Erdos-Renyi')
	plt.plot(twoD_embedded_graphs[100:200,0], twoD_embedded_graphs[100:200,1], 'r*', label='complete')
	plt.plot(twoD_embedded_graphs[200:300,0], twoD_embedded_graphs[200:300, 1], 'g*', label="Barabasi-Albert")
	plt.plot(twoD_embedded_graphs[300:400, 0], twoD_embedded_graphs[300:400,1], 'k*', label='Neumann-Watts-Strogatz')
	plt.plot(twoD_embedded_graphs[400:500, 0], twoD_embedded_graphs[400:500,1], 'c*', label='cycle')
	plt.plot(twoD_embedded_graphs[500:, 0], twoD_embedded_graphs[500:,1], 'm*', label='path')
	plt.legend(loc='lower left')
	plt.title('LDP embedding ('+str(nbins)+' bins) \n TSNE visualization of input graphs')
	fname='./figs/graph-embedding-visualizations/synthetic-graphs-LDP-'+str(nbins)+'-bins.png'
	plt.savefig(fname)
	plt.clf()


