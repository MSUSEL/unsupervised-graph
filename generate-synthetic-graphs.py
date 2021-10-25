# This code generates synthetic graphs, with varying numbers of nodes and edges,
#	and more importantly, differing topological properties

import networkx as nx
import random

graph_list = [] # karate club package graph embedding algorithms assume a graph list as input

# generate Erdos-Renyi graphs
for i in range(100):
    n = random.randint(50,500) # select a random number of nodes from a pre-defined range
    m = random.randint(1000, 50000) # select a random number edges from a pre-defined range
    graph = nx.gnm_random_graph(n,m)
    graph_list.append(graph)

# save Erdos-Renyi graphs (for reproducibility)
counter=0
for g in graph_list[:100]:
    counter += 1
    fname = './data/synthetic-graphs/erdos-renyi-graphs/erdos-renyi-'+str(counter)
    nx.write_gpickle(g, fname+".gpickle")

print('generated and saved '+ str(len(graph_list)) + ' Erdos-Renyi graphs')

# generate complete graphs
for i in range(100):
    n = random.randint(50,1000) # select a random number of nodes from a pre-defined range
    graph = nx.complete_graph(n)
    graph_list.append(graph)

# save complete graphs (for reproducibility)
counter=0
for g in graph_list[100:200]:
    counter += 1
    fname = './data/synthetic-graphs/complete-graphs/complete-'+str(counter)
    nx.write_gpickle(g, fname+".gpickle")

print('generated and saved '+ str(counter) + ' complete graphs')

# generate power law graphs (Barabasi-Albert)
for i in range(100):
    n = random.randint(50,500) # select a random number of nodes from a pre-defined range
    m = random.randint(1, 6) # select a random number edges to attach at each step from a pre-defined range
    graph = nx.barabasi_albert_graph(n,m)
    graph_list.append(graph)

# save Barabasi-Albert graphs (for reproducibility)
counter=0
for g in graph_list[200:300]:
    counter += 1
    fname = './data/synthetic-graphs/barabasi-albert-graphs/barabasi-albert-'+str(counter)
    nx.write_gpickle(g, fname+".gpickle")

print('generated and saved '+ str(counter) + ' Barabasi-Albert graphs')

# generate Newman-Watts-Strogatz graphs 
for i in range(100):
    n = random.randint(50,500) # select a random number of nodes from a pre-defined range
    k = random.randint(1, 6) # select a random number edges to attach in a ring topology at each step from a pre-defined range
    p = random.uniform(0,1) # select a random probability of adding a new edge for each edge, between 0 and 1
    graph = nx.newman_watts_strogatz_graph(n,k,p)
    graph_list.append(graph)

# save Newman-Watts-Strogatz graphs (for reproducibility)
counter=0
for g in graph_list[300:400]:
    counter += 1
    fname = './data/synthetic-graphs/newman-watts-strogatz-graphs/newman-watts-strogatz-'+str(counter)
    nx.write_gpickle(g, fname+".gpickle")

print('generated and saved '+ str(counter) + ' Newman-Watts-Strogatz graphs')

# generate cycle graphs 
for i in range(100):
    n = random.randint(50,500) # select a random number of nodes from a pre-defined range
    graph = nx.cycle_graph(n)
    graph_list.append(graph)

# save cycle graphs (for reproducibility)
counter=0
for g in graph_list[400:500]:
    counter += 1
    fname = './data/synthetic-graphs/cycle-graphs/cycle-'+str(counter)
    nx.write_gpickle(g, fname+".gpickle")

print('generated and saved '+ str(counter) + ' cycle graphs')

# generate path graphs 
for i in range(100):
    n = random.randint(50,500) # select a random number of nodes from a pre-defined range
    graph = nx.path_graph(n)
    graph_list.append(graph)

# save path graphs (for reproducibility)
counter=0
for g in graph_list[500:600]:
    counter += 1
    fname = './data/synthetic-graphs/path-graphs/path-'+str(counter)
    nx.write_gpickle(g, fname+".gpickle")

print('generated and saved '+ str(counter) + ' path graphs')
