###########################
#
# author: Daniel Laden
#         Reese Pearsall
# 
#
###########################
#angr test file

import angr
#from angrutils import * #Has errors trying to load this on Redshift? But doesn't seem necessary
import monkeyhex
import networkx as nx
from node2vec import Node2Vec
import json
import time
import numpy as np
import pyvis

import subprocess
import gc

from karateclub.graph_embedding import Graph2Vec
from karateclub.graph_embedding import GL2Vec
from karateclub.graph_embedding import IGE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from pyvis.network import Network
import os
from os import walk

start_time = time.time()



###########################
#functions

#
# Takes a file and returns a count of all instructions and vex instructions using angr
#
def binaryToCounts(filename):
	binary = angr.Project(filename, auto_load_libs=False)
	try:
		cfg = binary.analyses.CFGFast()

		#Prints all the functions in the binary as well as the address of the function in the binary
		print(binary.kb.functions.items())
		for func in binary.kb.functions.values():
			print("[+] Function {}, found at {}".format(func.name, hex(func.addr)))

		cfg.normalize()

		total_binary_instr_counts = {}
		total_binary_vex_counts = {}
		functions_out_of_memory = []
		binary_breakdown = {}


		for func_node in cfg.functions.values():
			function_counts = {}
			function_name = func_node.name

			#print(function_name)

			counts_per_instr_block = {}
			counts_per_vex_block = {}
			addr = func_node.addr

			if func_node.name.startswith("__"): #outside function we don't need to pull the code from these
				print("\033[93m\nOutside function %s detected skipping over this function.\n\033[0m" % (function_name))
				functions_out_of_memory.append(function_name)
				continue
			else:
				for block in func_node.blocks:

					#Get the instruction counts for a binary
					#block.pp() # pretty print to see what a block looks like
					for instr in block.capstone.insns:
						#print(instr.mnemonic)
						if instr.mnemonic in counts_per_instr_block.keys():
							counts_per_instr_block[instr.mnemonic] += 1
						else:
							counts_per_instr_block[instr.mnemonic] = 1

						#Add to the total counts for the binary
						if instr.mnemonic in total_binary_instr_counts.keys():
							total_binary_instr_counts[instr.mnemonic] += 1
						else:
							total_binary_instr_counts[instr.mnemonic] = 1
					#print(counts_per_instr_block) # Print to check proper counts

					#Get the vex instruction count as well
					try:
						vex_block = block.vex
						#vex_block.pp()
						for stmt in vex_block.statements:
							#print(stmt.tag)
							if stmt.tag in counts_per_vex_block.keys():
								counts_per_vex_block[stmt.tag] += 1
							else:
								counts_per_vex_block[stmt.tag] = 1

							#Add to the total counts for the binary
							if stmt.tag in total_binary_vex_counts.keys():
								total_binary_vex_counts[stmt.tag] += 1
							else:
								total_binary_vex_counts[stmt.tag] = 1
						#print(counts_per_block) # Print to check proper counts
					except Exception as e:
							print("\033[91m\nError: %s" % (e))
							print("Function %s failed to run vex skipping this function\n\033[0m" % (function_name))
							functions_out_of_memory.append(function_name)

				if counts_per_instr_block and counts_per_vex_block:
					function_counts[addr] = {"Instruction Counts" : counts_per_instr_block, "Vex Counts" : counts_per_vex_block}
					#print(function_counts)
				else:
					continue

			# A test print to make sure it's getting instruction counts
			# if function_counts and switcher:
			# 	print(function_counts)
			# else:
			# 	pass

			binary_breakdown[function_name] = function_counts

		#print(binary_breakdown)


		return ({"Total Counts" : {"Instruction Counts" : total_binary_instr_counts,"Vex Counts" : total_binary_vex_counts}, "Function Counts" : binary_breakdown, "OoM Functions" : functions_out_of_memory})
	except Exception as e:
		print("\033[91m\n\n\nError: %s" % (e))
		print("File %s failed to run skipping this file\n\n\n\033[0m" % (filename))
		time.sleep(5)


#
# Takes a list of filenames(could be changed to directory later) and creates a JSON representation of the data.
#
def filesToJSON(filename_list):
	data = {}
	for filename in filename_list:
		counts = binaryToCounts(filename)

		data[filename] = counts

	with open('Binary-file-counts.json', 'w') as outfile:
		json.dump(data, outfile, indent=2)


#
# Runs the test for a given graph embedding model for Logistic Regression(a basic method)
#
def modelTest(model, graphs, y,report):
	model.fit(graphs)

	X = model.get_embedding()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#Uses logistric regression to fit the data and make classification decisions
	try:
		downstream_model = LogisticRegression(random_state=0, max_iter=20000).fit(X_train, y_train)
		y_hat = downstream_model.predict_proba(X_test)[:, 1]
		auc = roc_auc_score(y_test, y_hat)
		print('AUC: {:.4f}'.format(auc))
		report.write("AUC: " + str(auc) + "\n")
	except Exception as e:
		print('AUC: N/A')
		report.write("AUC: Error" + str(e) + "\n")


# Finds file type for each file and writes to report
def getFileInfo(fe,counter,report,size):

	if counter > size // 2:
		mal_file = open("../malware_info2.txt","r")
		for each_line in mal_file:
			linelist = each_line.split()
			if str(linelist[0][:-1]).strip() in str(fe):
				report.write("File type: " + " ".join(linelist[1:]) + "\n")
				break
		mal_file.close()
	else:
		ben_file = open("../datasetinfo.txt","r")
		for line in ben_file:
			linelist = line.split()
			if str(linelist[0][:-1]).strip() in str(fe):
				report.write("File type: " + " ".join(linelist[1:]) + "\n")
				break

		ben_file.close()	

#
# Given a set of binary files this will output a list of cfgs for those binaries
#
def buildCFGGraphs(files, y, bc, mc,report):

	

	graphs = []
	index = 0
	size = len(files)
	counter = 0
	
	b_suc = 0 # Number of benign that succusfully generate
	b_fail = 0 # Number of benign that fail to generate generate

	m_suc = 0 # Number of malware that succusfully generate
	m_fail = 0 # Number of malware that fail to generate	 


	file_names = []  # list of file names that succusfully generated a CFG

	for fe in files:
	
		try:
			binary = angr.Project(fe,auto_load_libs=False)
			
			temp = open("temp.txt","a") # historical list of all files that succufully generated CFGs
			name = fe 

			counter +=1
			print(str(counter) + "/" + str(size))
			
			cfg = binary.analyses.CFGFast()  # CFGFast = Generates CFG using static methods
			cfg = cfg.graph
			cfg = cfg.to_undirected()
			
			file_names.append(name)
			temp.write(name + "\n")
			temp.close()
						
			graphs.append(cfg) # Graph2Vec needs a list of CFG graphs
			index += 1
			
			report.write(str(counter) + " Success. File Name: " + str(fe) + "\n") # Keep track of how many passed and failed
			if counter >= bc:
				m_suc += 1
			else: 
				b_suc += 1
			getFileInfo(fe,counter,report,size)
		

		# This exception shows so the file that errors out and has some time for a user to see the popup before continuing
		# Change this if you wanna collect what gets removed from the dataset or etc
		except Exception as e:
			print("\033[91m\n\n\nError: %s" % (e))
			print("File %s failed to run skipping this file\n\n\n\033[0m" % (fe))
			del y[index]
			#del files[index]
			report.write(str(counter) + " Failure. File Name: " + str(fe) + "\n")
			counter += 1
			report.write("Reason: " + str(e) + "\n") # Writes out the error message to report
			if counter >= bc:
				m_fail += 1
			else: 
				b_fail += 1
			getFileInfo(fe,counter,report,size)
			
			time.sleep(30)
		
		report.write(" " + "\n")
	
	report.write("Malware Success: " + str(m_suc) + "/" + str(mc) + "\n")
	report.write("Benign Success: " + str(b_suc) + "/" + str(bc) + "\n")

	return [graphs, y, file_names]


#
# Given a set of binary files this will output a list of ddgs for those binaries
#
def buildDDGGraphs(files):
	graphs = []
	for file in files:
		binary = angr.Project(file, auto_load_libs=False)
		cfg = binary.analyses.CFGEmulated(keep_state=True, state_add_options=angr.sim_options.refs, context_sensitivity_level=2)

		#Generate the control dependence graph
		print("\n\n\nCDG being built...\n\n\n")
		time.sleep(10)
		cdg = binary.analyses.CDG(cfg)

		#Build the data dependence graph. Might take time
		print("\n\n\nDDG being built...\n\n\n")
		time.sleep(10)
		ddg = binary.analyses.DDG(cfg)

		graphs.append(ddg)

	return graphs






# Writes out file name, type, size, # nodes, # edges to a report each time an experiment is run
# The file should be generated in the same directory as this Python file
def write_out_graph_info(graph_file, file_name, most_connected_graph):

	# Determines what kind of file type (executable, pdf, html, etc)
	command = ["file",str(file_name)]
	type_of_file = subprocess.Popen(command, stdout=subprocess.PIPE)
	output, error = type_of_file.communicate()

	size = len(file_name)
	line = ""
	line += str(file_name) + ","
	
	if "/mnt/" in file_name:  # then it is malware
		file_class = "m"
	elif "/py-angr-data/" in file_name: # then it is benign
		file_class = "b"
	else:
		file_class = "???"
		
	line += file_class + ","
	line += (str(output.decode("utf-8"))[size:]).strip().replace(",","") + ","
	line += str(os.path.getsize(file_name)) + ","
	line += str(len(most_connected_graph.edges())) + ","
	line += str(len(most_connected_graph.nodes())) + "\n"
	
	graph_file.write(line)

#
# We need the largest connected component because Graph2Vec doesn't work with disconnected graphs(?)
# This function will cycle through and find the graph with the most edges within it.
#
def findMostConnectedGraph(graphs, files,filenames):
	
	#switcher = True
	max_graphs = []
	
	graph_file = open("graph_temp.csv","w")
	graph_file.write("file,class,file type,size(bytes),edges,nodes" +"\n")
	
	for graph, file, name in zip(graphs, files,filenames):
		most_connected = 0
		most_connected_graph = None
		components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
		for idx, g in enumerate(components,start=1):
			if len(g.edges()) > most_connected:
				#print("Old value: %d\tNew value: %d" % (most_connected, len(g.edges())))
				most_connected = len(g.edges())
				most_connected_graph = g
				#time.sleep(0.5)
			
		if most_connected_graph:
			max_graphs.append(most_connected_graph)
			
			write_out_graph_info(graph_file, name, most_connected_graph)
		
		else: #Raise an error for that file
			print("File %s failed to run quiting program\n\n\n\033[0m" % (file))
			#quit()
	graph_file.close()
	return max_graphs


#
# Change the node mappings due to CFGNode comparison errors switches them to the function name
#
def swapNodeMapping(graphs):
	new_graphs = []
	for graph in graphs:
		G = nx.Graph()
		nodes = []
		edges = []
		for node in graph.nodes():
			if node.name:
				nodes.append(node.name) #using the function's address for the remapping use node.addr
			else:
				nodes.append(str(node.addr))
		for edge in graph.edges():
			if edge[0].name and edge[1].name: # might cause errors with resolving things to .addr when it doesn't need

				e = (edge[0].name, edge[1].name)
				edges.append(e)
			elif edge[0].name:
				e = (edge[0].name, edge[1].addr)
			elif edge[1].name:
				e = (edge[0].addr, edge[1].name)
			else:
				e = (str(edge[0].addr), str(edge[1].addr))
				edges.append(e)
			
		G.add_nodes_from(nodes)
		G.add_edges_from(edges)
		new_graphs.append(G)

	return new_graphs


#
# Creates html visualizations for all given graphs
#
def createDDGVisualization(graphs, files):
	for file, ddg in zip(files, graphs):
		ddg_graph = fixDDGNodes(ddg.graph)

		net = Network(notebook=True)
		net.from_nx(ddg_graph)
		filename = file + "-ddg.html"
		net.show(filename)


#
# gets around the annoying code object so we can do visualization
#
def fixDDGNodes(ddg):
	G = nx.Graph()
	nodes = []
	edges = []
	for node in ddg.nodes():
		if not node.block_addr: #ignore nonetype returned addresses OoM addresses?
			continue
		nodes.append(node.block_addr)
	for edge in ddg.edges():
		e = (edge[0].block_addr, edge[1].block_addr)
		if e[1] == e[0]: #skip self representation
			continueFailure
		elif not e[1] or not e[0]: #ignore nonetype returned addresses OoM addresses?
			continue
		edges.append(e)

	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	return G


#
# gets around the annoying code object so we can do visualization
#
def fixVSANodes(ddg):
	G = nx.Graph()
	nodes = []
	edges = []
	for node in ddg.nodes():
		#print(dir(node))
		if not node.addr: #ignore nonetype returned addresses OoM addresses?
			continue
		nodes.append(node.addr)
	for edge in ddg.edges():
		e = (edge[0].addr, edge[1].addr)
		if e[1] == e[0]: #skip self representation
			continue
		elif not e[1] or not e[0]: #ignore nonetype returned addresses OoM addresses?
			continue
		edges.append(e)

	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	return G


def graph_analysis(malware_directory="8a0"):

	print(binary.arch)


	print("\n===================Loaded Objects===================\n")
	for obj in binary.loader.all_objects:
		print(obj)
	print("\n===================Loaded Objects===================\n")

	print("Program has an executable stack? %s\n" % (binary.loader.main_object.execstack))



	#
	# Deeper loader information
	#

	#All loaded in elf objects
	all_elf = binary.loader.all_elf_objects 

	#All external objects that help with unresolved imports
	external = binary.loader.extern_object

	#Object used to provide addresses for emulated syscalls
	kernel = binary.loader.kernel_object



	#
	# Object Metadata
	#
	main_binary = binary.loader.main_object
	print("\n===================Included Sections===================\n")
	for obj in main_binary.sections:
		print(obj)
	print("\n===================Included Sections===================\n")

	
	#
	# Basic Block extraction
	#

	block = binary.factory.block(main_binary.entry)
	#print(block)

	counter = 0

	# Report name. Uses current time as file name
	report_name = "report_files_" + str(start_time)
	report = open("/home/reese/Documents/research/Documents/Research/py-angr/file_reports/" + report_name,"w")



	# Benign data
	mypath = "/home/daniel/Documents/Research/datasets/py-angr-data/"
	f_benign = []
	y_benign = []

	bc =0
	mc = 0

	# Go through directory of malware + benign directories
	for (dirpath, dirnames, filenames) in walk(mypath):
		print(dirpath)
		for f in filenames:
		
			# Determine file type
			command = ["file",str(dirpath + f)]
			type_of_file = subprocess.Popen(command, stdout=subprocess.PIPE)
			output, error = type_of_file.communicate()


			# Filter out everything that is not an executable
			if "executable" in output.decode("utf-8") or "ELF" in output.decode("utf-8"):
		
				file_b = dirpath + f
				f_benign.append(file_b)
				counter += 1	 
				y_benign.append(0)

				if counter == 3:
					bc = counter
					break
		break

	counter = 0

	# Malware Data
	mypath = "/mnt/sda/vol1/" + str(malware_directory) + "/"
	f_malware = []
	y_malware = []
	for (dirpath, dirnames, filenames) in walk(mypath):
		print(dirpath)
		for f in filenames:
		
			command = ["file",str(dirpath + f)]
		
			type_of_file = subprocess.Popen(command, stdout=subprocess.PIPE)
			output, error = type_of_file.communicate()

			
			if "executable" in output.decode("utf-8") or "ELF" in output.decode("utf-8"):
		
				file_m = dirpath + f
				f_malware.append(file_m)
				counter += 1
				y_malware.append(1)

				if counter == 285:
					mc = counter
					break
		break


	files = f_benign + f_malware
	y = y_benign + y_malware

	#
	# Graph Analysis
	#
	print(len(y),len(files))
	graphs = buildCFGGraphs(files, y, bc, mc,report)

	y = graphs[1]
	file_names = graphs[2]
	graphs = graphs[0]
	
	
	## write out names of good files
	good = open("/home/reese/Documents/research/Documents/Research/py-angr/good_files/"+ str(malware_directory) + ".txt","w")
	for each in file_names:
		good.write(each + "\n")
	good.close()
	
	

	print(len(y),len(graphs))
	opt_graphs = findMostConnectedGraph(graphs, files,file_names)
	remapped_graphs = swapNodeMapping(opt_graphs)


	model = Graph2Vec()
	model.fit(remapped_graphs)
	X = model.get_embedding()



	#Tests three different g2v like models
	model = Graph2Vec()
	#model2 = GL2Vec()
	#model3 = IGE()
	modelTest(model,remapped_graphs,y,report)
	#modelTest(model2,remapped_graphs,y)
	#modelTest(model3,remapped_graphs,y)
	print("\n\n\n")
	print(len(y))
	report.close()
	time.sleep(30)


	del graphs
	del file_names
	del y
	gc.collect()


	print("--- Runtime of program is %s seconds ---" % (time.time() - start_time))


## Kicks off the program
if __name__ == "__main__":
	graph_analysis()


#########################
#
# https://docs.angr.io/core-concepts/toplevel
# https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
# https://stackoverflow.com/a/2259250
# https://reverseengineering.stackexchange.com/a/24666
# http://angr.io/api-doc/angr.html#angr.analyses.disassembly.Disassembly
# https://stackoverflow.com/questions/40243753/exception-dot-not-found-in-path-in-python-on-mac
#
# https://breaking-bits.gitbook.io/breaking-bits/vulnerability-discovery/automated-exploit-development/analyzing-functions
# https://docs.angr.io/advanced-topics/ir
#
# https://github.com/eliorc/node2vec/issues/5
# https://karateclub.readthedocs.io/en/latest/notes/introduction.html
#
# https://stackoverflow.com/questions/6886493/get-all-object-attributes-in-python
# https://stackabuse.com/reading-and-writing-json-to-a-file-in-python
# https://www.delftstack.com/howto/python/python-print-colored-text/
#
# https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.to_undirected.html
# https://stackoverflow.com/questions/48820586/removing-isolated-vertices-in-networkx
# https://networkx.org/documentation/stable/reference/generated/networkx.relabel.relabel_nodes.html
# https://stackoverflow.com/questions/21739569/finding-separate-graphs-within-a-graph-object-in-networkx
#
#########################
