# SOLUTION 1
import time
start_time = time.time()

import numpy as np
def question1(s, t):
    (result, indices) = (True, [])
    
    for element in list(t):
        boolean = element in (list(s))
        if boolean == False: 
            (result, indices) = (False,[])
        else:
            indices.append(list(s).index(element))
            result = np.logical_and(result, boolean)
    
    if (result == True) & (sum(map(abs, np.diff(indices))) == len(indices) - 1):
        return True
    else: return False
    
# TEST 1
s = "udacity"
t = ["cit", "ad", "mod"]
for el in t:
    print (el + " in " + s + " : " + str(question1(s, el)))
    
# SOLUTION 2
# with helper functions
def question2(a):
    result = search_palindrome_in(a)
    if (len(list(a)) == 0):        
        return '' 
    elif (len(list(a)) == 1) | (result == []):
        if 'a' in list(a):
            return 'a'
        else:
            return None
    else:
        max_length = max(len(s) for s in result)
        longest_result = [s for s in result if len(s) == max_length]
        return longest_result[0]
    
def is_palindrome(string):
    if string == "":
        return True
    else:
        if string[0] == string[-1]:
            return is_palindrome(string[1:-1])
        else:
            return False
        
def substrings(string):
    for n in range(2, len(string)):
        for start in range(0, len(string)-n+1):
            yield string[start:start+n]

def search_palindrome_in(string):
    palindromes = []
    parts = substrings(string)
    for element in parts:
        if is_palindrome(element):
            palindromes.append(element)
    if is_palindrome(string):
        palindromes.append(string)
    return palindromes

# TEST 2
test_cases = ['', 'test', 'mart', 'letter', 'parallelogram', 'radarrotator']
for test in test_cases:   
    print (list(test))
    print (list(substrings(test)))
    print (search_palindrome_in(test))
    print (question2(test))
    
# SOLUTION 3
import networkx as nx
from collections import OrderedDict

def question3(G):
    GW = create_wgraph(G)
    MST = nx.minimum_spanning_tree(GW)
    D = create_wdictionary(MST)
    OD = OrderedDict(sorted(D.items(), key=lambda t: t[0]))
    return OD

def create_wgraph(G):
    W = nx.Graph()
    for n, edges in G.items():
        for (u,v) in edges:
            W.add_edge(n, u, weight=v)
    return W

def create_wdictionary(W):
    D ={}
    Ws = nx.get_edge_attributes(W,'weight')
    for ((u,v), n) in Ws.items():
        D[u] = [(v, n)]
    return D

# TEST 3
GE = {'A': [('B', 2)],
      'B': [('A', 2), ('C', 5)], 
      'C': [('B', 5)]}
print ("TEST 3.0, GRAPH G", question3(GE))
G = {'A': [('B', 2), ('D', 6), ('G', 4), ('J', 3)],
     'B': [('A', 2), ('C', 5)], 
     'C': [('B', 5), ('E', 5)],
     'D': [('A', 6), ('E', 8)],
     'E': [('C', 5), ('D', 8), ('F', 3)],
     'F': [('E', 3), ('G', 7)],
     'G': [('A', 4), ('F', 7), ('H', 8), ('I', 3)],
     'H': [('G', 8)], 
     'I': [('G', 3), ('J', 7)],
     'J': [('A', 3), ('I', 7)]}
print ("TEST 3.0, GRAPH G", question3(G))
G1 = {'A': [('B', 2), ('D', 6), ('G', 4), ('J', 3)],
      'B': [('A', 2), ('C', 5), ('L', 4)], 
      'C': [('B', 5), ('E', 5), ('K', 8)],
      'D': [('A', 6), ('E', 8)],
      'E': [('C', 5), ('D', 8), ('F', 3)],
      'F': [('E', 3), ('G', 7)],
      'G': [('A', 4), ('F', 7), ('H', 8), ('I', 3)],
      'H': [('G', 8), ('K', 5), ('O', 2)], 
      'I': [('G', 3), ('J', 7), ('N', 4)],
      'J': [('A', 3), ('I', 7)],
      'K': [('C', 8), ('H', 5), ('L', 3)],  
      'L': [('B', 4), ('K', 3), ('M', 2)],
      'M': [('L', 2), ('N', 7)],
      'N': [('I', 4), ('M', 7), ('O', 5)],
      'O': [('N', 5), ('H', 2)]}
print ("TEST 3.1, GRAPH G1", question3(G1))

# SOLUTION 4
import scipy as sp
import pandas as pd
def question4(T, node1, node2, root):
    DF = create_df(T)
    BTG = create_graph(DF)

    if BTG != None:   
        nodes = nx.dijkstra_path(BTG, node1, node2)
    
        shortest_paths_nodes = []
        for element in nodes:
            shortest_paths_nodes.append(nx.shortest_path(BTG, root, element))
        
        index = []
        for path in shortest_paths_nodes:
            index.append(len(path))
        root_path_nodes = shortest_paths_nodes[min(index)]
    
        result = list(set(nodes) & set(root_path_nodes))
        return result[0]
    
    else: 
        print ("This graph is not a binary search tree")
        return None        
    
def create_df(T):
    BT = np.matrix(T)
    sparse = sp.sparse.coo_matrix(BT, dtype=np.int32)
    nodes = range(BT.shape[0])
    DF = pd.DataFrame(sparse.toarray(), index=nodes, columns=nodes)
    return DF
    
def create_graph(DF):
    BTG = nx.Graph()
    BTG.add_nodes_from(list(DF.index))
    for i in range(DF.shape[0]):
        column_label = DF.columns[i]
        for j in range(DF.shape[1]):
            row_label = DF.index[j]
            node = DF.iloc[i,j]
            if node == 1:
                BTG.add_edge(column_label,row_label)
    if nx.is_tree(BTG):
        return BTG
    else:
        return None

# TEST 4
T0 = [[0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]]
print("T0", question4(T0, 2, 4, 0))
T = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
print('The least common ancestor between 12 and 14 for T (root 0)', question4(T, 12, 14, 0))
T1 = [[0, 1, 1, 0, 0, 0, 0],
      [1, 0, 0, 1, 1, 0, 0],
      [1, 0, 0, 0, 0, 1, 1],
      [0, 1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0]]     
print('The least common ancestor between 2 and 4 for T1 (root 0)', question4(T1, 2, 4, 0))

# SOLUTION 5
def question5(l, m):
    ll = create_linked_list(l)
    return ll.getItemLeft(m)

def create_linked_list(L):
    linked_list = LinkedList()
    for el in L:
        linked_list.add(el)
    return linked_list

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        
    def __repr__(self):
        return str(self.data)

class LinkedList:
    def __init__(self):
        self.head = None
        
    def add(self, data):
        new_node = Node(data)
        if self.head == None:
            self.head = new_node
        else:
            current_node = self.head
            while current_node.next:
                current_node = current_node.next
            current_node.next = new_node
        
    def getLength(self):
        current = self.head
        count = 1
        while current.next != None:
            count += 1
            current = current.next
        return count
    
    def getIndex(self, data):
        current_node = self.head        
        index = 1
        while current_node.next:
            if current_node.data == data:
                break
            else:     
                current_node = current_node.next
                index += 1
        if (index == self.getLength()) & (current_node.data != data):
            print ("This item is not in the list")
            return None
        return index
    
    def getIndexLeft(self, data):
        if self.getIndex(data) == None:
            return None
        else: 
            pos = int(self.getIndex(data))
            end = int(self.getLength())
            return end - pos + 1
    
    def getItem(self, position):
        if position > self.getLength():
            print ("This index is out of the list range")
            return None
        else:
            current_node = self.head
            while (current_node.next != None):
                if self.getIndex(current_node.data) == position:
                    break
                else:
                    current_node = current_node.next
            return current_node.data
        
    def getItemLeft(self, position):
        if position > self.getLength():
            print ("This index is out of the list range")
            return None
        else:
            end = int(self.getLength())
            pos = end - position + 1
            return self.getItem(pos)
                    
# TEST 5
print (question5(['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'hhh'], 3))
print (question5(['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'hhh'], 20))   
print (question5([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 8))
print (question5([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 100))

print("--- %s seconds ---" % (time.time() - start_time))