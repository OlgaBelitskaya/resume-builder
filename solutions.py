# SOLUTION 1
import numpy as np

def question1(s, t):
    result = True
    for element in list(t):
        boolean = element in list(s)
        result = np.logical_and(result, boolean)
    return result

# TEST 1
s = "udacity"
t = ["cit", "ad", "mod"]
for el in t:
    print (el, "in", s, ":", question1(s, el))
    
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

def search_palindrome_in(string):
    palindromes = []
    cuts = cut_collection(string)
    for element in cuts:
        substrings = sub_collection(element)
        for el in substrings:
            if is_palindrome(el):
                palindromes.append(el)
    return palindromes
    
def is_palindrome(string):
    if string == "":
        return True
    else:
        if string[0] == string[-1]:
            return is_palindrome(string[1:-1])
        else:
            return False
        
def sub_collection(string):
    sub_string = list(string)
    sub_collection = []
    for i in range(len(sub_string)+1):
        join_string = "".join(sub_string[:i])
        sub_collection.append(join_string )   
    sub_collection = sub_collection[2:len(sub_string)+1]
    return sub_collection     

def cut_collection(string):
    cut_collection = []
    current = string
    for i in range(len(list(string))+1):
        cut_collection.append(current)
        cut_string = current[1:]
        current = cut_string
    cut_collection = cut_collection[:len(list(string))-1]
    return cut_collection

# TEST 2
test_cases = ['', 'test', 'mart', 'letter', 'parallelogram', 'radarrotator']
for test in test_cases:   
#    print (list(test))
#    print (string_collection(test))
#    print (cut_collection(test))
#    print (search_palindrome_in(test))
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
G = {'A': [('B', 2), ('D', 6), ('G', 4)],
     'B': [('A', 2), ('C', 5)], 
     'C': [('B', 5)],
     'D': [('B', 3), ('C', 4)],
     'E': [('C', 5), ('D', 8)],
     'F': [('C', 3), ('E', 7)],
     'G': [('B', 5), ('D', 5), ('F', 1)]}
print (question3(G))
W = create_wgraph(G)
print (W)
print (create_wdictionary(W))

# SOLUTION 4

# TEST 4


# SOLUTION 5

# TEST 5


