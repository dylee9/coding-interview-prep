# Test Cases

from algorithms import *

def create_generic_test_graph1(G):
	G.addVertex(0)
	G.addVertex(1)
	G.addVertex(2)
	G.addVertex(3)
	G.addVertex(4)
	G.addVertex(5)
	G.addEdge(0,1)
	G.addEdge(1,2)
	G.addEdge(2,3)
	G.addEdge(1,3)
	G.addEdge(1,4)
	G.addEdge(4,5)

	return G

G_AL = create_generic_test_graph1(Graph_AL())
G_AM = create_generic_test_graph1(Graph_AM())
G_UG = create_generic_test_graph1(Graph())

## Search Algorithm Tests

# Linear Search
assert lin_search([1,5,7,2,3,10,9], 2) == 3, "testcase 1 failed..."
assert lin_search([1,5,7,2,3,10,9], 6) == -1, "testcase 2 failed..."
print("Linear Search testcases passed...")

# Binary Search
assert bin_search_iter([1,3,6,8,10,20,22], 20) == 5, "testcase 3 failed..."
assert bin_search_iter([1,3,6,8,10,20,22], 24) == -1, "testcase 4 failed..."

assert bin_search_rec([1,3,6,8,10,20,22], 20) == 5, "testcase 5 failed..."
assert bin_search_rec([1,3,6,8,10,20,22], 24) == -1, "testcase 6 failed..."
print("Binary Search testcases passed...")

# DFS
assert DFS_iter_AL(G_AL, 3, 5) == True, "testcases 7 failed..."
assert DFS_iter_AL(G_AL, 2, 6) == False, "testcases 8 failed..."
assert DFS_rec_AL(G_AL, 3, 5) == True, "testcases 9 failed..."
assert DFS_rec_AL(G_AL, 2, 6) == False, "testcases 10 failed..."

assert DFS_iter_AM(G_AM, 3, 5) == True, "testcases 11 failed..."
assert DFS_iter_AM(G_AM, 2, 6) == False, "testcases 12 failed..."
assert DFS_rec_AM(G_AM, 3, 5) == True, "testcases 13 failed..."
assert DFS_rec_AM(G_AM, 2, 6) == False, "testcases 14 failed..."

assert DFS_iter_UG(G_UG, 3, 5) == True, "testcases 15 failed..."
assert DFS_iter_UG(G_UG, 2, 6) == False, "testcases 16 failed..."
assert DFS_rec_UG(G_UG, 3, 5) == True, "testcases 17 failed..."
assert DFS_rec_UG(G_UG, 2, 6) == False, "testcases 18 failed..."
print("Depth First Search testcases passed...")

# BFS
assert BFS_iter_AL(G_AL, 3, 5) == True, "testcases 19 failed..."
assert BFS_iter_AL(G_AL, 2, 6) == False, "testcases 20 failed..."
assert BFS_rec_AL(G_AL, 3, 5) == True, "testcases 21 failed..."
assert BFS_rec_AL(G_AL, 2, 6) == False, "testcases 22 failed..."

assert BFS_iter_AM(G_AM, 3, 5) == True, "testcases 23 failed..."
assert BFS_iter_AM(G_AM, 2, 6) == False, "testcases 24 failed..."
assert BFS_rec_AM(G_AM, 3, 5) == True, "testcases 25 failed..."
assert BFS_rec_AM(G_AM, 2, 6) == False, "testcases 26 failed..."

assert BFS_iter_UG(G_UG, 3, 5) == True, "testcases 27 failed..."
assert BFS_iter_UG(G_UG, 2, 6) == False, "testcases 28 failed..."
assert BFS_rec_UG(G_UG, 3, 5) == True, "testcases 29 failed..."
assert BFS_rec_UG(G_UG, 2, 6) == False, "testcases 30 failed..."
print("Breadth First Search testcases passed...")

print("All testcases passed succesfully!")