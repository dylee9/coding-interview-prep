## Data Structures

# Graphs
# As adjacency list
class Graph_AL:
	def __init__(self):
		self.graph = dict()

	def addVertex(self, u):
		self.graph[u] = list()

	def addEdge(self, u, v):
		self.graph[u].append(v)
		self.graph[v].append(u)

# As DAGs using adjacency list
class Graph_DAG:
	def __init__(self):
		self.graph = dict()

	def addVertex(self, u):
		self.graph[u] = list()

	def addEdge(self, u, v):
		self.graph[u].append(v)

# As adjacency matrix
class Graph_AM:
	def __init__(self):
		self.graph = []
		self.count = 0

	def addVertex(self, i):
		self.count += 1

		if self.count > 1:
			for vertex in self.graph:
				vertex.append(0)

		tmp = []
		for _ in range(self.count+1):
			tmp.append(0)

		self.graph.append(tmp)

	def addEdge(self, i, j):
		assert i < len(self.graph) 
		assert j < len(self.graph)
		self.graph[i][j] = 1
		self.graph[j][i] = 1

# As Undirected Graph
class Node:
	def __init__(self, value):
		self.value = value
		self.neighbors = []

	def addNeighbor(self, neighbor):
		self.neighbors.append(neighbor)

class Graph:
	def __init__(self):
		self.nodes = []

	def getNode(self, value):
		for node in self.nodes:
			if node.value == value:
				return node
		return None

	def addVertex(self, value):
		self.nodes.append(Node(value))

	def addEdge(self, value1, value2):
		node1 = self.getNode(value1)
		node2 = self.getNode(value2)

		if node1 and node2:
			node1.addNeighbor(node2)
			node2.addNeighbor(node1)
		else:
			print("Error: nodes not found")

class WeightedGraph:
	def __init__(self):
		self.vertices = 0
		self.graph = []

	def addVertex(self, v):
		self.vertices += 1

	def addEdge(self, u, v, w):
		self.graph.append([u, v, w])

class WeightedAdjacencyMatrix(Graph_AM):
	def addEdge(self, i, j, w):
		assert i < len(self.graph) 
		assert j < len(self.graph)
		self.graph[i][j] = w
		self.graph[j][i] = w

# Linked Lists
class LLNode:
	def __init__(self, x):
		self.val = x
		self.next = None

## Search Algos

# Linear Search (lists, iterative)
# Time: O(N)
# Space: O(1)
def lin_search(arr, n):
	for i in range(len(arr)):
		if arr[i] == n:
			return i

	return -1

# Binary Search - Assumes arr is sorted, otherwise sort with merge sort O(log N)
# Iterative approach
# Time: O(log N)
# Space: O(1)
def bin_search_iter(arr, n):
	lo, hi = 0, len(arr)-1
	while lo <= hi:
		mid = lo + (hi - lo) // 2 #avoids overflow when hi+lo > 2^31
		if arr[mid] > n:
			hi = mid-1
		elif arr[mid] < n:
			lo = mid+1
		else:
			return mid

	return -1

# Recursive approach
# Time: O(log N)
# Space: O(log N) -> from call stack
def bin_search_rec(arr, n):

	def helper(lo, hi):
		if lo > hi:
			return -1

		mid = lo + (hi - lo) // 2
		if arr[mid] > n:
			return helper(lo, mid-1)
		elif arr[mid] < n:
			return helper(mid+1, hi)
		else:
			return mid

	return helper(0, len(arr)-1)

# Depth-First Search
# Iterative approach using Adjacency List
# Time: O(V+E)
# Space: O(h)
def DFS_iter_AL(G, u, v):

	visited = set()
	stack = []

	visited.add(u)
	stack.append(u)

	while stack:
		node = stack.pop()
		# print(node)
		if node == v:
			return True
		for neighbor in G.graph[node]:
			if neighbor not in visited:
				visited.add(neighbor)
				stack.append(neighbor)

	return False

# Recursive approach using Adjacency List
# Time: O(V+E)
# Space: O(h)
def DFS_rec_AL(G, u, v):

	def dfs(visited, node):
		# print(node)
		if node == v:
			return True

		for neighbor in G.graph[node]:
			if neighbor not in visited:
				visited.add(neighbor)
				res = dfs(visited, neighbor)
				if res: return True

		return False

	return dfs(set(), u)


# Iterative approach using Adjacency Matrix
# Time: O(V^2)
# Space: O(h)
def DFS_iter_AM(G, i, j):

	visited = set()
	stack = []

	visited.add(i)
	stack.append(i)

	while stack:
		curr = stack.pop()
		# print(curr)
		if curr == j:
			return True
		for node, connected in enumerate(G.graph[curr]):
			if connected: # checks whether edge exists
				if node not in visited:
					visited.add(node)
					stack.append(node)

	return False

# Recursive approach using Adjacency Matrix
# Time: O(V^2)
# Space: O(h)
def DFS_rec_AM(G, u, v):

	def dfs(visited, curr):
		# print(curr)
		if curr == v:
			return True

		for node, connected in enumerate(G.graph[curr]):
			if connected: # check whether edge exists
				if node not in visited:
					visited.add(node)
					res = dfs(visited, node)
					if res: return True

		return False

	return dfs(set(), u)

# Iterative approach using undirected graph
# Time: O(V+E)
# Space: O(h)
def DFS_iter_UG(G, u, v):

	visited = set()
	stack = []

	visited.add(u)
	stack.append(G.getNode(u))

	while stack:
		node = stack.pop()
		# print(node.value)
		if node.value == v:
			return True
		for neighbor in node.neighbors:
			if neighbor.value not in visited:
				visited.add(neighbor.value)
				stack.append(neighbor)

	return False

# Recursive approach using undirected graph
# Time: O(V+E)
# Space: O(h)
def DFS_rec_UG(G, u, v):

	def dfs(visited, node):
		# print(node.value)
		if node.value == v:
			return True

		for neighbor in node.neighbors:
			if neighbor.value not in visited:
				visited.add(node.value)
				res = dfs(visited, neighbor)
				if res: return True

		return False

	return dfs(set(), G.getNode(u))

# Breadth First Search

# Iterative approach using adjacency list
# Time: O(V+E)
# Space: O(w)
def BFS_iter_AL(G, u, v):
	visited = set()
	queue = []

	queue.append(u)
	visited.add(u)

	while queue:
		node = queue.pop(0)
		# print(node)
		if node == v:
			return True

		for neighbor in G.graph[node]:
			if neighbor not in visited:
				visited.add(neighbor)
				queue.append(neighbor)

	return False

# Recursive approach using adjacency list
# Time: O(V+E)
# Space: O(w)
def BFS_rec_AL(G, u, v):

	def bfs(queue, visited):
		if not queue:
			return False

		node = queue.pop(0)
		# print(node)
		if node == v:
			return True

		for neighbor in G.graph[node]:
			if neighbor not in visited:
				visited.add(neighbor)
				queue.append(neighbor)

		return bfs(queue, visited)

	queue = []
	visited = set()

	queue.append(u)
	visited.add(u)
	return bfs(queue, visited)

# Iterative approach using adjacency matrix
# Time: O(V^2)
# Space: O(w)
def BFS_iter_AM(G, u, v):
	visited = set()
	queue = []

	queue.append(u)
	visited.add(u)

	while queue:
		curr = queue.pop(0)
		# print(curr)
		if curr == v:
			return True

		for node, connected in enumerate(G.graph[curr]):
			if connected:
				if node not in visited:
					visited.add(node)
					queue.append(node)

	return False

# Recursive approach using adjacency matrix
# Time: O(V^2)
# Space: O(w)
def BFS_rec_AM(G, u, v):

	def bfs(queue, visited):
		if not queue:
			return False

		curr = queue.pop(0)
		# print(curr)
		if curr ==  v:
			return True

		for node, connected in enumerate(G.graph[curr]):
			if connected:
				if node not in visited:
					visited.add(node)
					queue.append(node)

		return bfs(queue, visited)

	queue = []
	visited = set()

	queue.append(u)
	visited.add(u)
	return bfs(queue, visited)

# Iterative approach using undirected graph
# Time: O(V+E)
# Space: O(w)
def BFS_iter_UG(G, u, v):

	visited = set()
	queue = []

	visited.add(u)
	queue.append(G.getNode(u))

	while queue:
		node = queue.pop(0)
		# print(node.value)
		if node.value == v:
			return True
		for neighbor in node.neighbors:
			if neighbor.value not in visited:
				visited.add(neighbor.value)
				queue.append(neighbor)

	return False

# Recursive approach using undirected graph
# Time: O(V+E)
# Space: O(w)
def BFS_rec_UG(G, u, v):

	def bfs(queue, visited):
		if not queue:
			return False

		node = queue.pop(0)
		# print(node.value)
		if node.value == v:
			return True

		for neighbor in node.neighbors:
			if neighbor.value not in visited:
				visited.add(neighbor.value)
				queue.append(neighbor)

		return bfs(queue, visited)

	queue = []
	visited = set()

	queue.append(G.getNode(u))
	visited.add(u)
	return bfs(queue, visited)

"""
Note about DFS/BFS Complexity:

1. AL & UG  vs AM
Unlike AL & UG, AM requires us to check each index of G[node] for each
node we visit. Because len(G[node]) == |V|, we get a space complexity of
O(V^2).

2. Recursive vs Iterative
Recursive and iterative approaches to DFS have no difference in time
space complexity because the technical data structure 'space' not used
in a recursive DFS function is instead used in the call-stack, which
requires space. For BFS, the same queue data structure is used for both
recursive & iterative so that makes the explanation easier.

3. Space Complexity
The easiest way to understand space complexity for DFS and BFS is by
how they process graphs. DFS looks to search narrow and deep; thus, 
the space complexity is the max height of the traversal tree represented
as h. BFS looks to search wide and shallow; thus, the space complexity
is the max width of the traversal tree represented as w.
"""

## Sorting Algos

# Insertion Sort
# Idea: For each element starting from the left, insert it between the smaller
#       and larger element to the left of it.
# Time: O(N^2) -> for every step, check every j to the left
# Space: O(1)
def insertion_sort(arr):

	for step in range(1, len(arr)): # assume first index is sorted
		key = arr[step]
		j = step - 1

		# compare key with each element to the left of it until smaller element
		# is found. Copy elements larger than key to the right until initial
		# smaller element and slide key in betweenn.
		while j >= 0 and key < arr[j]:
			arr[j+1] =  arr[j]
			j -= 1

		arr[j+1] = key

# Heap Sort
# Idea: Build a max-heap and pop the max element one at a time and heapify after
#       each pop until no nodes remain.

"""
Notes about Heaps

A binary tree is a heap iff:
1. it is a complete binary tree (all leaf elements lean to the left & last 
leaf element may not have a sibling)
2. all nodes follow the property that the parent node has a greater value
than its children (or smaller value for min heap)
"""

def heapify(arr, n, i):
	# find the largest between parent and 2 children 
	largest = i
	l = 2 * i + 1
	r = 2 * i + 2

	if l < n and arr[i] < arr[l]:
		largest = l

	if r < n and arr[largest] < arr[r]:
		largest = r

	# if parent is not largest, swap with largest and heapify down for swapped
	if largest != i:
		arr[i], arr[largest] = arr[largest], arr[i] # swap
		heapify(arr, n, largest)

# Time: O(Nlog N)
# Space: O(1)
"""
Explanation for time complexity:

Step 1: Building max/min heap
We heapify for n/2 elements. Heapify will swap the root element all the way
to the bottom of the tree at the worst case. The height of the tree is log n
since it is a binary tree. Therefore, we do roughly n/2 * log n operations ~
O(nlog n)

Step 2: Sorting step
We swap the root with the bottom element: O(1) and then we heapify the root,
which will do at most log n swaps. We do this for all n nodes: O(nlog n)
"""
def heap_sort(arr):
	n = len(arr)

	# build a max-heap
	# run heapify for every non-leaf node starting with bottom-right-most node
	for i in range(n//2, -1, -1): # n//2 is the right-most non-leaf node
		heapify(arr, n, i)

	# sort 
	for i in range(n-1, 0, -1):
		# swap max (root) with last node
		arr[i], arr[0] = arr[0], arr[i] 

		# heapify from root (bubble down the minimum value)
		# remove bottom right node by setting n argument to i
		heapify(arr, i, 0)

# Selection Sort
# Idea: Find the smallest element and place it at the beginning of the array.
#       Repeat again for elements that are not yet sorted.
# Time: O(N^2)
# Space: O(1)
def selection_sort(arr):
	for i in range(len(arr)):
		min_idx = i
		for j in range(i+1, len(arr)):
			if arr[j] < arr[min_idx]:
				min_idx = j
		arr[i], arr[min_idx] = arr[min_idx], arr[i]

# Merge Sort
# Idea: Divide & Conquer algorithm. Divide array into sub-arrays and merge
#       them in a sorted manner. Merge sub-arrays all the way up to the final.
# Time: O(Nlog N) -> at each level of the recursion call stack (logN levels),
#                    we process N elements. Thus, O(N * log N).
# Space: O(N) -> auxillary array L & R are created of size N
def merge_sort(arr):
	# base case - single element in array is already sorted
	if len(arr) <= 1:
		return

	# recursive case
	else:
		mid = len(arr)//2
		L = arr[:mid]
		R = arr[mid:]

		# assume Left and Right halves are already sorted
		merge_sort(L)
		merge_sort(R)

		# three pointers (one for L, one for R and one for main)
		i = j = k = 0

		while i < len(L) and j < len(R):
			if L[i] < R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				j += 1
			k += 1

		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1

		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1

# Quick Sort
# Idea: Divide and Conquer sorting algorithm. Select a pivot and move all 
#		elements smaller to the left and larger to the right of it. Recursively
#       apply the same logic to subarrays left and right to the pivot.
# Time: O(Nlog N) -> log N levels * sum of all elements = log N * N
# Space: O(N) -> worst case N level recursive call stack when pivot is always
#                the smallest element
def quick_sort(arr):

	def partition(lo, hi):
		p = arr[hi]
		j = lo
		for i in range(lo, hi):
			if arr[i] < p:
				arr[i], arr[j] = arr[j], arr[i]
				j += 1

		arr[j], arr[hi] =  arr[hi], arr[j]

		return j

	def quick_sort_rec(lo, hi):
		if lo < hi:
			pivot = partition(lo, hi)

			quick_sort_rec(lo, pivot-1)
			quick_sort_rec(pivot+1, hi)

	quick_sort_rec(0, len(arr)-1)


# Counting Sort
# Idea: Create a counter for each element in array and use that to generate
#       a sorted array. Uses more space than other sorting algos.
# Time: O(N + max) -> max is the largest value 
# Space: O(max)
def counting_sort(arr):
	maximum = max(arr)
	count = [0] * (maximum+1)
	output = [0] * len(arr)

	# build count array
	for i in range(len(arr)):
		count[arr[i]] += 1

	# make it running cumulative
	for i in range(1, len(count)):
		count[i] += count[i-1]

	# build output array
	for i in range(len(arr)):
		output[count[arr[i]]-1] = arr[i]
		count[arr[i]] -= 1

	# copy output back to array
	for i in range(len(output)):
		arr[i] =  output[i]


## Graph Algos

# Kruskal's Algorithm (for Weighted Graph)
# Idea: Greedy Algorithm used to find the minimum spanning tree in a connected
#       weighted graph by picking edges starting with the lowest weights and
#       adding them to the MST as long as a cycle is not created in the MST.
# Time: O(Elog E) or O(Elog V)
# Space: O(V + E)
"""
Explanation for Time Complexity:

1) Sorting all edges takes O(Elog E)

2) While Loop: Iterate through all edges and apply find-union algorithm. Find
   with path compression takes O(log V). Union with rank takes O(log V).
   Thus, this step takes O(Elog V)

Overall time complexity is O(Elog E + Elog V) ~ O(Elog E) because E is at most
V^2 and O(log V^2) = 2O(log V) ~ O(log V). Thus, final complexity is O(Elog E)
or O(Elog V).
"""
def kruskals_mst(G):

	# finds absolute parent of a node in a graph
	# with path compression to improve runtime
	def find(parent, i):
		if parent[i] != i:
			parent[i] = find(parent, parent[i]) # path compression

		return parent[i]

	# combines the two sets that contain x and y (if x and y are in the same
	# set, then a cycle is formed)
	# Uses union by rank to improve runtime
	def union(parent, rank, x, y):
		if rank[x] < rank[y]:
			parent[x] = y
		elif rank[x] > rank[y]:
			parent[y] = x
		else:
			parent[x] = y
			rank[y] += 1

	# output of edges in resulting MST
	result = []

	# array to keep track of absolute parents of nodes
	parent = []

	# array to keep track of ranks of absolute parents
	rank = []

	# populate parent and rank with initial values
	# initially, every node is it's own parent and rank is 0
	for node in range(G.vertices):
		parent.append(node)
		rank.append(0)

	# index for G.edges
	i = 0

	# counter for number of vertices in MST
	e = 0

	# STEP 1: sort edges in graph by weight (low -> high)
	G.graph = sorted(G.graph, key=lambda item: item[2])

	# STEP 2: For each edge (starting with the lowest weight), if adding it
	#         to the MST does NOT create a cycle, then we add it to the MST.
	while e < G.vertices - 1: #MST always has n-1 edges where n = number of vertices
		u, v, w = G.graph[i] # lowest weight edge
		i += 1

		# find absolute parents of two nodes
		x = find(parent, u)
		y = find(parent, v)

		# if u and v don't share the same set, then we add the edge to MST
		# because it does not result in a cycle
		if x != y:
			e = e + 1
			result.append([u, v, w])

			# update parents and ranks via union by rank
			union(parent, rank, x, y)

	return result

# Dijkstra's Algorithm (for Weighted Adjacency Matrix)
# Idea: Greedy algorithm to find the Shorest Path Tree(SPT) with a given source
#       as the root. Method is to greedily find the next shortest vertex
#       to the SPT, update the distances to visited nodes and repeat
#       until all nodes have been added to SPT.
# Time:
# Space:
def dijkstras_spt(G, u):

	def minDistance(dist, sptSet):
		min = float('inf')

		# for every vertex nott in SPT, find min distance one
		for u in range(G.count):
			if dist[u] < min and not sptSet[u]:
				min = dist[u]
				min_idx = u

		return min_idx

	count = 0

	# array to keep track of distances from source u
	dist = [float('inf')] * G.count
	dist[u] = 0 

	# array to keep track of SPT set of vertices
	sptSet = [False] * G.count

	# add vertices to SPT until SPT holds all vertices in graph
	while count < G.count:

		# STEP 1: Pick the minimum distance vertex from the set of vertices
		#         not yet in SPT. Note: x will always be source u in the
		#         first iteration because dist[u] will be the minimum distance
		#         vertex with dist[u] == 0.
		x = minDistance(dist, sptSet)

		# STEP 2: Add the minimum distance vertex to the SPT
		sptSet[x] = True

		# STEP 3: For each neighbor of x (G.graph[x][y] > 0) that is not
		#         already in the SPT (not sptSet[y]), if the new distance
		#         to y from source u through x is shorter than the
		#         previously known distance to y not through x, then update
		#         the new distance to y with the shorter distance.
		for y in range(G.count):
			if G.graph[x][y] > 0 and not sptSet[y] and \
					dist[y] > dist[x] + G.graph[x][y]:
				dist[y] = dist[x] + G.graph[x][y]

		count += 1

	return dist

# Bellman-Ford Algorithm (for Weighted Graph)
# Idea: Is a single source shortest path algorithm much like Dijkstra's.
#       However, BF algorithm can work with negative weight edges as long 
#       as there are no negative cycles. As a result, it can also be used to 
#       check for negative cycles if a reduction in distance can be found on
#       the Nth iteration. This is because the algorithm expects all minimum
#       distances to be found by the N-1 iteration since a shortest path
#       betweeen N nodes has N-1 edges. The algorithm relies on the idea that
#       after N-1 distance reductions, the optimal distances have been found
#       and, otherwise, a negative cycle is found.
# Time: O(VE) -> N iterations and loop through O(E) edges per iteration
# Space: O(V) -> dist table
def bellman_ford(G, u):

	# array to keep track of min distances from source
	dist = [float('inf')] * G.vertices
	dist[u] = 0

	# For N-1 iterations, loop through all edges (u,v,w) for visited nodes
	# and evaluate whether we can reduce the distance to v by going 
	# through u and adding w.
	for iteration in range(G.vertices - 1):
		for u, v, w in G.graph:
			if dist[u] != float('inf') and dist[u] + w < dist[v]:
				dist[v] = dist[u] + w 

	# If there is a reduction in dist in the Nth iteration, then that
	# implies a negative cycle exists.
	for u, v, w in G.graph:
		if dist[u] != float('inf') and dist[u] + w < dist[v]:
			raise Exception("Negative cycle found")
			return

	return dist

# Floyd-Warshall Algorithm (for Weighted Graph)
# Idea: Finds the shortest path between all pairs of vertices and works with
#       negative edges as well. However, it will stil l not work with negative
#       cycles. This approach takes on dynamic programming and uses 3 nested
#       loops to update the DP table of shortest path pairs.
# Time: O(V^3) -> 3 nested loops
# Space: O(V^2) -> 2d dist table (if AM is used as the graph, then O(1))
def floyd_warshall(G):

	# array to keep track of node pair shortest paths
	dist = [[float('-inf') for _ in range(G.vertices)] for _ in range(G.vertices)]

	# shortest distance to itself is 0
	for i in range(G.vertices):
		dist[i][i] = 0

	# initially, shortest distance from u,v is w if edge exists 
	for u, v, w in G.graph:
		dist[u][v] = w

	# nested 3 loop
	# for every 3 nodes (i,j,k) in the graph, if the distance from i to j
	# is larger than taking a detour through k (i to k, k to j), then the
	# shortest distance from i to j is through k.
	for i in range(G.vertices):
		for j in range(G.vertices):
			for k in range(G.vertices):
				if dist[i][j] > dist[i][k] + dist[k][j]:
					dist[i][j] = dist[i][k] + dist[k][j]

	return dist

# Topological Sorting Algorithm (for Adjacency List)
# Idea: Linear ordering of DAGs (Directed Acyclic Graphs). Essentially,
#       we pick an unvisited node and apply DFS recursively and adding
#       visited nodes to a stack AFTER visiting all it's neighbors.
#       By doing so, we get write the lower level nodes to the stack
#       first and then we can reverse the stack for the actual topological
#       sorted order.
# Time: O(V + E) -> Same as DFS
# Space: O(V) -> keep tracking of stack
def top_sort(G):

	# DFS but keeps track of a stack and adds current node to it once
	# all of it's neighbors are visited first
	def dfs(u, visited, stack):
		visited[u] = True

		for neighbor in G.graph[u]:
			if not visited[neighbor]:
				dfs(neighbor, visited, stack)

		stack.append(u)

	# array to keep track of visited nodes
	visited = [False] * len(G.graph)

	# output array for topological ordering
	stack = []

	# start at any arbitrary node and apply DFS
	# once all lowest level nodes are reached from that node, pick another
	# node and repeat until all vertices have been visited
	for v in range(G.graph):
		if not visited[v]:
			dfs(v, visited, stack)

	# stack is the order of low to high but we want to return high to low
	return stack[::-1]

# Flood Fill Algorithm
# Idea: A dynamic programming + BFS algorithm to fill an index of a 2D matrix
#       with a new color and spread it out to all adjacent indices with the 
#       same color until no more valid indices are left to change. 
#       Invalid indices examples include going past the boundaries of the 
#       matrix, running into a wall or a different color index.
# Time: O(M*N) -> worst case we run bfs for every cell of matrix
# Space: O(M*N) -> worst case we have a queue with every cell of matrix
def flood_fill(G, x, y, new_color):

	n = len(G)
	m = len(G[0])
	prev_color = G[y][x]
	dirs = [(1,0),(0,1),(-1,0),(0,-1)]

	# helper function to determine whether cell should be flood-filled.
	# cells that are OOB or already the new color or not the previous
	# color are not valid cells to be flood-filled.
	def is_valid(i, j):
		if i < 0 or i >= n \
				or j < 0 or j >= m \
				or G[i][j] != prev_color \
				or G[i][j] == new_color:
			return False

		return True

	# BFS function that checks 4 directions (up, right, down, left) for valid
	# cells and flood-fills them
	def bfs(x, y):
		Q = []
		Q.append((y,x))
		G[y][x] = new_color

		while Q:
			posX, posY = Q.pop()

			for dirX, dirY in dirs:
				if is_valid(posY+dirY, posX+dirX):
					G[posY+dirY][posX+dirX] = new_color
					Q.append((posY+dirY,posX+dirX))

	return G

# Lee's Algorithm
# Idea: In a MxN matrix, use BFS to find the shortest path from a source cell
#       to the target cell. Often, this is used to solve the shortest path 
#       from the start to the cheese within the maze. In the matrix, 1 is used
#       to represent an open path and 0 to represent a wall.
# Time: O(M*N) -> worst case, visit every cell before finding target
# Space: O(M*N) -> visited matrix
def lees(G, x1, y1, x2, y2):

	n = len(G)
	m = len(G[0])
	dirs = [(1,0),(0,1),(-1,0),(0,-1)]

	# helper function to determine if index is OOB or a wall
	def is_valid(i, j):
		if i < 0 or i >= n \
				or j < 0 or j >= m \
				or G[j][i] == 0:
			return False

		return True

	# BFS algorithm that visits adjacent neighbors and the Q keeps
	# track of the depth/distance
	Q = []
	Q.append((y1, x1, 0))

	# array to keep track of visited cells
	visited = [[False for _ in range(m)] for _ in range(n)]
	visisted[y1][x1] = True

	# iterative BFS using queue
	while Q:
		posY, posX, dist = Q.pop(0)

		# if target is found, return dist
		if posX == x2 and posY == y2:
			return dist

		# for every neighbor in left, right, up down direction, if
		# valid cell and not visited, add to queue and repeat
		for dirX, dirY in dirs:
			if is_valid(posY+dirY, posX+dirX) \
					and not visited[posY+dirY][posX+dirX]:
				Q.append((posY+dirY, posX+dirX, dist+1))
				visited[posY+dirY][posX+dirX] = True

	return -1

## Array Algos

# Kadane's Algorithm
# Idea: A greedy/DP solution to the largest continiguous subarray problem.
#       Main idea is that we can calculate the largest continuguous
#       subarray ending at each position i of the array by taking the max
#       of either the maximum continuguous subarray sum of the previous
#       index + the current index or without the current index. 
# Time: O(n)
# Space: O(1)
def kadanes(arr):

	# keeps track of max sum for each ending index
	current_max = 0

	# keeps track of global max for output
	global_max = float('-inf')

	# find the largest contiguous subarray sum for each ending position i
	for i in range(len(arr)):

		# we can either choose to start new max at i or keep max subarray before
		# i and add arr[i]. Take the max of either.
		current_max = max(arr[i], current_max+arr[i])

		# update global max if found
		if current_max > global_max:
			global_max = current_max

	return global_max

# Floyd's Cycle Detection Algorithm
# Idea: A pointer algorithm to detect cycles in linked lists. The main idea is
#       to have one pointer going fast along a linked list (2 nodes at a time)
#       and another pointer going slow along a linked list (1 node at a time).
#       The main idea is that if the slow and fast pointer ever reach the same
#       node, then a cycle is found because it implies the fast pointer went
#       through a cycle to end up back behind the slow pointer and eventually
#       land on the same node as the slow pointer. If either pointer reaches
#       the end of a LL, then there is no cycle.
# Time: O(n)
# Space: O(1)
def floyds_cycle_detection(head):
	fast = head
	slow = head

	while slow != None and fast != None and fast.next != None:
		slow = slow.next
		fast = fast.next.next

		if slow == fast:
			return 1

	return 0

# KMP (Knuth-Morris-Pratt) Algorithm
# Idea: A string pattern matching algorithm. The brute force approach takes 
#       O(n*m) since it has to backtrack whenever a mistmatch is found. With
#       KMP, we reduce any backtracking of the text string by using a longest
#       suffix preffix array, which acts as a dictionary that tells us where in
#       the pattern we are still matched up to regardless of the latest
#       mismatch. Thus, we don't backtrack on the text and only on the pattern.
# Time: O(n+m)
# Space: O(1)
def KMP(text, pattern):
	m = len(pattern)
	n = len(text)

	def generateLPSArray():

		# array to keep track of longest preffix suffix length
		lps = [0] * m

		# keep track of preffix
		l = 0

		# loop through pattern
		i = 1

		while i < m:
			if pattern[i] == pattern[l]:
				l += 1
				lps[i] = l
				i += 1
			else:
				# tricky case: similar to KMP but basically backtrack within
				#              within pattern to last seen preffix that might
				#              still match
				if l != 0:
					l = lps[l-1]

				else:
					lps[i] = 0
					i += 1

		return lps

	lps = generateLPSArray()
	output = []
	i = 0
	j = 0

	# when there are equal or more characters left in text compared to pattern
	# (if there isn't, then pattern can't match anyways)
	while (n - i) >= (m - j):

		if text[i] == pattern[j]:

			# full match found
			if j == m-1:
				output.append(i-j)
				# pattern pointer backtracks to last seen longest preffix length
				j = lps[j-1]

			i += 1
			j += 1

		elif j > 0:
			# pattern pointer backtracks to last seen longest preffix length
			j = lps[i-1]

		else:
			i += 1

	return output

# Quick Select -> finds the kth smallest/largest element in an unsorted list
# Idea: Similar to quick sort except instead of recursively partitioning both
#       sides of the pivot, only recurse in the section that holds the kth
#       value.
# Time: O(n)
# Space: O(1)
def quick_select(arr, k):

	def partition(lo, hi):
		p = arr[hi]
		j = lo
		for i in range(lo, hi):
			if arr[i] < p:
				arr[i], arr[j] = arr[j], arr[i]
				j += 1

		arr[j], arr[hi] =  arr[hi], arr[j]

		return j

	def quick_select_rec(lo, hi, k):
		if lo < hi:
			pivot = partition(lo, hi)

			if pivot == k-1:
				return arr[pivot]
			elif pivot < k-1:
				return quick_select_rec(pivot+1, hi, k-pivot-1)
			else:
				return quick_select_rec(lo, pivot-1, k)

	return quick_select_rec(0, len(arr)-1, k)

# Boyer-Moore Voting Algorithm
# Idea: A voting algorithm used to find the majority element in an array. A
#       majority element is one that occurs atleast N/2 times in the array.
#       The algorithm works by voting for candidate elements without having
#       to keep a counter for each distinct element. We iterate once to find
#       the majority element and then another loop to count if it has more
#       than N/2 occurances. 
# Time: O(n)
# Space: O(1)
def boyer_moore_voting(arr):
	candidate = -1
	votes = 0

	# loop 1 - take votes for each element
	# majority element will come out from this loop if it TRULY is the majority
	for e in arr:
		if not votes:
			candidate = e
			votes += 1
		else:
			if e == candidate:
				votes += 1
			else:
				votes -= 1

	# loop 2 - count majority
	count = 0
	for e in arr:
		if e == candidate:
			count += 1

	if count >= len(arr)//2:
		return candidate
	else:
		return -1

## Basics

# Huffman Coding Compression Algorithm



	

