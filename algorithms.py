# Data Structures

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

# As adjacency matrix
class Graph_AM:
	def __init__(self):
		self.graph = []
		self.count = 0

	def addVertex(self, i):
		print(self.count)
		print(self.graph)
		self.count += 1

		if self.count > 1:
			for vertex in self.graph:
				vertex.append(0)

		tmp = []
		for _ in range(self.count+1):
			tmp.append(0)

		self.graph.append(list())

		print(self.count)
		print(self.graph)

	def addEdge(self, i, j):
		assert i < len(self.graph) 
		assert j < len(self.graph)
		self.graph[i][j] = 1

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


# Search Algos

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

	def helper(arr, lo, hi):
		if lo > hi:
			return -1

		mid = lo + (hi - lo) // 2
		if arr[mid] > n:
			return helper(arr[:mid], lo, mid-1)
		elif arr[mid] < n:
			return helper(arr[mid+1:], mid+1, hi)
		else:
			return mid

	return helper(arr, 0, len(arr)-1)

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
		print(node)
		if node == v:
			return True
		for neighbor in G[node]:
			if neighbor not in visited:
				visited.add(neighbor)
				stack.append(neighbor)

	return False

# Recursive approach using Adjacency List
# Time: O(V+E)
# Space: O(h)
def DFS_rec_AL(G, u, v):

	def dfs(visited, node):
		print(node)
		if G[node] == v:
			return True

		for neighbor in G[node]:
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
		print(curr)
		if curr == j:
			return True
		for node, connected in enumerate(G[curr]):
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
		print(curr)
		if curr == v:
			return True

		for node, connected in enumerate(G[curr]):
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
		print(node.value)
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
		print(node.value)
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
		print(node)
		if node == v:
			return True

		for neighbor in G[node]:
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
		print(node)
		if node == v:
			return True

		for neighbor in G[node]:
			if neighbor not in visited:
				visited.add(neighbor)
				queue.append(neighbor)

		return bfs(queue, visited)

	return bfs([], set())

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
		print(curr)
		if curr == v:
			return True

		for node, connected in enumerate(G[curr]):
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
		print(curr)
		if curr ==  v:
			return True

		for node, connected in enumerate(G[curr]):
			if connected:
				if node not in visited:
					visited.add(node)
					queue.append(node)

		return bfs(queue, visited)

	return bfs([], set())

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
		print(node.value)
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
		print(node.val)
		if node.val == v:
			return True

		for neighbor in node.neighbors:
			if neighbor.val not in visited:
				visited.append(neighbor.val)
				queue.append(neighbor)

		return bfs(queue, visited)

	return bfs([], set())

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

