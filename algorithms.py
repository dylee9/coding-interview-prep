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
		for _ in range(count):
			tmp.append(0)

		self.graph.append(list())

	def addEdge(self, i, j):
		assert i < len(self.graph) 
		assert j < len(self.graph)
		self.graph[i][j] = 1

# As nodes

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

		for neighbor in range(G[node]):
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
		for neighbor in G.getNode(node_val).neighbors:
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

	





