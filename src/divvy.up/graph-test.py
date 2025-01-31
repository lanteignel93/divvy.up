import random
from collections import deque

INF = float("inf")
OFFSET = int(1000000000 * random.random())


class Edge:
    def __init__(self, from_node, to_node, capacity, cost=0):
        self.from_node = from_node
        self.to_node = to_node
        self.capacity = capacity
        self.cost = cost
        self.flow = 0
        self.residual = None

    def is_residual(self):
        return self.capacity == 0

    def remaining_capacity(self):
        return self.capacity - self.flow

    def augment(self, bottle_neck):
        self.flow += bottle_neck
        self.residual.flow -= bottle_neck

    def __str__(self):
        return f"Edge {self.from_node} -> {self.to_node} | flow = {self.flow} | capacity = {self.capacity} | is residual: {self.is_residual()}"


class NetworkFlowSolverBase:
    def __init__(self, n, vertex_labels):
        self.n = n
        self.vertex_labels = vertex_labels
        self.graph = [[] for _ in range(n)]
        self.edges = []
        self.max_flow = 0
        self.min_cost = 0
        self.min_cut = [False] * n
        self.solved = False
        self.s = None
        self.t = None

    def add_edge(self, from_node, to_node, capacity, cost=0, verbose=False):
        if capacity < 0:
            raise ValueError("Capacity < 0")
        e1 = Edge(from_node, to_node, capacity, cost)
        e2 = Edge(to_node, from_node, 0, -cost)
        e1.residual = e2
        e2.residual = e1
        self.graph[from_node].append(e1)
        self.graph[to_node].append(e2)
        self.edges.append(e1)
        if verbose:
            print(
                self.vertex_labels[from_node],
                " owes ",
                self.vertex_labels[to_node],
                capacity,
            )

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge.from_node, edge.to_node, edge.capacity, edge.cost)

    def solve(self):
        pass  # Abstract method

    def get_graph(self):
        self._execute()
        return self.graph

    def get_edges(self):
        return self.edges

    def get_max_flow(self):
        self._execute()
        return self.max_flow

    def get_min_cost(self):
        self._execute()
        return self.min_cost

    def get_min_cut(self):
        self._execute()
        return self.min_cut

    def set_source(self, s):
        self.s = s

    def set_sink(self, t):
        self.t = t

    def get_source(self):
        return self.s

    def get_sink(self):
        return self.t

    def recompute(self):
        self.solved = False

    def print_edges(self):
        for edge in self.edges:
            print(
                f"{self.vertex_labels[edge.from_node]} ----{edge.capacity}----> {self.vertex_labels[edge.to_node]}"
            )

    def _execute(self):
        if self.solved:
            return
        self.solved = True
        self.solve()


class DebtSimplifier(NetworkFlowSolverBase):
    def __init__(self, n, vertex_labels):
        super().__init__(n, vertex_labels)
        self.level = [-1] * n

    def solve(self):
        while self._bfs():
            next_edge = [0] * self.n
            while True:
                flow = self._dfs(self.s, next_edge, INF)
                if flow == 0:
                    break
                self.max_flow += flow

        for i in range(self.n):
            if self.level[i] != -1:
                self.min_cut[i] = True

    def _bfs(self):
        self.level = [-1] * self.n
        self.level[self.s] = 0
        q = deque([self.s])
        while q:
            node = q.popleft()
            for edge in self.graph[node]:
                cap = edge.remaining_capacity()
                if cap > 0 and self.level[edge.to_node] == -1:
                    self.level[edge.to_node] = self.level[node] + 1
                    q.append(edge.to_node)
        return self.level[self.t] != -1

    def _dfs(self, at, next_edge, flow):
        if at == self.t:
            return flow

        num_edges = len(self.graph[at])
        while next_edge[at] < num_edges:
            edge = self.graph[at][next_edge[at]]
            cap = edge.remaining_capacity()
            if cap > 0 and self.level[edge.to_node] == self.level[at] + 1:
                bottle_neck = self._dfs(edge.to_node, next_edge, min(flow, cap))
                if bottle_neck > 0:
                    edge.augment(bottle_neck)
                    return bottle_neck
            next_edge[at] += 1
        return 0


def create_graph_for_debts():
    person = ["Alice", "Bob", "Charlie", "David", "Ema", "Fred", "Gabe"]
    n = len(person)
    solver = DebtSimplifier(n, person)
    solver = add_all_transactions(solver)

    print("\nSimplifying Debts...")
    print("--------------------")
    print("\n")

    visited_edges = set()

    while True:
        edge_pos = get_non_visited_edge(solver.get_edges(), visited_edges)
        if edge_pos is None:
            break

        solver.recompute()
        first_edge = solver.get_edges()[edge_pos]
        solver.set_source(first_edge.from_node)
        solver.set_sink(first_edge.to_node)
        residual_graph = solver.get_graph()
        new_edges = []

        for all_edges in residual_graph:
            for edge in all_edges:
                remaining_flow = (
                    edge.capacity if edge.flow < 0 else (edge.capacity - edge.flow)
                )
                if remaining_flow > 0:
                    new_edges.append(Edge(edge.from_node, edge.to_node, remaining_flow))

        max_flow = solver.get_max_flow()
        source = solver.get_source()
        sink = solver.get_sink()
        visited_edges.add(get_hash_key_for_edge(source, sink))

        solver = DebtSimplifier(n, person)
        solver.add_edges(new_edges)
        solver.add_edge(source, sink, max_flow, verbose=False)

    solver.print_edges()
    print("\n")


def add_all_transactions(solver):
    solver.add_edge(1, 2, 40, verbose=True)
    solver.add_edge(2, 3, 20, verbose=True)
    solver.add_edge(3, 4, 50, verbose=True)
    solver.add_edge(5, 1, 10, verbose=True)
    solver.add_edge(5, 2, 30, verbose=True)
    solver.add_edge(5, 3, 10, verbose=True)
    solver.add_edge(5, 4, 10, verbose=True)
    solver.add_edge(6, 1, 30, verbose=True)
    solver.add_edge(6, 3, 10, verbose=True)
    return solver


def get_non_visited_edge(edges, visited_edges):
    for i, edge in enumerate(edges):
        if get_hash_key_for_edge(edge.from_node, edge.to_node) not in visited_edges:
            return i
    return None


def get_hash_key_for_edge(u, v):
    return u * OFFSET + v


if __name__ == "__main__":
    create_graph_for_debts()
