import heapq
from typing import Dict, List, Optional, Tuple

INF = float('inf')


class TransportGraph:
    def __init__(self):
        self.adj: Dict[str, Dict[str, float]] = {}
        self.edges: List[Tuple[str, str, float]] = []

    def add_station(self, name: str) -> None:
        if name not in self.adj:
            self.adj[name] = {}

    def add_route(self, u: str, v: str, w: float) -> None:
        self.add_station(u)
        self.add_station(v)
        self.adj[u][v] = w
        self.adj[v][u] = w
        if (v, u, w) not in self.edges:
            self.edges.append((u, v, w))

    def add_directed_route(self, u: str, v: str, w: float) -> None:
        self.add_station(u)
        self.add_station(v)
        self.adj[u][v] = w
        self.edges.append((u, v, w))

    def shortest_path_Dijkstra(self, start: str) -> Dict[str, float]:
        if start not in self.adj:
            return {}
        dist = {node: INF for node in self.adj}
        dist[start] = 0.0
        pq = [(0.0, start)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            for v, w in self.adj[u].items():
                if w < 0:
                    raise ValueError("Dijkstra не поддерживает отрицательные веса")
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def shortest_path_BellmanFord(self, start: str) -> Optional[Dict[str, float]]:
        dist = {node: INF for node in self.adj}
        dist[start] = 0.0
        n = len(self.adj)
        for _ in range(n - 1):
            updated = False
            for u, v, w in self.edges:
                if dist[u] != INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    updated = True
            if not updated:
                break
        for u, v, w in self.edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                return None
        return dist

    def all_pairs_shortest_paths_FloydWarshall(self) -> List[List[float]]:
        nodes = list(self.adj.keys())
        idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0.0
        for u in self.adj:
            i = idx[u]
            for v, w in self.adj[u].items():
                j = idx[v]
                dist[i][j] = min(dist[i][j], w)
        for k in range(n):
            for i in range(n):
                if dist[i][k] == INF:
                    continue
                for j in range(n):
                    if dist[k][j] == INF:
                        continue
                    nd = dist[i][k] + dist[k][j]
                    if nd < dist[i][j]:
                        dist[i][j] = nd
        return dist


class NetworkOptimizer:
    @staticmethod
    def _find(parent, x):
        if parent[x] != x:
            parent[x] = NetworkOptimizer._find(parent, parent[x])
        return parent[x]

    @staticmethod
    def _union(parent, rank, x, y):
        rx, ry = NetworkOptimizer._find(parent, x), NetworkOptimizer._find(parent, y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True

    @staticmethod
    def minimum_spanning_tree_Kruskal(edges, nodes):
        sorted_edges = sorted(edges, key=lambda x: x[2])
        parent = {node: node for node in nodes}
        rank = {node: 0 for node in nodes}
        mst = []
        for u, v, w in sorted_edges:
            if NetworkOptimizer._union(parent, rank, u, v):
                mst.append((u, v, w))
                if len(mst) == len(nodes) - 1:
                    break
        return mst

    @staticmethod
    def minimum_spanning_tree_Prim(graph, start):
        visited = {start}
        heap = []
        mst = []
        for nb, w in graph[start].items():
            heapq.heappush(heap, (w, start, nb))
        while heap and len(visited) < len(graph):
            w, u, v = heapq.heappop(heap)
            if v in visited:
                continue
            visited.add(v)
            mst.append((u, v, w))
            for nb, wt in graph[v].items():
                if nb not in visited:
                    heapq.heappush(heap, (wt, v, nb))
        return mst


def build_city_graph():
    g = TransportGraph()
    routes = [
        ("Центр. вокзал", "Пл. Ленина", 1.2),
        ("Центр. вокзал", "Автостанция-1", 3.5),
        ("Центр. вокзал", "Вокзал-Северный", 4.0),
        ("Пл. Ленина", "Автостанция-1", 2.1),
        ("Пл. Ленина", "Университет", 0.8),
        ("Пл. Ленина", "Автостанция-2", 3.0),
        ("Пл. Ленина", "Парковая", 1.5),
        ("Автостанция-1", "Автостанция-2", 5.4),
        ("Автостанция-1", "Портовая", 4.2),
        ("Университет", "Вокзал-Северный", 2.7),
        ("Университет", "Автостанция-2", 1.9),
        ("Вокзал-Северный", "Парковая", 3.3),
        ("Вокзал-Северный", "Портовая", 6.0),
        ("Автостанция-2", "Парковая", 2.5),
    ]
    for u, v, w in routes:
        g.add_route(u, v, w)
    return g


if __name__ == "__main__":
    # --- Пример из задания: станции A, B, C, D ---
    stations = ["A", "B", "C", "D"]
    routes = [("A", "B", 4), ("A", "C", 2), ("B", "C", 1), ("B", "D", 5), ("C", "D", 8)]

    g = TransportGraph()
    for s in stations:
        g.add_station(s)
    for u, v, w in routes:
        g.add_route(u, v, w)

    print("=" * 60)
    print("ПРИМЕР ИЗ ЗАДАНИЯ: станции A, B, C, D")
    print("=" * 60)
    d = g.shortest_path_Dijkstra("A")
    print(f"Кратчайшие пути от 'A' (Дейкстра): {d}")
    mst_k = NetworkOptimizer.minimum_spanning_tree_Kruskal(g.edges, stations)
    print(f"Минимальное остовное дерево (Крускал): {mst_k} "
          f"(суммарный вес = {sum(w for _, _, w in mst_k)})")
    mst_p = NetworkOptimizer.minimum_spanning_tree_Prim(g.adj, "A")
    print(f"Минимальное остовное дерево (Прим):   {mst_p} "
          f"(суммарный вес = {sum(w for _, _, w in mst_p)})")

    # --- Городская транспортная сеть 8x8 ---
    g_city = build_city_graph()
    print()
    print("=" * 60)
    print("ГОРОДСКАЯ ТРАНСПОРТНАЯ СЕТЬ 8x8")
    print("=" * 60)

    print("=== ДЕЙКСТРА от \'Центр. вокзал\' ===")
    dists = g_city.shortest_path_Dijkstra("Центр. вокзал")
    for station, dist in dists.items():
        print(f"  {station:<17}{dist:>5.1f} км")

    print()
    print("=== ФЛОЙД-УОРШЕЛЛ (справочник пассажира) ===")
    fw = g_city.all_pairs_shortest_paths_FloydWarshall()
    nodes = list(g_city.adj.keys())
    idx = {n: i for i, n in enumerate(nodes)}
    header = "".join(f"{n[:6]:>8}" for n in nodes)
    print(f"{'':>18}{header}")
    for n_row in nodes:
        row = fw[idx[n_row]]
        cells = "".join(f"{row[idx[n_col]]:>8.1f}" for n_col in nodes)
        print(f"{n_row:<18}{cells}")
    row_start = fw[idx["Центр. вокзал"]]
    check = all(abs(row_start[idx[k]] - dists[k]) < 1e-9 for k in dists)
    print(f"  Сверка строки \'Центр. вокзал\' с Дейкстрой: {check}")

    print()
    print("=== КРУСКАЛ (минимальная сеть) ===")
    mst_k = NetworkOptimizer.minimum_spanning_tree_Kruskal(g_city.edges, nodes)
    for u, v, w in mst_k:
        print(f"  {u} — {v}: {w} км")
    wk = sum(w for _, _, w in mst_k)
    print(f"  Суммарный вес: {wk:.1f} км")

    print()
    print("=== ПРИМ (минимальная сеть) ===")
    mst_p = NetworkOptimizer.minimum_spanning_tree_Prim(g_city.adj, "Центр. вокзал")
    for u, v, w in mst_p:
        print(f"  {u} — {v}: {w} км")
    wp = sum(w for _, _, w in mst_p)
    print(f"  Суммарный вес: {wp:.1f} км")
    print(f"  Крускал и Прим дали одно дерево: {sorted(mst_k) == sorted(mst_p)}")

    print()
    print("=== БЕЛМАН-ФОРД с \'выгодной\' пересадкой ===")
    g_bf = build_city_graph()
    g_bf.add_directed_route("Университет", "Автостанция-1", -1.0)
    dists_bf = g_bf.shortest_path_BellmanFord("Центр. вокзал")
    if dists_bf is None:
        print("  Обнаружен отрицательный цикл!")
    else:
        print(f"  До Автостанции-1: было {dists['Автостанция-1']}, "
              f"стало {dists_bf['Автостанция-1']}")
        print("  (путь: Центр. вокзал -> Пл. Ленина -> Университет -> Автостанция-1)")

    print()
    print("=== ПРОВЕРКА НА ОТРИЦАТЕЛЬНЫЙ ЦИКЛ ===")
    g_cycle = TransportGraph()
    g_cycle.add_directed_route("X", "Y", 1)
    g_cycle.add_directed_route("Y", "Z", 2)
    g_cycle.add_directed_route("Z", "X", -4)
    print(f"  X->Y(1), Y->Z(2), Z->X(-4) -> {g_cycle.shortest_path_BellmanFord('X')}")
