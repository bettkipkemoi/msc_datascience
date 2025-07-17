import heapq

def dijkstra(graph, start, end):
    queue = [(0, start, [])]
    visited = set()
    while queue:
        (cost, current, path) = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        path = path + [current]
        if current == end:
            return (cost, path)
        for neighbor, weight in graph.get(current, []):
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path))
    return float("inf"), []

# Graph definition
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('C', 5), ('D', 10)],
    'C': [('D', 3), ('E', 1)],
    'D': [('E', 7)],
    'E': []
}

# Execution
start, end = 'A', 'E'
distance, path = dijkstra(graph, start, end)
print(f"Shortest Path: {' -> '.join(path)}")
print(f"Total Distance: {distance}")