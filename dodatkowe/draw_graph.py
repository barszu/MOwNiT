T = [
    [0, 3, 0, 0],
    [3, 0, 1, 0],
    [0, 1, 0, 4],
    [0, 0, 4, 0]
]

T = [[0, 1, 200, 200, 200, 200], [1, 0, 2, 200, 200, 200], [200, 2, 0, 40, 200, 200], [200, 200, 40, 0, 40, 200], [200, 200, 200, 40, 0, 117], [200, 200, 200, 200, 117, 0]]


import networkx as nx
import matplotlib.pyplot as plt

# Stworzenie pustego grafu
G = nx.Graph()

# Dodawanie krawędzi z macierzy sąsiedztwa
n = len(T)  # liczba wierzchołków
for i in range(n):
    for j in range(i + 1, n):  # zaczynamy od i+1, aby nie dodawać dwukrotnie tych samych krawędzi
        if T[i][j] != 0 and T[i][j] != 200:  # jeśli waga nie jest zero, dodajemy krawędź
            G.add_edge(i, j, weight=T[i][j])

# Rysowanie grafu
pos = nx.spring_layout(G)  # układ grafu, spring layout próbuje rozmieścić wierzchołki tak, aby krawędzie były równomiernie rozciągnięte
labels = nx.get_edge_attributes(G, 'weight')  # pobieranie wag krawędzi
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700)  # rysowanie wierzchołków i krawędzi
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)  # dodanie etykiet wag do krawędzi

# Wyświetlenie grafu
plt.show()
