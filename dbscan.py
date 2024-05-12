import numpy as np


def euclidean_dist(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def find_neighbors(data, point_idx, eps):
    neighbors = []
    for i, point in enumerate(data):
        if euclidean_dist(data[point_idx], point) < eps:
            neighbors.append(i)
    return neighbors


def dbscan(data, eps, min_pts):
    labels = [-1] * len(data)  # -1 означає, що точка ще не відвідана
    cluster_id = 0

    for point_idx in range(len(data)):
        if labels[point_idx] != -1:
            continue

        # Знайдемо сусідів точки
        neighbors = find_neighbors(data, point_idx, eps)

        if len(neighbors) < min_pts:
            labels[point_idx] = -2  # -2 означає шум
            continue

        # Розпочати новий кластер
        labels[point_idx] = cluster_id

        # Обробка всіх точок у сусідстві
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if labels[neighbor] == -2:
                labels[neighbor] = cluster_id
            elif labels[neighbor] == -1:
                labels[neighbor] = cluster_id
                new_neighbors = find_neighbors(data, neighbor, eps)
                if len(new_neighbors) >= min_pts:
                    neighbors += new_neighbors
            i += 1

        cluster_id += 1

    return labels


# Датасет
data = np.array([
    [(1, 1), (1, 8), (2, 2), (2, 5), (3, 1), (4, 3), (5, 2), (6, 1), (6, 8), (8, 6)],
    [(1, 1), (1, 2), (1, 5), (2, 8), (3, 7), (4, 2), (7, 5), (8, 3), (8, 7), (9, 3)],
    [(2, 1), (2, 4), (3, 5), (3, 6), (4, 1), (4, 9), (5, 4), (5, 6), (7, 2), (9, 8)],
    [(1, 4), (2, 5), (2, 8), (3, 4), (3, 5), (4, 1), (4, 7), (5, 6), (7, 6), (8, 1)]
])

# Виконання DBSCAN для кожного набору даних
eps = 2.5
min_pts = 2
for i, dataset in enumerate(data, 1):
    labels = dbscan(dataset, eps, min_pts)
    print(f"Dataset {i} clustering results:")
    print("Labels:", labels)
