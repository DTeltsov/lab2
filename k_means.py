import numpy as np


def initialize_centroids(data, k):
    """ Ініціалізація випадкових центроїдів """
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]


def assign_clusters(data, centroids):
    """ Призначення точок до найближчого кластера """
    distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)


def update_centroids(data, labels, k):
    """ Оновлення центроїдів на основі середніх значень кластерів """
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])


def k_means(data, k, iterations=100):
    """ Алгоритм k-means """
    centroids = initialize_centroids(data, k)
    for _ in range(iterations):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels


# Визначення наборів даних
datasets = [
    np.array([(1, 1), (1, 8), (2, 2), (2, 5), (3, 1), (4, 3), (5, 2), (6, 1), (6, 8), (8, 6)]),
    np.array([(1, 1), (1, 2), (1, 5), (2, 8), (3, 7), (4, 2), (7, 5), (8, 3), (8, 7), (9, 3)]),
    np.array([(2, 1), (2, 4), (3, 5), (3, 6), (4, 1), (4, 9), (5, 4), (5, 6), (7, 2), (9, 8)]),
    np.array([(1, 4), (2, 5), (2, 8), (3, 4), (3, 5), (4, 1), (4, 7), (5, 6), (7, 6), (8, 1)])
]

results = []
k = 3
for i, data in enumerate(datasets, start=1):
    centroids, labels = k_means(data, k)
    results.append((centroids, labels))
    print(f"Набір даних {i}:")
    print("Центроїди:")
    print(centroids)
    print("Призначення кластерів:")
    print(labels)
    print("\n")

