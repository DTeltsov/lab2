import matplotlib.pyplot as plt
import numpy as np


def distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def cluster_distance(cluster1, cluster2, data, method):
    distances = [distance(data[p1], data[p2]) for p1 in cluster1 for p2 in cluster2]
    if method == 'single':
        return min(distances)
    elif method == 'complete':
        return max(distances)
    else:  # average
        return np.mean(distances)


def hierarchical_clustering(data, method):
    n = data.shape[0]
    clusters = {i: [i] for i in range(n)}
    linkage_matrix = []
    current_cluster_label = n

    while len(clusters) > 1:
        # Знаходимо пару найближчих кластерів
        min_pair, min_dist = None, float('inf')
        keys = list(clusters.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                dist = cluster_distance(clusters[keys[i]], clusters[keys[j]], data, method)
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (keys[i], keys[j])

        # Об'єднуємо ці кластери
        new_cluster = clusters.pop(min_pair[0]) + clusters.pop(min_pair[1])
        clusters[current_cluster_label] = new_cluster
        linkage_matrix.append([min_pair[0], min_pair[1], min_dist, len(new_cluster)])
        current_cluster_label += 1

    # Візуалізація дендрограми
    from scipy.cluster.hierarchy import dendrogram
    plt.figure(figsize=(8, 4))
    dendrogram(linkage_matrix, labels=np.arange(n))
    plt.title(f'{method.capitalize()} Linkage Dendrogram')
    plt.xlabel('Index of Points')
    plt.ylabel('Distance')
    plt.show()


datasets = [
    np.array([(1, 1), (1, 8), (2, 2), (2, 5), (3, 1), (4, 3), (5, 2), (6, 1), (6, 8), (8, 6)]),
    np.array([(1, 1), (1, 2), (1, 5), (2, 8), (3, 7), (4, 2), (7, 5), (8, 3), (8, 7), (9, 3)]),
    np.array([(2, 1), (2, 4), (3, 5), (3, 6), (4, 1), (4, 9), (5, 4), (5, 6), (7, 2), (9, 8)]),
    np.array([(1, 4), (2, 5), (2, 8), (3, 4), (3, 5), (4, 1), (4, 7), (5, 6), (7, 6), (8, 1)])
]
methods = ['single', 'complete', 'average']
for data in datasets:
    for method in methods:
        print(f"Using {method}-link method:")
        hierarchical_clustering(data, method)
