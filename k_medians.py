import numpy as np

def initialize_medians(data, k):
    """ Initialize random medians """
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, medians):
    """ Assign points to the nearest cluster """
    distances = np.linalg.norm(data[:, np.newaxis] - medians, axis=2)
    return np.argmin(distances, axis=1)

def update_medians(data, labels, k):
    """ Update medians based on the median values of the clusters """
    new_medians = []
    for i in range(k):
        if np.any(labels == i):
            new_medians.append(np.median(data[labels == i], axis=0))
        else:
            # Re-initialize medians for an empty cluster
            new_medians.append(data[np.random.randint(0, data.shape[0])])
    return np.array(new_medians)

def k_medians(data, k, iterations=100):
    """ k-medians algorithm """
    medians = initialize_medians(data, k)
    for _ in range(iterations):
        labels = assign_clusters(data, medians)
        new_medians = update_medians(data, labels, k)
        if np.allclose(medians, new_medians, atol=1e-2):
            break
        medians = new_medians
    return medians, labels

# Define datasets
datasets = [
    np.array([(1,1),(1,8),(2,2),(2,5),(3,1),(4,3),(5,2),(6,1),(6,8),(8,6)]),
    np.array([(1,1),(1,2),(1,5),(2,8),(3,7),(4,2),(7,5),(8,3),(8,7),(9,3)]),
    np.array([(2,1),(2,4),(3,5),(3,6),(4,1),(4,9),(5,4),(5,6),(7,2),(9,8)]),
    np.array([(1,4),(2,5),(2,8),(3,4),(3,5),(4,1),(4,7),(5,6),(7,6),(8,1)])
]

results = []
k = 3  # Number of clusters for each dataset

# Apply k-medians to each dataset
for i, data in enumerate(datasets, start=1):
    medians, labels = k_medians(data, k)
    results.append((medians, labels))
    print(f"Dataset {i}:")
    print("Medians:")
    print(medians)
    print("Cluster Assignments:")
    print(labels)
    print("\n")
