'''import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import streamlit as st
import io
import base64

def generate_dendrogram(data):
    Z = linkage(data, method='single', metric='euclidean')
    plt.figure(figsize=(8, 6))
    dendrogram(Z)
    plt.title('Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def main():
    st.title('Dendrogram Generator')

    st.write('Enter X and Y coordinates separated by commas.')

    x_input = st.text_input('X coordinates')
    y_input = st.text_input('Y coordinates')

    if st.button('Generate Dendrogram'):
        X = np.array([float(x) for x in x_input.split(',')])
        Y = np.array([float(y) for y in y_input.split(',')])
        data = np.column_stack((X, Y))
        dendrogram_image = generate_dendrogram(data)
        st.image(io.BytesIO(base64.b64decode(dendrogram_image)), caption='Dendrogram', use_column_width=True)

if __name__ == '__main__':
    main()
'''
import numpy as np
import streamlit as st

def generate_dendrogram(data):
    n = len(data)
    distances = np.zeros((n, n))

    # Calculate distances between each pair of points
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(data[i] - data[j])

    st.text("Dendrogram:")

    # Perform hierarchical clustering (single-linkage method)
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        min_dist = np.inf
        merge_i, merge_j = -1, -1
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                for k in clusters[i]:
                    for l in clusters[j]:
                        if distances[k, l] < min_dist:
                            min_dist = distances[k, l]
                            merge_i, merge_j = i, j
        new_cluster = clusters[merge_i] + clusters[merge_j]
        clusters.pop(max(merge_i, merge_j))
        clusters.pop(min(merge_i, merge_j))
        clusters.append(new_cluster)

        # Display dendrogram
        st.text(f"Cluster {merge_i} merges with Cluster {merge_j}: {clusters}")

def main():
    st.title('Dendrogram Generator')

    st.write('Enter X and Y coordinates separated by commas.')

    x_input = st.text_input('X coordinates')
    y_input = st.text_input('Y coordinates')

    if st.button('Generate Dendrogram'):
        X = np.array([float(x) for x in x_input.split(',')])
        Y = np.array([float(y) for y in y_input.split(',')])
        data = np.column_stack((X, Y))
        generate_dendrogram(data)

if __name__ == '__main__':
    main()
