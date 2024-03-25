import numpy as np
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
