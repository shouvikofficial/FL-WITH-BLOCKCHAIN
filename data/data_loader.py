import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_split_data(path, label_column, num_clients=3):
    # Load Excel
    df = pd.read_excel(path)
    df = df.dropna()

    X = df.drop(label_column, axis=1).values
    y = df[label_column].values

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Split indices
    split_indices = np.array_split(indices, num_clients)

    # Create client datasets (LISTS, not NumPy arrays)
    client_data = []
    for idx in split_indices:
        client_data.append(
            [(X[i], y[i]) for i in idx]
        )

    return client_data
