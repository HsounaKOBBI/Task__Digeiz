import numpy as np
from sklearn.ensemble import IsolationForest


def detection_anomalies(frames):

    # Convertir les images en vecteurs plats
    flattened_frames = [frame.flatten() for frame in frames]

    # Entraîner un modèle Isolation Forest pour détecter les anomalies
    model = IsolationForest(n_estimators=220)
    model.fit(flattened_frames)

    # Prédire les anomalies dans les images
    anomalies = model.predict(flattened_frames)

    # Filtrer les images non pertinentes (anomalies)
    indices_images_non_pertinentes = [index for index, anomaly in enumerate(anomalies) if np.any(anomaly == -1)]
    print(indices_images_non_pertinentes)
    main_cluster_index = [valeur for valeur in list(range(1, len(frames))) if valeur not in indices_images_non_pertinentes]
    return main_cluster_index
