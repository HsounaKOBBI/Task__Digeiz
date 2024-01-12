import cv2
from PIL import Image
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

from parameters import cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_resnet_embeddings_of_video( frames):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # remove the last layer (fc layer) to get the embeddings and not the cls logits

    outputs = None
    tensor_frames = []

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #extract frames from video
    for frame in frames:
        #preprocess the frames so that we can later feed them to resnet18
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        tensor_frames.append(frame)
    print(f"Number of frames: {len(tensor_frames)}")

    #stack in a mini batch
    tensor_frames = torch.stack(tensor_frames)
    tensor_frames = tensor_frames.to(device)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    with torch.no_grad():
        outputs = feature_extractor(tensor_frames)
    return outputs.squeeze()
def clustering_resnet(embeddings):
    pca = PCA(n_components=cfg.apply_pca)
    embeddings_pca = pca.fit_transform(embeddings)
    # should be (num_frames, cfg.apply_pca)
    print(f"Embeddings shape after PCA dim reduction: {embeddings_pca.shape}")

    if (cfg.cluster_method.lower() == "kmeans"):
        num_clusters = 2
        kmean_model = KMeans(num_clusters, n_init='auto', random_state=0)
        clustering_results = kmean_model.fit_predict(embeddings_pca)
        print(f"Cluster results kmeans :\n{clustering_results}")

    elif (cfg.cluster_method.lower() == "gmm"):
        gmm = GaussianMixture(n_components=1, n_init=12, random_state=10)
        gmm.fit(embeddings_pca)
        densities = gmm.score_samples(embeddings_pca)

        density_threshold = np.percentile(densities, 10)
        print(f"Density threshold: {density_threshold}")
        clustering_results = np.array([densities < density_threshold]).astype(int).squeeze()
        print(f"Cluster results gmm:\n{clustering_results}")

    elif (cfg.cluster_method.lower() == "hierarchical"):
        hc = AgglomerativeClustering(n_clusters=2)
        clustering_results = hc.fit_predict(embeddings_pca)
        print(f"Cluster results hierarchical :\n{clustering_results}")

    else:
        raise ValueError("Unknown clustering method")
    unique, counts = np.unique(clustering_results, return_counts=True)
    print("Clustering cluster sizes:")
    print(dict(zip(unique, counts)))
    main_cluster = unique[np.argmax(counts)]
    print(f"Main cluster: {main_cluster}")
    # get the indices of the frames that are in the main cluster, to remove outlier frames
    main_cluster_indices = np.where(clustering_results == main_cluster)[0]
    return main_cluster_indices