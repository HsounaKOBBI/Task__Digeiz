import json
from math import isnan
from multiprocessing import Pool
from itertools import combinations
import cv2
import numpy as np

def compute_distance(n1, n2):

    imagePath1 = "anomalie_detection/BrutForce/ResizedFrame/frame" + str(n1) + ".png"
    imagePath2 =  "anomalie_detection/BrutForce/ResizedFrame/frame" + str(n2) + ".png"

    img1 = cv2.imread(imagePath1, 0)
    img2 = cv2.imread(imagePath2, 0)


    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if len(kp1) < 2 or len(kp2) < 2:
        return {
            "frames": [n1, n2],
            "mean": 0,
            "median": 0,
            "matches": 0,
            "fraction": 0,
            "x": 0,
            "y": 0,
        }
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    distances = [match[0].distance for i, match in enumerate(matches) if matchesMask[i][0] == 1]
    pts1 = [kp1[match[0].queryIdx].pt for i, match in enumerate(matches) if matchesMask[i][0] == 1]
    pts2 = [kp2[match[0].trainIdx].pt for i, match in enumerate(matches) if matchesMask[i][0] == 1]
    direction = np.array([[pt2[0] - pt1[0], pt2[1] - pt1[1]] for pt1, pt2 in zip(pts1, pts2)])
    median_direction = np.median(direction, axis=0)

    print("Finished", n1, n2)
    return {
        "frames": [n1, n2],
        "mean": np.mean(distances),
        "median": np.median(distances),
        "matches": len(distances),
        "fraction": len(distances) / len(matches) if len(matches) > 0 else 0,
        "x": median_direction[0] if len(direction) > 0 else 0,
        "y": median_direction[1] if len(direction) > 0 else 0,
    }

def extract_matching_data(N):
    """  compute_distance on each pair of frame and store the data in matching_data.json. """


    frames = [i for i in range(N)]

    # result = []
    with Pool(processes=4) as pool:
        procs = [pool.apply_async(compute_distance, (frame1, frame2,)) for frame1, frame2 in combinations(frames, 2)]

        result = [res.get() for res in procs]
    print('Done')
    with open('anomalie_detection/BrutForce/matching_data.json', 'w') as f:
        f.write(json.dumps(result, indent=2))
def compute_score(d):
    """ Compute a score for a given pair of frame (the lower the
    closer the frames). """
    median = d['median'] if (not isnan(d['median']) and d['median'] != 0) else 1000
    return 1 / ((d['fraction'] + 0.0001) * (1 + d['matches'])) * median


def clustering_brut_force_matcher(N,invalid_frames=[]):
    """ Cleans frames that shouldn't be include in the video and
    reorder the frames. """
    with open('anomalie_detection/BrutForce/matching_data.json', 'r') as f:
        data = json.load(f)

    frames_index = list(range(1, N))
    # Filter out invalid frames:
    # Invalid frames have a low number of feature matching other images which
    # make the their score go high
    to_plot = np.zeros([N, N])
    for d in data:
        to_plot[d['frames'][0], d['frames'][1]] = compute_score(d)
    size = len(to_plot)
    if len(invalid_frames)==0:
        invalid_frames = []
        for i in range(size):
            if np.median(np.concatenate((to_plot[:i, i], to_plot[i, (i+1):]))) >= 1000:
                invalid_frames.append(i)
        print("Invalid frames:", invalid_frames)
        main_cluster_index = [valeur for valeur in frames_index if valeur not in invalid_frames]
    else :
        main_cluster_index = [valeur for valeur in frames_index if valeur not in invalid_frames]
    clean_data = list(filter(lambda d: d['frames'][0] not in invalid_frames
                                       and d['frames'][1] not in invalid_frames, data))
    return main_cluster_index, clean_data

