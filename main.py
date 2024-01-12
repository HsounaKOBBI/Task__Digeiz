import json
import os
import sys
from pathlib import Path
import argparse
import cv2

from collections import OrderedDict

from Reorder_frames.brutforce_reorder import sort_frames_brutForce
from Reorder_frames.resnet_reorder import sort_frames
from anomalie_detection.BrutForce.brut_force import extract_matching_data, clustering_brut_force_matcher
from anomalie_detection.IsolationForest.isolation_forest_anmoalies import detection_anomalies
from anomalie_detection.ResNetClustering.resnet_clustering import get_resnet_embeddings_of_video,  \
    clustering_resnet

os.chdir(sys.path[0])

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Reorder the mixed frames of a video given, and remove noisy frames"
    )

    parser.add_argument("--video_path", type=Path, help="path to video", required=True)

    parser.add_argument("--anomalie_detection", type=str, help="anomalie detection approach ; choose between ResNetClustering , BrutForce, IsolationForest ", default="ResNetClustering")

    parser.add_argument("--reorder_frames", type=str, help="reordrer frames approach ; choose between ResNet, BrutForce", default="ResNet")

    parser.add_argument("--reverse-order", "-r", action='store_true', help="if this arg is present, the frames will be ordered in reverse order")

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    return parser


def split_video_into_frames(video_path):
    print("****** Start split frames ******")
    cap = cv2.VideoCapture(video_path)
    frames = []
    # Extract frames
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    print(str(len(frames)) + " frames")

    for j, i in enumerate(frames):
        cv2.imwrite("output/frames_extracted/" + str(j + 1) + ".jpg", i)
    print("****** end split frames ******")
    return frames
def save_normal_frames(frames,index):
    for j, i in enumerate(frames):
        if (j in index):
            cv2.imwrite("output/normal_frame/" + str(j + 1) + ".jpg", i)
    return index

def resize_image(image, max_dimension_cap=500):
    img_height, img_width, _ = image.shape
    largest_dimension = max(img_height, img_width)
    resized_img = image
    if(largest_dimension > max_dimension_cap):
        resized_img = cv2.resize(image, (int(img_width * max_dimension_cap / largest_dimension), int(img_height * max_dimension_cap / largest_dimension)))
    return resized_img
def recreate_video(video_path, sorted_frames_idx, reversed_order):
    # extract frames from video
    frames = []
    vidcap = cv2.VideoCapture(str(video_path))
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)

    success, frame = vidcap.read()
    if (not success):
        raise ValueError("Video not found")
    while (success):
        # resize the images
        frame = resize_image(frame, max_dimension_cap=1000)
        frames.append(frame)
        success, frame = vidcap.read()
    vidcap.release()
    # recreate the video in different order
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recreated_video_filename = 'original_video_recreated'
    if (reversed_order):
        recreated_video_filename += "_reversed"
    recreated_video_filename += ".mp4"
    out = cv2.VideoWriter("output/"+recreated_video_filename, fourcc, original_fps, (frames[0].shape[1], frames[0].shape[0]))
    for i in sorted_frames_idx:
        out.write(frames[i])
    out.release()
    print(f"Video recreated and saved as {recreated_video_filename}")
    pass


def main(args):
    video_path=str(args.video_path)
    frames = split_video_into_frames(video_path)
    print(args.anomalie_detection)
    print(args.reorder_frames)
    if args.anomalie_detection == "ResNetClustering":
        embeddings = get_resnet_embeddings_of_video(frames)
        # should be (num_frames, 512)
        print(f"Embeddings shape: {embeddings.shape}")
        # convert to numpy to perform clustering
        embeddings = embeddings.cpu().numpy()
        # get the indices of the frames that are in the main cluster, to remove outlier frames
        main_cluster_index = clustering_resnet(embeddings)
        print(main_cluster_index)
        main_cluster_index=save_normal_frames(frames,main_cluster_index)
        # orderedDict with key = frame index, value = embedding
        embeddings_dict = OrderedDict()
        for i in main_cluster_index:
            embeddings_dict[i] = embeddings[i]
        print("****** normal frame index ********" )
        print(main_cluster_index)


    elif args.anomalie_detection == "BrutForce":
        print(1)
        resized_frames=[]
        for i,j in enumerate(frames):
            resized_frame = cv2.resize(j, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            cv2.imwrite("anomalie_detection/BrutForce/ResizedFrame/frame"+str(i)+".png", resized_frame)
            resized_frames.append(resized_frame)
        extract_matching_data(len(resized_frames))
        main_cluster_index,clean_data= clustering_brut_force_matcher(len(resized_frames))
        print("****** normal frame index ********" )
        print(main_cluster_index)
        main_cluster_index = save_normal_frames(frames, main_cluster_index)

    elif args.anomalie_detection == "IsolationForest":
        main_cluster_index=detection_anomalies(frames)
        print("****** normal frame index ********" )
        print(main_cluster_index)
        main_cluster_index = save_normal_frames(frames, main_cluster_index)


    if args.reorder_frames == "BrutForce":
        if args.anomalie_detection == "BrutForce":
            sorted_frames_idx=sort_frames_brutForce(clean_data,main_cluster_index)
            print("****** sorted frame index ********")
            print(sorted_frames_idx)
        else :
            resized_frames = []
            for i, j in enumerate(frames):
                resized_frame = cv2.resize(j, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                cv2.imwrite("anomalie_detection/BrutForce/ResizedFrame/frame" + str(i) + ".png", resized_frame)
                resized_frames.append(resized_frame)
            # extract_matching_data(len(resized_frames))
            with open('anomalie_detection/BrutForce/matching_data.json', 'r') as f:
                data = json.load(f)
            invalid_frames=  [valeur for valeur in list(range(1, len(resized_frames))) if valeur not in main_cluster_index]
            _, clean_data = clustering_brut_force_matcher(len(resized_frames),invalid_frames)
            sorted_frames_idx = sort_frames_brutForce(clean_data, main_cluster_index)
            print("****** sorted frame index ********")
            print(sorted_frames_idx)



    elif args.reorder_frames=="ResNet":
        print(2)
        if  args.anomalie_detection == "ResNetClustering":
            sorted_frames_idx = sort_frames(embeddings_dict, args.reverse_order)
            print("****** sorted frame index ********")
            print(sorted_frames_idx)
        else :
            embeddings = get_resnet_embeddings_of_video(frames)
            embeddings = embeddings.cpu().numpy()
            embeddings_dict = OrderedDict()
            for i in main_cluster_index:
                embeddings_dict[i] = embeddings[i]
            sorted_frames_idx = sort_frames(embeddings_dict, args.reverse_order)
            print("****** sorted frame index ********")
            print(sorted_frames_idx)
    recreate_video(args.video_path, sorted_frames_idx, args.reverse_order)







if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    main(args)