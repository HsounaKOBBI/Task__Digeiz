from sklearn.metrics.pairwise import cosine_similarity



from parameters import cfg

def sort_frames(embeddings_dict, reverse_order=False):
    sorted_frames_indices = []
    unsorted_frames_indices = list(embeddings_dict.keys())
    # we place one frame as a starting point
    sorted_frames_indices.append(unsorted_frames_indices.pop(0))
    while (len(unsorted_frames_indices) > 0):
        #find the frame with the highest similarity to the last frame in the sorted list
        max_similarity = -1
        max_similarity_index = -1
        insert_at_end = False
        for idx in unsorted_frames_indices:
            for j in [0, -1]:
                similarity = cosine_similarity([embeddings_dict[sorted_frames_indices[j]]], [embeddings_dict[idx]])[0][0]
                if(similarity > max_similarity):
                    max_similarity = similarity
                    max_similarity_index = idx
                    if(j == 0):
                        insert_at_end = False
                    else:
                        insert_at_end = True
        if(insert_at_end):
            sorted_frames_indices.append(max_similarity_index)
        else:
            sorted_frames_indices.insert(0, max_similarity_index)

        unsorted_frames_indices.remove(max_similarity_index)
    if(reverse_order):
        sorted_frames_indices.reverse()
    return sorted_frames_indices
