import pickle
from typing import List

import answers.boxinfo as BoxInfo

def test_pickle(dataset_root):
    # Load the pickle file
    with open(f'{dataset_root}/annot_all.pkl', 'rb') as f:
        videos_annot = pickle.load(f)
    # Example: Access a specific video's annotations
    videoId = list(videos_annot.keys())[0]
    clipId = list(videos_annot[videoId].keys())[0]
    frameId = list(videos_annot[videoId][clipId]['frame_boxes_dct'].keys())[0]

    # Extract bounding boxes and annotations for a frame
    boxes: List[BoxInfo] = videos_annot[videoId][clipId]['frame_boxes_dct'][frameId]
    for box in boxes:
        print(f'box {box.box} category {box.category} playerId {box.player_ID} frameId {box.frame_ID}')