import os
import pickle
from typing import List

from answers.boxinfo import BoxInfo
from answers.volleyball_annot_loader import working_dir, dataset_root

# dataset_root = '/Users/ahmedabbas/Documents/deep-learning/vollyball_project/volleyball-dataset'

def test_pkl_version():
    with open(f'{working_dir}/annot_all.pkl', 'rb') as file:
        videos_annot = pickle.load(file)

    boxes: List[BoxInfo] = videos_annot['0']['13456']['frame_boxes_dct'][13454]
    for box_info in boxes:
        print(box_info.category, box_info.box, box_info.player_ID, box_info.frame_ID)
    print(boxes[0].category)
    print(boxes[0].box)

# load only the main/middle frame
# please note that in this approach we don't need player positions but the whole frame
def load_volleyball_middle_frame_dataset(videos_root):
    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    videos_annot = {}
    index = 1
    # Iterate on each video and for each video iterate on each clip
    for vIndex , video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{vIndex}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        video_annot = os.path.join(video_dir_path, 'annotations.txt')
        clip_category_dct = load_video_annotـmiddle_frame_only(video_annot)

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        for clip_dir in clips_dir:

            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue
            print(f'\t{clip_dir_path}')
            clip_file_name = clip_dir + '.jpg'
            assert clip_file_name in clip_category_dct
            # /Users/ahmedabbas/Documents/deep-learning/vollyball_project/volleyball-dataset/videos/52/10975/10975.jpg
            # root_path = /Users/ahmedabbas/Documents/deep-learning/vollyball_project/volleyball-dataset/
            # videos
            # video dir
            # clip dir
            # middle frame file name
            videos_root = f'{dataset_root}/videos'
            img_path = f'{videos_root}/{video_dir}/{clip_dir}/{clip_file_name}'
            videos_annot[index] = {
                'label': clip_category_dct[clip_file_name],
                'img_path': img_path
            }
            index+=1

    return videos_annot

def load_video_annotـmiddle_frame_only(video_annot):
    with open(video_annot, 'r') as file:
        clip_category_dct = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0]
            clip_category_dct[clip_dir] = items[1]
        return clip_category_dct

def create_pkl_middle_frame_version():
    # You can use this function to create and save pkl version of the dataset
    videos_root = f'{dataset_root}/videos'

    videos_annot = load_volleyball_middle_frame_dataset(videos_root)

    with open(f'{dataset_root}/annot_middle_frame.pkl', 'wb') as file:
        pickle.dump(videos_annot, file)

def test_pkl_middle_frame_version():
    with open(f'{dataset_root}/annot_middle_frame.pkl', 'rb') as file:
        videos_annot = pickle.load(file)

    for idx in videos_annot:
        clip = videos_annot[idx]
        print(f'{idx}--{clip}')
    # clip = videos_annot['13456']
    # print(clip['category'])
    # print(clip['file_name'])
    # print(clip['video'])

if __name__ == '__main__':
    # annot_file = f'{dataset_root}/annotations/4/24855/24855.txt'
    # clip_dir_path = os.path.dirname(annot_file).replace('annotations', 'videos')
    # vis_clip(annot_file, clip_dir_path)
    # create_pkl_version()
    # test_pkl_version()
    # videos_root = f'{dataset_root}/videos'
    # videos_annot = load_volleyball_middle_frame_dataset(videos_root)
    # create_pkl_middle_frame_version()
    create_pkl_middle_frame_version()