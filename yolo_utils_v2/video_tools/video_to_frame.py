import cv2
import os
from yolo_utils_v2.data_loader.video_loader import VideoLoader
from yolo_utils_v2.data_writer.image_writer import ImageWriter
from tqdm import tqdm
#
# class VideoToFrame:
#     def __init__(self, time_tag_list, video_length, filename):
#         # self.time_tag_list = [(337, 378), (422, 429), (440, 449), (483, 490), (898, 900)]
#         self.time_tag_list = time_tag_list
#         self.video_length = video_length
#         self.filename = filename
#         # self.time_tag_list = [(152, 160), (196, 203)]
#         self.fps = 0
#         self.frame_count = 0
#
#         # self.base_video_folder = '/media/taka/dataset/label_preprocess_data/videos'
#         # self.output_folder = '/media/taka/dataset/label_preprocess_data/frames'
#         self.base_video_folder = '/media/taka/dataset/浮標與欄油柱/videos'
#         self.output_folder = '/media/taka/dataset/浮標與欄油柱/test_frame'
#         self.index_list = []
#         self.frame_path_list = []
#         self.img_list = []
#
#         self.all_frame_folder = ''
#         self.output_video_folder = ''
#         self.video_path = ''
#         self._folder_init()
#
#
#     def _folder_init(self):
#         video_name = self.filename[:-4]
#         self.video_path = '{}/{}'.format(self.base_video_folder, self.filename)
#
#         self.output_video_folder = '{}/{}'.format(self.output_folder, video_name)
#         self._make_folder(path=self.output_video_folder)
#         self.all_frame_folder = '{}/all'.format(self.output_video_folder)
#         self._make_folder(path=self.all_frame_folder)
#
#
#     def _time_tag_to_frame_index(self):
#         fps = self._get_fps()
#         print('fps = ', fps)
#         for time_tag in self.time_tag_list:
#             start_index = time_tag[0] * fps
#             end_index = time_tag[1] * fps
#             self.index_list.append((int(start_index), int(end_index)))
#
#         print(self.index_list)
#
#     def _get_fps(self):
#         fps = self.frame_count / self.video_length
#         fps = int(fps)
#         return fps
#
#
#     def _load_frame(self):
#         """
#         load frames and save to the output_folder/all
#         :return:
#         """
#         print(self.video_path)
#         vidcap = cv2.VideoCapture(self.video_path)
#         success, image = vidcap.read()
#         count = 0
#         while success:
#             # path = '{}/{}.jpg'.format(self.output_folder, str(count).zfill(6))
#             file_index = str(count).zfill(6)
#             # path = '{}/all/{}.jpg'.format(self.output_folder, str(count).zfill(6))
#             # path = '{}/all/{}.jpg'.format(self.output_folder, file_index)
#             path = '{}/{}.jpg'.format(self.all_frame_folder, file_index)
#
#             # cv2.imwrite("/media/taka/dataset/demo_data/peoples/frame%d.jpg" % count, image)     # save frame as JPEG file
#             cv2.imwrite(path, image)  # save frame as JPEG file
#             self.frame_path_list.append(file_index)
#             # self.img_list.append(image)
#
#             success, image = vidcap.read()
#             # print('Read a new frame: ', success)
#             count += 1
#             print(count)
#         self.frame_count = count
#
#     def _make_folder(self, path):
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#     def _save_frame(self, frame_group, subfolder_count):
#         folder_path = '{}/{}'.format(self.output_video_folder, str(subfolder_count))
#         self._make_folder(path=folder_path)
#         for file_index in frame_group:
#             # print(path, img)
#             # path = '{}/{}/{}'.format(self.output_folder, str(subfolder_count), path)
#             # path = '{}/{}'.format(folder_path, path)
#             # print(path)
#             # img_path = '{}/all/{}.jpg'.format(self.output_folder, file_index)
#             # img_path = '{}/all/{}.jpg'.format(self.output_folder, file_index)
#             img_path = '{}/{}.jpg'.format(self.all_frame_folder, file_index)
#
#             # print('img_path', img_path)
#             output_path = '{}/{}.jpg'.format(folder_path, file_index)
#             # print('output_path', output_path)
#             img = cv2.imread(img_path)
#             cv2.imwrite(output_path, img)
#
#
#     def _split_frame(self):
#         for index, (start_index, end_index) in enumerate(self.index_list):
#             frame_group = self.frame_path_list[start_index:end_index]
#             # img_group = self.img_list[start_index:end_index]
#             self._save_frame(
#                 frame_group=frame_group, subfolder_count=index
#             )
#             # print(frame_group)
#             # print(img_group)
#
#
#
#     def __call__(self):
#         self._load_frame()
#         self._time_tag_to_frame_index()
#


class VideoToFrame:
    """
    video to frame
    """
    def __init__(self, video_path, output_folder):
        """
        :param video_path:
        :param output_folder:
        """
        self.data_loader = VideoLoader(file_path=video_path)
        self.data_writer = ImageWriter(output_folder=output_folder)

    def run(self, frame_interval):
        """
        run
        :return: None
        """
        for i, (frame, filename) in tqdm(enumerate(self.data_loader.frame_iter())):
            if i % frame_interval == 0:
                self.data_writer.write_data(img=frame, filename=filename)


def main():

    v_to_f = VideoToFrame(video_path='', output_folder='')
    # print(v_to_f.__init__.__doc__)
    # print(v_to_f.__dict__)
    # print(v_to_f.__class__)
    # v_to_f()
    xx = VideoToFrame.__dict__
    for k, v in xx.items():
        # print(k)
        qq = getattr(VideoToFrame, k)
        print('qq', qq.__doc__)


if __name__ == '__main__':
    main()