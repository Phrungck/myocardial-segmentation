import cv2 as cv
import numpy as np
import pandas as pd
import scipy.io
import torchvision
import os


class HMCDataset(torchvision.datasets.VisionDataset):

    def __init__(self, root=None, axis='A4C', frame_size=224):

        super().__init__(root)

        self.root = root
        self.axis = axis.upper()
        self.frame_size = frame_size

        if self.root is None:
            self.root = os.getcwd()

        # folder to store trained model weights
        self.output_path = os.path.join(self.root, 'model_weights')

        try:
            os.mkdir(self.output_path)
        except:
            pass

        self.seg_path = os.path.join(
            self.root, 'LV Ground-truth Segmentation Masks')

        self.vid_path = os.path.join(self.root, 'HMC-QU', self.axis)

        # load a4c.xlsx
        # only a4c has ground-truth segmentation masks
        self.df = pd.read_excel(os.path.join(self.root, 'A4C.xlsx'))

        # array containing a4c files with available masks, and indices of frames
        self.sub_df = self.df.loc[self.df.iloc[:, -1]
                                  == 'ü'].iloc[:, [0, -3, -2, -1]].to_numpy()  # shape(109,4)

        # list of a4c files with available segmentation masks
        self.a4c_fn = list(self.df.loc[self.df.iloc[:, -1] == 'ü']['ECHO'])

    def __getitem__(self, index):
        s_idx, e_idx = self.sub_df[index, 1:3]
        s_idx = s_idx - 1  # subtract 1 to start from 0

        fn = self.a4c_fn[index]  # filename of raw video file

        v_fn_pth = os.path.join(self.vid_path, fn + '.avi')
        video = self.readVid(v_fn_pth, s_idx, e_idx, self.frame_size)

        s_fn_pth = os.path.join(self.seg_path, 'Mask_' + fn + '.mat')
        seg_frames = self.readMat(s_fn_pth, self.frame_size)

        return video, seg_frames

    def __len__(self):

        return len(self.a4c_fn)

    def readVid(self, filename, start_idx, end_idx, frame_size):

        frame_arr = []
        vid = cv.VideoCapture(filename)
        frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
        frame_w = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_h = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

        do_resize = False

        if frame_w != frame_size or frame_h != frame_size:
            do_resize = True

        for idx in range(frame_count):
            ret, frame = vid.read()

            if start_idx <= idx < end_idx:

                # convert frame to grayscale
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                if do_resize:
                    frame = cv.resize(frame, dsize=(frame_size, frame_size))

                frame_arr.append(frame)

        video = np.array(frame_arr, dtype=np.float32)
        video = np.expand_dims(video, -1)  # (F,H,W,C)

        return video.transpose(3, 0, 1, 2)  # (C,F,H,W)

    def readMat(self, filename, frame_size):

        mat = scipy.io.loadmat(filename)['predicted']

        f, h, w = mat.shape

        if h != frame_size or w != frame_size:

            frame_arr = []

            for i in range(f):

                frame_arr.append(
                    cv.resize(mat[i], dsize=(frame_size, frame_size)))

            mat = np.array(frame_arr, dtype=np.float32)

        return mat
