from pathlib import Path

import cv2
import numpy as np
import imutils
from tqdm import tqdm
from typing import Sequence, Union

from definitions import ROOT_DIR, DATA_DIR
from src.data.data_load import get_image_by_path
from src.data.data_preprocessing import prepare_image


class ImageStitcher:
    def __init__(self,
                 feature_extractor: str = 'sift',
                 feature_matching: str = 'bf',
                 match_ratio: float = 0.75):
        self.feature_extractor = feature_extractor
        if feature_extractor not in ['sift', 'surf', 'orb', 'brisk']:
            raise ValueError("feature_extractor must be 'sift', 'surf', 'orb' or 'brisk'")
        self.descriptor = None
        self._build_descriptor()

        self.feature_matching = feature_matching
        if feature_matching not in ['bf', 'knn']:
            raise ValueError("feature_matching must be 'bf' or 'knn'")
        self.matcher = None
        self._build_matcher()

        self.match_ratio = match_ratio

    def _build_descriptor(self) -> None:
        descriptors = {
            'sift': cv2.SIFT_create,
            'surf': cv2.xfeatures2d.SURF_create,
            'brisk': cv2.BRISK_create,
            'orb': cv2.ORB_create,
        }
        self.descriptor = descriptors[self.feature_extractor]()

    def _build_matcher(self) -> None:
        cross_check = self.feature_matching == 'bf'
        if self.feature_extractor in ['sift', 'surf']:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
        else:  # 'orb' or 'brisk'
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)

    def _get_keypoints_and_features(self, img: np.ndarray):
        kps, features = self.descriptor.detectAndCompute(img, None)
        return kps, features

    def _match_keypoints(self, features_a, features_b):
        matches = []
        if self.feature_matching == 'bf':  # brute force
            best_matches = self.matcher.match(features_a, features_b)

            # Sort the features in order of distance.
            # The points with small distance (more similarity) are ordered first in the vector
            matches = sorted(best_matches, key=lambda x: x.distance)
        elif self.feature_matching == 'knn':
            # compute the raw matches and initialize the list of actual matches
            raw_matches = self.matcher.knnMatch(features_a, features_b, k=2)

            for m, n in raw_matches:
                # ensure the distance is within a certain ratio of each
                # other (i.e. Lowe's ratio test)
                if m.distance < n.distance * self.match_ratio:
                    matches.append(m)
        # print('Match: ', len(matches))
        return matches

    def _get_homography_matrix(self, kps_a, kps_b, matches, threshold):
        # convert the keypoints to numpy arrays
        kps_a = np.float32([kp.pt for kp in kps_a])
        kps_b = np.float32([kp.pt for kp in kps_b])

        # print('Homo:', kps_a.shape, kps_b.shape)

        H_matrix = None
        if len(matches) > 4:
            # construct the two sets of points
            pts_a = np.float32([kps_a[m.queryIdx] for m in matches])
            pts_b = np.float32([kps_b[m.trainIdx] for m in matches])

            # estimate the homography between the sets of points
            (H_matrix, status) = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, threshold)
        return H_matrix

    def stitch(self,
               img_left: np.ndarray,
               img_right: np.ndarray,
               only_x: bool = False) -> np.ndarray:
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        kps_a, features_a = self._get_keypoints_and_features(gray_right)
        kps_b, features_b = self._get_keypoints_and_features(gray_left)

        matches = self._match_keypoints(features_a, features_b)

        H = self._get_homography_matrix(kps_a, kps_b, matches, threshold=4)
        if H is None:
            return img_left

        if only_x:
            x_shift = H[0, 2]
            H = np.zeros_like(H)
            H[0, 0] = H[1, 1] = H[2, 2] = 1.0
            H[0, 2] = x_shift

        width = img_left.shape[1] + img_right.shape[1]
        height = img_left.shape[0] + img_right.shape[0]

        merged = cv2.warpPerspective(img_right, H, (width, height))
        merged[0:img_left.shape[0], 0:img_left.shape[1]] = img_left

        # transform the panorama image to grayscale and threshold it
        merged_gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(merged_gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Finds contours from the binary image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # get the maximum contour area
        c = max(cnts, key=cv2.contourArea)

        # get a bbox from the contour area
        (x, y, w, h) = cv2.boundingRect(c)

        # crop the image to the bbox coordinates
        result = merged[y:y + h, x:x + w]
        return result

    def stitch_images(self,
                      img_list: Sequence[np.ndarray],
                      by_pairs: bool = False,
                      only_x: bool = False):
        assert len(img_list) > 0
        if by_pairs:
            imgs = img_list
            while len(imgs) > 1:
                # print(len(imgs))
                buffer = []
                for i in range(0, len(imgs), 2):
                    if i + 1 < len(imgs):
                        buffer.append(self.stitch(imgs[i], imgs[i + 1], only_x=only_x))
                    else:
                        buffer.append(imgs[i])
                imgs = [x for x in buffer]
            return imgs[0]

        img_left = img_list[0]
        for img in img_list[1:]:
            img_left = self.stitch(img_left, img, only_x=only_x)
        return img_left


if __name__ == '__main__':
    pass
