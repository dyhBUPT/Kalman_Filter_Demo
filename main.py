import cv2
import numpy as np
import scipy.linalg
from copy import copy


class KalmanFilter(object):
    """
    Ref:
    [1] Kalman, Rudolph Emil. "A new approach to linear filtering and prediction problems." (1960): 35-45.
    [2] https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py
    """

    def __init__(self, print_values=False):
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        self.print = print_values

    def initiate(self, z):
        """
        Initialization.
        """
        x = np.r_[z, [0, 0]]

        std = [
            2 * self._std_weight_position,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            10 * self._std_weight_velocity,
        ]
        P = np.diag(np.square(std))

        return x, P

    def predict(self, x, P):
        """
        Run Kalman filter prediction step.
        """
        std = [
            self._std_weight_position,
            self._std_weight_position,
            self._std_weight_velocity,
            self._std_weight_velocity,
        ]
        Q = np.diag(np.square(std))

        x = self.F @ x
        P = self.F @ P @ self.F.T + Q

        if self.print:
            print('predicted mean & covariance')
            print(x)
            print(P)

        return x, P

    def project(self, x, P, factor=10):
        """
        Project state distribution to measurement space.
        """
        std = [
            self._std_weight_position * factor,
            self._std_weight_position * factor,
        ]
        R = np.diag(np.square(std))

        Hx = self.H @ x
        HPH = self.H @ P @ self.H.T

        return Hx, HPH + R  # innovation covariance

    def update(self, x, P, z):
        """
        Run Kalman filter correction step.
        """
        Hx, S = self.project(x, P)

        chol_factor, lower = scipy.linalg.cho_factor(
            S, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve(
            (chol_factor, lower), (P @ self.H.T).T,
            check_finite=False).T  # [4,2]

        x = x + K @ (z - Hx)  # [4,]
        P = P - K @ S @ K.T  # [4,4]

        if self.print:
            print('projected mean & covariance')
            print(Hx)
            print(S)
            print('Kalman Gain')
            print(K)
            print('updated mean & covariance')
            print(x)
            print(P)

        return x, P


def get_noisy_observation(x, scale=50):
    x = copy(x)
    x += np.random.randint(low=-scale, high=scale, size=2, dtype=int)
    return x


def demo(path=None):
    gts = np.loadtxt('data/data1.txt', delimiter=',')
    canvas = np.ones([1120, 2240, 3], dtype=np.uint8) * 255
    kf = KalmanFilter(print_values=True)

    if path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename=path, fourcc=fourcc, fps=5, frameSize=(2240, 1120))

    for i, row in enumerate(gts):
        gt = row[2:]

        if i == 0:
            x, P = kf.initiate(gt)
            z = copy(gt)
        else:
            x, P = kf.predict(x, P)
            z = get_noisy_observation(gt)
            x, P = kf.update(x, P, z)

        cv2.circle(canvas, list(map(int, gt)), radius=10, color=(0, 255, 0), thickness=-1)
        cv2.circle(canvas, list(map(int, z)), radius=10, color=(0, 0, 255), thickness=-1)
        cv2.circle(canvas, list(map(int, x[:2])), radius=10, color=(255, 0, 0), thickness=-1)
        cv2.waitKey(5)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', canvas)

        if path is not None:
            out.write(canvas)

    out.release()


if __name__ == '__main__':
    demo()
  
