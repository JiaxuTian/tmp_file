import cv2
import numpy as np

HueTemplates = {
    "i": [(0.00, 0.05)],
    "V": [(0.00, 0.26)],
    "L": [(0.00, 0.05), (0.25, 0.22)],
    "mirror_L": [(0.00, 0.05), (-0.25, 0.22)],
    "I": [(0.00, 0.05), (0.50, 0.05)],
    "T": [(0.25, 0.50)],
    "Y": [(0.00, 0.26), (0.50, 0.05)],
    "X": [(0.00, 0.26), (0.50, 0.26)],
}
template_types = list(HueTemplates.keys())
M = len(template_types)
A = 360


def deg_distance(a, b):
    d1 = np.abs(a - b)
    d2 = np.abs(360 - d1)
    d = np.minimum(d1, d2)
    return d


class HueSector:

    def __init__(self, center, width):
        # In Degree [0,2 pi)
        self.center = center
        self.width = width
        self.border = [(self.center - self.width / 2), (self.center + self.width / 2)]

    def is_in_sector(self, H):
        # True/False matrix if hue resides in the sector
        return deg_distance(H, self.center) < self.width / 2

    def distance_to_border(self, H):
        H_1 = deg_distance(H, self.border[0])
        H_2 = deg_distance(H, self.border[1])
        H_dist2bdr = np.minimum(H_1, H_2)
        return H_dist2bdr


class HarmonicScheme:

    def __init__(self, m, alpha):
        self.m = m
        self.alpha = alpha
        self.reset_sectors()

    def reset_sectors(self):
        self.sectors = []
        for t in HueTemplates[self.m]:
            center = t[0] * 360 + self.alpha
            width = t[1] * 360
            sector = HueSector(center, width)
            self.sectors.append(sector)

    def harmony_score(self, X):
        # Opencv store H as [0, 180) --> [0, 360)
        H = X[:, :, 0].astype(np.int32) * 2
        # Opencv store S as [0, 255] --> [0, 1]
        S = X[:, :, 1].astype(np.float32) / 255.0

        H_dis = self.hue_distance(H)
        H_dis = np.deg2rad(H_dis)

        return np.sum(np.multiply(H_dis, S))

    def hue_distance(self, H):
        H_dis = []
        for i in range(len(self.sectors)):
            sector = self.sectors[i]
            H_dis.append(sector.distance_to_border(H))
            H_dis[i][sector.is_in_sector(H)] = 0
        H_dis = np.asarray(H_dis)
        H_dis = H_dis.min(axis=0)
        return H_dis


def B(X):
    F_matrix = np.zeros((M, A))
    for i in range(M):
        m = template_types[i]
        for j in range(A):
            alpha = 360 / A * j
            harmomic_scheme = HarmonicScheme(m, alpha)
            F_matrix[i, j] = harmomic_scheme.harmony_score(X)
    np.set_printoptions(threshold=np.inf)

    (best_m_idx, best_alpha) = np.unravel_index(np.argmin(F_matrix), F_matrix.shape)
    return F_matrix[best_m_idx][best_alpha]


image_filename = r"C:\Users\86151\Desktop\2.png"
color_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
HSV_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

score = B(HSV_image)
print(score)
