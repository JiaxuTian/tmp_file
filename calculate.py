import time

import cv2
import numpy as np
import torch

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
    d1 = torch.abs(a - b)
    d2 = torch.abs(360 - d1)
    return torch.min(d1, d2)


class HueSector:
    def __init__(self, center, width):
        self.center = center
        self.width = width
        self.border = [(self.center - self.width / 2), (self.center + self.width / 2)]

    def is_in_sector(self, H):
        return deg_distance(H, self.center) < self.width / 2

    def distance_to_border(self, H):
        H_1 = deg_distance(H, self.border[0])
        H_2 = deg_distance(H, self.border[1])
        return torch.min(H_1, H_2)


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
        H = X[:, :, 0].to(torch.int32) * 2
        S = X[:, :, 1].to(torch.float32) / 255.0
        H_dis = self.hue_distance(H)
        H_dis = torch.deg2rad(H_dis)
        return torch.sum(H_dis * S)

    def hue_distance(self, H):
        H_dis = []
        for sector in self.sectors:
            H_d = sector.distance_to_border(H)
            H_d[sector.is_in_sector(H)] = 0
            H_dis.append(H_d)
        H_dis = torch.stack(H_dis)
        return torch.min(H_dis, axis=0).values


def B(X):
    F_matrix = torch.zeros((M, A), device='cuda')
    X = torch.tensor(X, device='cuda')
    for i, m in enumerate(template_types):
        for j in range(A):
            alpha = 360 / A * j
            harmonic_scheme = HarmonicScheme(m, alpha)
            F_matrix[i, j] = harmonic_scheme.harmony_score(X)

    best_m_idx, best_alpha = torch.unravel_index(torch.argmin(F_matrix), F_matrix.shape)
    return F_matrix[best_m_idx, best_alpha].item()


# Load and convert image to HSV
image_filename = r"C:\Users\86151\Desktop\1.png"
color_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
HSV_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

# Calculate harmony score
score = B(HSV_image)

print(score)
