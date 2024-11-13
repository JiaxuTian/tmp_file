import cv2
import torch

# Define HueTemplates and constants
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


# Helper function to calculate angular distance between two angles
def deg_distance(a, b):
    d1 = torch.abs(a - b)
    d2 = torch.abs(360 - d1)
    return torch.min(d1, d2)


# Function to check if a hue is within a sector
def is_in_sector(H, center, width):
    return deg_distance(H, center) < width / 2


# Function to calculate distance to the nearest border of the hue sector
def distance_to_border(H, center, width):
    border_left = center - width / 2
    border_right = center + width / 2
    H_left = deg_distance(H, border_left)
    H_right = deg_distance(H, border_right)
    return torch.min(H_left, H_right)


# Function to initialize sectors based on a given template
def reset_sectors(template_type, alpha):
    sectors = []
    for t in HueTemplates[template_type]:
        center = t[0] * 360 + alpha
        width = t[1] * 360
        sectors.append((center, width))
    return sectors


# Function to calculate harmony score for a given HSV image and a set of sectors
def harmony_score(X, sectors):
    H = X[:, :, 0].to(torch.int32) * 2  # Convert to degrees
    S = X[:, :, 1].to(torch.float32) / 255.0  # Normalize saturation
    H_dis = hue_distance(H, sectors)
    H_dis = torch.deg2rad(H_dis)
    return torch.sum(H_dis * S)


# Function to calculate hue distance for each sector
def hue_distance(H, sectors):
    H_dis = []
    for center, width in sectors:
        H_d = distance_to_border(H, center, width)
        H_d[is_in_sector(H, center, width)] = 0  # Set distance to 0 if within sector
        H_dis.append(H_d)
    H_dis = torch.stack(H_dis)
    return torch.min(H_dis, axis=0).values


# Function to compute the best harmony score for an image
def B(X):
    F_matrix = torch.zeros((M, A), device='cuda')
    X = torch.tensor(X, device='cuda')
    for i, m in enumerate(template_types):
        for j in range(A):
            alpha = 360 / A * j
            sectors = reset_sectors(m, alpha)
            F_matrix[i, j] = harmony_score(X, sectors)

    best_m_idx, best_alpha = torch.unravel_index(torch.argmin(F_matrix), F_matrix.shape)
    return F_matrix[best_m_idx, best_alpha].item()


# Load and convert image to HSV
image_filename = r"path/to/image.png"
color_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
HSV_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

# Calculate harmony score
score = B(HSV_image)

print(score)
