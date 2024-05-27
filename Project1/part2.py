import cv2
import numpy as np
import os
from skimage.feature import peak_local_max
import json 

def read_gold_standard(gold_path):
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold standard file at {gold_path} not found")
    return np.loadtxt(gold_path, dtype=int)

def detect_cell_boundaries(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def calculate_distance_transform(foreground_mask, edges):
    combined = np.bitwise_or(foreground_mask, edges)
    dist_transform = cv2.distanceTransform(255 - combined, cv2.DIST_L2, 3)
    return dist_transform

def find_cell_locations(dist_transform):
    coordinates = peak_local_max(dist_transform, min_distance=20, threshold_abs=2, indices=True)
    return coordinates

def calculate_metrics(detected_labels, gold_labels):
    unique_gold_labels = np.unique(gold_labels[gold_labels > 0])
    unique_detected_labels = np.unique(detected_labels[detected_labels > 0])
    TP = sum(label in unique_detected_labels for label in unique_gold_labels)
    FP = len(unique_detected_labels) - TP
    FN = len(unique_gold_labels) - TP
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f_score

def process_image(image_path, mask_path, gold_standard_path):
    rgb_image = cv2.imread(image_path)
    foreground_mask = cv2.imread(mask_path, 0)
    gold_mask = read_gold_standard(gold_standard_path)

    if rgb_image is None or foreground_mask is None or gold_mask is None:
        print(f"Error loading image or mask from {image_path} or {mask_path}")
        return

    edges = detect_cell_boundaries(rgb_image)
    dist_transform = calculate_distance_transform(foreground_mask, edges)
    cell_locations = find_cell_locations(dist_transform)

    detected_cells = np.zeros_like(gold_mask)
    for idx, point in enumerate(cell_locations):
        detected_cells[point[0], point[1]] = idx + 1  

    precision, recall, f_score = calculate_metrics(detected_cells, gold_mask)

    result_image = rgb_image.copy()
    for point in cell_locations:
        cv2.circle(result_image, tuple(point[::-1]), 3, (0, 255, 0), -1)

    output_visualization = f"output/cell_locations_{os.path.basename(image_path)}"
    output_distance_transform = f"output/dist_transform_{os.path.basename(image_path)}"
    cv2.imwrite(output_visualization, result_image)
    cv2.imwrite(output_distance_transform, np.uint8(dist_transform / dist_transform.max() * 255))

    cell_location_file = f'output/cell_locations_{os.path.basename(image_path).replace(".jpg", "")}.json' #saved cell locations in json format 
    with open(cell_location_file, 'w') as f:
        json.dump(cell_locations.tolist(), f)

    print(f"Processed {image_path}. Results saved to {output_visualization} and {output_distance_transform}")
    print(f"{os.path.basename(image_path)} - Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")

if __name__ == "__main__":
    images = ["im1.jpg", "im2.jpg", "im3.jpg"]
    masks = ["output_mask1.png", "output_mask2.png", "output_mask3.png"]
    gold_standards = ["im1_gold_cells.txt", "im2_gold_cells.txt", "im3_gold_cells.txt"]

    for img, mask, gold in zip(images, masks, gold_standards):
        image_path = f"data/{img}"
        mask_path = f"output/{mask}"
        gold_path = f"data/{gold}"
        process_image(image_path, mask_path, gold_path)
