import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def read_foreground_map(map_file):
    mask = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Foreground mask file {map_file} not found.")
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def read_cell_locations(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return [tuple(coords) for coords in data]

def marking_function(img, x, y, threshold):
    return img[y, x] > threshold

def region_growing(img, seeds, mask, threshold=200):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    height, width = img.shape
    segmentation = np.zeros_like(img, dtype=np.int32)
    
    for label, seed in enumerate(seeds, start=1):
        x, y = seed
        if not (0 <= x < width and 0 <= y < height) or mask[y, x] == 0:
            continue
        stack = [(x, y)]
        segmentation[y, x] = label
        
        while stack:
            cx, cy = stack.pop()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and segmentation[ny, nx] == 0 and mask[ny, nx] == 255:
                    if marking_function(img, nx, ny, threshold):
                        segmentation[ny, nx] = label
                        stack.append((nx, ny))

    return segmentation
def label_to_color(segmentation):
   
    unique_labels = np.unique(segmentation)[1:]  
    label_colors = np.random.randint(0, 256, size=(unique_labels.max()+1, 3), dtype=np.uint8)
    label_colors[0] = [0, 0, 0]  

    colored_segmentation = label_colors[segmentation]
    return colored_segmentation


def calculate_metrics(segmentation, gold_standard_path, thresholds=[0.5, 0.75, 0.9]):
    gold_standard = np.loadtxt(gold_standard_path, dtype=np.int32)
 

    ids1 = np.unique(segmentation)[1:] 
    ids2 = np.unique(gold_standard)[1:]  
    iou_scores = {threshold: [] for threshold in thresholds}
    dice_scores = {threshold: [] for threshold in thresholds}

    for id1 in ids1:
        mask1 = segmentation == id1
        for threshold in thresholds:
            best_iou = 0
            for id2 in ids2:
                mask2 = gold_standard == id2
                intersection = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                iou = intersection / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
            if best_iou > threshold:
                iou_scores[threshold].append(best_iou)
                dice = 2 * intersection / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) > 0 else 0
                dice_scores[threshold].append(dice)

   
    avg_iou_scores = {threshold: np.mean(scores) if scores else 0 for threshold, scores in iou_scores.items()}
    avg_dice_scores = {threshold: np.mean(scores) if scores else 0 for threshold, scores in dice_scores.items()}
    return avg_iou_scores, avg_dice_scores
def calculate_performance(segmentation, gold_standard_path):
    gold_standard = np.loadtxt(gold_standard_path, dtype=np.int32)

    ids1 = np.unique(segmentation)[1:]  

    ids2 = np.unique(gold_standard)[1:]  

    TP = 0
    for id1 in ids1:
        mask1 = segmentation == id1
        overlaps = [np.logical_and(mask1, gold_standard == id2).sum() for id2 in ids2]
        max_overlap = max(overlaps) if overlaps else 0
        if max_overlap > 0:  
            TP += 1

    FP = len(ids1) - TP  
    FN = len(ids2) - TP  

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f_score

def process_images(images, masks, cell_locations):
    results = []
    for image_file, mask_file, location_file in zip(images, masks, cell_locations):
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        mask = read_foreground_map(mask_file)
        seeds = read_cell_locations(location_file)

        segmentation = region_growing(img, seeds, mask)
        colored_segmentation = label_to_color(segmentation)

        plt.figure(figsize=(10, 5))
        plt.imshow(colored_segmentation, cmap='prism')
        plt.axis('off')
        plt.title(f'Segmentation for {os.path.basename(image_file)}')
        plt.show()

        output_file = image_file.replace('.jpg', '_segmentation_colored.jpg')
        cv2.imwrite(output_file, colored_segmentation)
        print(f"Processed and saved segmentation for {image_file}")

        gold_standard_path = image_file.replace('.jpg', '_gold_cells.txt')
        iou_scores, dice_scores = calculate_metrics(segmentation, gold_standard_path)
        results.append((image_file, iou_scores, dice_scores))
        precision, recall, f_score = calculate_performance(segmentation, gold_standard_path)

        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f_score:.2f}")

    for result in results:
        image_file, iou_scores, dice_scores = result
        print(f"Metrics for {image_file}:")
        for threshold in iou_scores.keys():
            print(f"At threshold {threshold}: IOU = {iou_scores[threshold]:.2f}, Dice = {dice_scores[threshold]:.2f}")

if __name__ == "__main__":
    images = ['data/im1.jpg', 'data/im2.jpg', 'data/im3.jpg']
    masks = ['output/output_mask1.png', 'output/output_mask2.png', 'output/output_mask3.png']
    cell_locations = ['cell_locations_im1.json', 'cell_locations_im2.json', 'cell_locations_im3.json']
    process_images(images, masks, cell_locations)
