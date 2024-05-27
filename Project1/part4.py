import cv2
import numpy as np
from skimage import filters, morphology


def preprocess_image(image_path):
    image = cv2.imread(image_path, 0)
    if image is None:
        raise FileNotFoundError(f"Could not find or open {image_path}")
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image


def segment_vessels(image):
    
    blur = cv2.GaussianBlur(image, (9, 9), 0)
    median = cv2.medianBlur(blur, 7)
    
   
    sobelx = cv2.Sobel(median, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(median, cv2.CV_64F, 0, 1, ksize=5)
    sobel_mag = np.hypot(sobelx, sobely)

   
    vessel_mask = filters.threshold_local(sobel_mag, block_size=99) < sobel_mag

 
    cleaned_mask = morphology.remove_small_objects(vessel_mask.astype(bool), min_size=150)
    cleaned_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=150)
    
    return cleaned_mask.astype(np.uint8)


def calculate_metrics(estimated_mask, gold_mask):
    estimated_mask = (estimated_mask > 0).astype(np.uint8)
    gold_mask = (gold_mask > 0).astype(np.uint8)

    tp = np.sum((estimated_mask == 1) & (gold_mask == 1))
    fp = np.sum((estimated_mask == 1) & (gold_mask == 0))
    fn = np.sum((estimated_mask == 0) & (gold_mask == 1))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f_score


def process_images(image_paths, gold_paths):
    for img_path, gold_path in zip(image_paths, gold_paths):
        image = preprocess_image(img_path)
        estimated_mask = segment_vessels(image)
        
        gold_mask = cv2.imread(gold_path, cv2.IMREAD_UNCHANGED)
        if gold_mask is None:
            raise FileNotFoundError(f"Could not find or open {gold_path}")
      
        gold_mask = (gold_mask > 0).astype(np.uint8)

     
        print(f"Unique gold mask values: {np.unique(gold_mask)}")
        print(f"Gold mask dimensions: {gold_mask.shape}")
        print(f"Estimated mask dimensions: {estimated_mask.shape}")

      
        if gold_mask.shape != estimated_mask.shape:
            print("Error: Dimension mismatch between estimated mask and gold mask.")
            continue

        precision, recall, f_score = calculate_metrics(estimated_mask, gold_mask)
        print(f"{img_path}: Precision={precision:.4f}, Recall={recall:.4f}, F-Score={f_score:.4f}")
        
      
        segmented_path = img_path.replace('.jpg', '_segmented.png')
        cv2.imwrite(segmented_path, estimated_mask * 255)

image_paths = ['d4_h.jpg', 'd7_dr.jpg', 'd11_g.jpg']
gold_paths = ['d4_h_gold.png', 'd7_dr_gold.png', 'd11_g_gold.png']

process_images(image_paths, gold_paths)
