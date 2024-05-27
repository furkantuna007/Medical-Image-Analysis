import cv2
import numpy as np

def obtain_foreground_mask(image_path):
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Using Otsu's thresholding
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"After thresholding, unique values in the mask: {np.unique(mask)}") # for debugging purposes
    return mask

def evaluate_foreground_mask(estimated_mask, gold_standard_mask):
   
    estimated_mask_bool = estimated_mask == 255
    gold_standard_mask_bool = gold_standard_mask > 0
    true_positives = np.sum(np.logical_and(estimated_mask_bool, gold_standard_mask_bool))
    false_positives = np.sum(np.logical_and(estimated_mask_bool, np.logical_not(gold_standard_mask_bool)))
    false_negatives = np.sum(np.logical_and(np.logical_not(estimated_mask_bool), gold_standard_mask_bool))

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f_score

if __name__ == '__main__':
    for i in range(1, 4):
        image_path = f'data/im{i}.jpg'
        gold_mask_path = f'data/im{i}_gold_mask.txt'
        
        try:
            estimated_mask = obtain_foreground_mask(image_path)
            gold_standard_mask = np.loadtxt(gold_mask_path, dtype=np.uint8)
            
            precision, recall, f_score = evaluate_foreground_mask(estimated_mask, gold_standard_mask)
            output_path = f'output/output_mask{i}.png'
            cv2.imwrite(output_path, estimated_mask)
            
          
            saved_image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            print(f"Image {i} - Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")
            print(f"Unique values in the saved mask for Image {i}: {np.unique(saved_image)}")
        except Exception as e:
            print(f"An error occurred processing Image {i}: {e}")
