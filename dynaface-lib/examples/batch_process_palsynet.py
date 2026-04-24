import csv
import logging
import os
import sys
from pathlib import Path

_LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LIB_DIR))

from tqdm import tqdm
import dynaface.facial as facial
import dynaface.measures as measures
from dynaface import models
from dynaface.measures import AnalyzeLandmarks
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATASET_ROOT = "/home/lize/projects/datasets/palsynet-data/frames"
# OUTPUT_ROOT = "csv_data"
CATEGORIES = ["affected", "unaffected"]

def process_subject(subject_path, category, subject_name, analyzer, headers):
    """
    Processes all images for a single subject and saves the results to a CSV file.
    """
    cropped_img_dir = os.path.join(subject_path, "cropped_img")
    
    if not os.path.exists(cropped_img_dir):
        logger.warning(f"Subject {subject_name} ({category}) has no 'cropped_img' directory. Skipping.")
        return

    # Create subject-specific output directory
    # subject_output_dir = os.path.join(OUTPUT_ROOT, subject_name)
    # os.makedirs(subject_output_dir, exist_ok=True)
    
    # csv_file_path = os.path.join(subject_output_dir, f"{subject_name}_measures.csv")
    subject_output_dir = subject_path
    csv_file_path = os.path.join(subject_output_dir, f"{subject_name}_measures.csv")

    # List and sort images
    image_files = [f for f in os.listdir(cropped_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    if not image_files:
        logger.warning(f"No images found for subject {subject_name} in {cropped_img_dir}.")
        return

    results = []
    
    logger.info(f"Processing subject: {subject_name} from {category}")
    for img_name in tqdm(image_files, desc=f"Subject {subject_name}"):
        img_path = os.path.abspath(os.path.join(cropped_img_dir, img_name))
        
        try:
            # Load the image
            # crop=True ensures it uses the StyleGAN crop logic if frontal or lateral crop if lateral
            # success = analyzer.load_image(img_path, crop=True)
            img = cv2.imread(img_path)

            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue

            success = analyzer.load_image(img, crop=False)
            
            if not success or analyzer.is_no_face():
                logger.debug(f"No face detected in {img_path}")
                continue
                
            # Perform analysis
            measures_dict = analyzer.analyze()
            
            if measures_dict:
                row = {"image_path": img_path}
                # We only take the measures that are in our headers
                row.update({k: v for k, v in measures_dict.items() if k in headers})
                results.append(row)
                
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")

    # Write to CSV if we have results
    if results:
        with open(csv_file_path, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Saved {len(results)} measurements to {csv_file_path}")
    else:
        logger.warning(f"No valid measurements found for subject {subject_name}.")

def main():
    # Detect device and initialize models
    device = models.detect_device()
    logger.info(f"Using device: {device}")
    
    # Ensure models are downloaded and initialized
    model_path = models.download_models()
    models.init_models(model_path, device)
    
    # Initialize all measures except raw landmarks
    all_m = measures.all_measures()
    filtered_measures = [m for m in all_m if not isinstance(m, AnalyzeLandmarks)]
    
    # Initialize the AnalyzeFace object once
    analyzer = facial.AnalyzeFace(measures=filtered_measures)
    
    # Get all measure item names for CSV headers
    # headers will be [image_path, fai, oce.l, oce.r, ...]
    measure_headers = analyzer.get_all_items()
    headers = ["image_path"] + sorted(measure_headers)
    
    # Traverse the dataset structure
    for category in CATEGORIES:
        category_dir = os.path.join(DATASET_ROOT, category)
        
        if not os.path.exists(category_dir):
            logger.error(f"Category directory {category_dir} not found. Skipping category.")
            continue
            
        subjects = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
        subjects.sort()
        
        for subject_name in subjects:
            subject_path = os.path.join(category_dir, subject_name)
            process_subject(subject_path, category, subject_name, analyzer, headers)

if __name__ == "__main__":
    main()
