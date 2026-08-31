import cv2
import numpy as np
import argparse

def create_flow(img1_path, img2_path, img3_path, output_path):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)
    
    if img1 is None or img2 is None or img3 is None:
        print("Error loading images")
        return

    # Resize to same height
    h = 600
    def resize_h(img, target_h):
        r = target_h / img.shape[0]
        dim = (int(img.shape[1] * r), target_h)
        return cv2.resize(img, dim)

    i1 = resize_h(img1, h)
    i2 = resize_h(img2, h)
    i3 = resize_h(img3, h)
    
    # Add Titles
    def add_title(img, text):
        # Add black bar at top
        bar = np.zeros((50, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return np.vstack((bar, img))

    i1 = add_title(i1, "1. Input Satellite")
    i2 = add_title(i2, "2. Extracted Geometry")
    i3 = add_title(i3, "3. Final Classification")

    # Stack
    combined = np.hstack((i1, i2, i3))
    
    cv2.imwrite(output_path, combined)
    print(f"Saved pipeline flow to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i1", required=True)
    parser.add_argument("--i2", required=True)
    parser.add_argument("--i3", required=True)
    parser.add_argument("--output", default="pipeline_flow.jpg")
    args = parser.parse_args()
    
    create_flow(args.i1, args.i2, args.i3, args.output)
