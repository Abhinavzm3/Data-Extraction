from aadhaar_extractor import AadhaarExtractor
import cv2
import glob

def test():
    extractor = AadhaarExtractor()
    images = glob.glob("sample_image*.png") + glob.glob("sample_image*.jpeg")
    
    print(f"Found {len(images)} images")
    
    for img_path in images:
        print(f"Processing {img_path}...")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
            
        data = extractor.extract(img)
        print(f"Result for {img_path}:")
        print(data)
        print("-" * 20)

if __name__ == "__main__":
    test()
