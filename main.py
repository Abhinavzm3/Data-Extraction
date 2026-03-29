import cv2
import numpy as np
import glob
import random
import easyocr
import os
reader = easyocr.Reader(['en'])

# Load Yolo
net = cv2.dnn.readNet("yolov3last2.weights", "yolov3.cfg")

# Name custom object
classes = ["name",'dob','gender','aadhar_no']

# Images path
images_path = glob.glob('sample_image2.png')


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
random.shuffle(images_path)

# loop through all the images
for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    # img = cv2.resize(img, None, fx=0.4, fy=0.4) # REMOVED RESIZING FOR BETTER OCR
    height, width, channels = img.shape
    
    # Detecting objects
    # Blob from image - we might need to adjust parameters if detection is poor, but let's try full res first
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.25: # Confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    print(f"Found {len(boxes)} potential boxes for {img_path}")
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    nms_count = len(indexes) if hasattr(indexes, '__len__') else 0
    print(f"NMS kept {nms_count} boxes")
    
    # Handle different OpenCV versions of NMSBoxes return
    if nms_count > 0:
        if isinstance(indexes[0], (list, tuple, np.ndarray)):
            indexes = [i[0] for i in indexes]
    
    # print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    
    # Store results to print them nicely
    extracted_data = {}

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            
            # Add padding to the crop
            padding_x = 5
            padding_y = 5
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = w + (2 * padding_x)
            h = h + (2 * padding_y)
            
            # Ensure crop is within image bounds
            if x + w > width: w = width - x
            if y + h > height: h = height - y
            
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            
            crop = img[y:y+h, x:x+w]
            
            # Only perform OCR if crop is valid
            if crop.size > 0:
                result = reader.readtext(crop, detail=0)
                
                if not result:
                    continue
                
                # Join text and basic cleaning
                text = " ".join(result)
                
                # Heuristics to correct labels and clean data
                import re
                
                # DOB Pattern: DD/MM/YYYY
                dob_match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
                
                # Aadhar Pattern: 12 digits, with or without spaces
                # Looks for 4-4-4 or continuous 12
                aadhar_match = re.search(r'[2-9]\d{3}\s?\d{4}\s?\d{4}', text) # Aadhar usually starts with 2-9
                
                # Gender Pattern
                gender_match = re.search(r'(fe)?male', text, re.IGNORECASE)
                
                if dob_match:
                    label = 'dob'
                    text = dob_match.group(1) # Keep only the date
                elif aadhar_match:
                    label = 'aadhar_no'
                    text = aadhar_match.group(0) # Keep the number
                elif gender_match:
                    label = 'gender'
                    text = gender_match.group(0).upper() # Normalize gender
                else:
                    label = 'name' # Default fallback
                    # Check if it looks like a name (mostly letters)
                    # Reject if it contains too many numbers or is too short
                    digit_count = sum(c.isdigit() for c in text)
                    if digit_count > 3 or len(text) < 3:
                         print(f"Skipping likely garbage: {text}")
                         continue
                
                # Store results - handle duplicates
                if label in extracted_data:
                        extracted_data[label].append(text)
                else:
                        extracted_data[label] = [text]

                # print(f"Detected {label}: {text}")
                
                # Draw box and label
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 5), font, 1, color, 1) # Put text above box

    print("\n--- Extracted Information ---")
    for key, value in extracted_data.items():
        # Clean list to unique values
        unique_values = list(set(value))
        print(f"{key}: {unique_values}")
    print("-----------------------------")

  
    # cv2.imshow("Image", img)
    # key = cv2.waitKey(0)

cv2.destroyAllWindows()