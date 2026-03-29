import cv2
import numpy as np
import easyocr
import re

class AadhaarExtractor:
    def __init__(self, yolo_weights="yolov3last2.weights", yolo_cfg="yolov3.cfg"):
        self.classes = ["name", 'dob', 'gender', 'aadhar_no']
        self.net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        # Initialize EasyOCR once
        self.reader = easyocr.Reader(['en'])

    def extract(self, img):
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.25:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        extracted_data = {
            'name': [],
            'dob': [],
            'gender': [],
            'aadhar_no': []
        }
        
        # Handle different OpenCV versions of NMSBoxes return
        nms_indexes = []
        if len(indexes) > 0:
            if isinstance(indexes[0], (list, tuple, np.ndarray)):
               nms_indexes = [i[0] for i in indexes]
            else:
               nms_indexes = indexes

        for i in range(len(boxes)):
            if i in nms_indexes:
                x, y, w, h = boxes[i]
                
                # Padding
                padding_x = 5
                padding_y = 5
                x = max(0, x - padding_x)
                y = max(0, y - padding_y)
                w = w + (2 * padding_x)
                h = h + (2 * padding_y)

                if x + w > width: w = width - x
                if y + h > height: h = height - y

                crop = img[y:y+h, x:x+w]

                if crop.size > 0:
                    result = self.reader.readtext(crop, detail=0)
                    if not result:
                        continue
                        
                    text = " ".join(result)
                    
                    # Heuristics
                    dob_match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
                    aadhar_match = re.search(r'[2-9]\d{3}\s?\d{4}\s?\d{4}', text)
                    gender_match = re.search(r'(fe)?male', text, re.IGNORECASE)
                    
                    label = 'name' # Default
                    final_text = text

                    if dob_match:
                        label = 'dob'
                        final_text = dob_match.group(1)
                    elif aadhar_match:
                        label = 'aadhar_no'
                        final_text = aadhar_match.group(0)
                    elif gender_match:
                        label = 'gender'
                        final_text = gender_match.group(0).upper()
                    else:
                        label = 'name'
                        # Garbage filter
                        digit_count = sum(c.isdigit() for c in text)
                        if digit_count > 3 or len(text) < 3:
                             continue
                    
                    if label in extracted_data:
                        extracted_data[label].append(final_text)

        # Clean and Flatten
        final_result = {}
        for key, value in extracted_data.items():
            unique_values = list(set(value))
            # Join multiple detections with a semicolon, or empty string if None
            final_result[key] = "; ".join(unique_values) if unique_values else ""
            
        return final_result
