# Document Data Extraction Project

This project extracts structured information from two types of document images:

- Aadhaar cards
- School ID cards

The application provides a simple Streamlit interface where the user can either upload images or capture them with the camera. After processing, the extracted data is shown in a table and can be downloaded as a CSV file.

## Project Objective

The main goal of this project is to convert unstructured document images into structured data that can be reviewed, stored, or exported.

For the current implementation:

- Aadhaar cards return `name`, `dob`, `gender`, and `aadhar_no`
- School ID cards return `name`, `enrollment_no`, `programme`, and `department`

## Tech Stack

- `Streamlit` for the web interface
- `OpenCV` for image decoding and document-region processing
- `EasyOCR` for text extraction
- `NumPy` for image array handling
- `Pandas` for tabular display and CSV export
- `YOLO` model files (`yolov3.cfg` and `yolov3last2.weights`) for Aadhaar field detection

## How The System Works

The overall flow is:

1. The user opens the Streamlit app.
2. The user selects the document type: Aadhaar Card or School ID Card.
3. The app loads the matching extractor only once and reuses it with `st.cache_resource`.
4. The user either uploads an image or captures a new one using the camera.
5. The image is converted into an OpenCV image array.
6. The selected extractor processes the image and returns structured fields.
7. The app displays the results in a table.
8. The results can be downloaded as a CSV file.

## Architecture

### 1. User Interface Layer

The Streamlit UI is implemented in [app.py](/c:/Users/abhin/OneDrive/Documents/Aadhar-Data-Extraction/app.py).

This file is responsible for:

- showing the document type selector
- handling file upload
- handling live camera capture
- calling the correct extractor
- displaying results in a dataframe
- allowing CSV download
- managing temporary camera-session data in `st.session_state`

### 2. Aadhaar Extraction Layer

The Aadhaar pipeline is implemented in [aadhaar_extractor.py](/c:/Users/abhin/OneDrive/Documents/Aadhar-Data-Extraction/aadhaar_extractor.py).

This extractor uses a two-step approach:

1. Detect the important text regions using a YOLO model.
2. Run OCR only on those detected regions.

This is useful for Aadhaar cards because the target fields usually appear in known visual areas, so object detection helps isolate the correct text before OCR.

### 3. School ID Extraction Layer

The School ID pipeline is implemented in [school_id_extractor.py](/c:/Users/abhin/OneDrive/Documents/Aadhar-Data-Extraction/school_id_extractor.py).

This extractor uses:

1. OCR on multiple image variants
2. text normalization
3. regex-based parsing of labeled fields

This approach is used because school ID cards are treated more like label-value documents, where fields such as `Name`, `Enrollment No.`, `Programme`, and `Department` can be extracted from OCR text directly.

## Detailed Working

### Aadhaar Card Pipeline

When the user selects Aadhaar Card, the app creates an `AadhaarExtractor`.

The extractor works like this:

1. It loads the YOLO network using `yolov3last2.weights` and `yolov3.cfg`.
2. It defines four target classes:
   - `name`
   - `dob`
   - `gender`
   - `aadhar_no`
3. The input image is converted into a blob using OpenCV.
4. YOLO predicts candidate bounding boxes for the fields.
5. Non-Max Suppression removes overlapping duplicate boxes.
6. Each remaining box is cropped with a little padding.
7. EasyOCR reads text from each cropped region.
8. Regex rules are applied to improve field classification:
   - date pattern for DOB
   - 12-digit pattern for Aadhaar number
   - `male` or `female` for gender
   - remaining mostly alphabetic text is treated as name
9. Duplicate values are removed.
10. Final output is returned as a dictionary.

Example output:

```python
{
    "name": "Abhinav Kumar Maurya",
    "dob": "10/01/2004",
    "gender": "MALE",
    "aadhar_no": "1234 5678 9012"
}
```

### School ID Card Pipeline

When the user selects School ID Card, the app creates a `SchoolIDExtractor`.

The extractor works like this:

1. It prepares multiple versions of the same image:
   - original image
   - upscaled grayscale image
   - thresholded high-contrast image
2. EasyOCR reads text from each variant.
3. OCR lines are normalized and duplicate lines are removed.
4. The parser tries to extract fields from the combined OCR text.
5. It supports:
   - inline values such as `Name : Abhinav Kumar Maurya`
   - split label/value pairs where the label is on one line and the value is on the next
6. If enrollment number is not found by label, a fallback regex searches for a likely alphanumeric ID.
7. Results from multiple image variants are merged so missing values from one pass can be filled by another.

Example output:

```python
{
    "name": "Abhinav Kumar Maurya",
    "enrollment_no": "20234181",
    "programme": "Bachelor of Technology",
    "department": "Electronics & Comm. Engg"
}
```

## Upload Mode And Camera Mode

The Streamlit app supports two input modes:

- `Upload Images`: users can upload multiple files at once
- `Camera Capture`: users can take photos directly inside the app

For camera capture, the app also:

- stores extracted rows in session state
- prevents duplicate capture entries by hashing image bytes
- allows deleting selected rows
- allows clearing all captured rows
- generates CSV output from the current captured records

## Important Files

- [app.py](/c:/Users/abhin/OneDrive/Documents/Aadhar-Data-Extraction/app.py): main Streamlit application
- [aadhaar_extractor.py](/c:/Users/abhin/OneDrive/Documents/Aadhar-Data-Extraction/aadhaar_extractor.py): Aadhaar extraction logic using YOLO + OCR
- [school_id_extractor.py](/c:/Users/abhin/OneDrive/Documents/Aadhar-Data-Extraction/school_id_extractor.py): School ID extraction logic using OCR + parsing
- [main.py](/c:/Users/abhin/OneDrive/Documents/Aadhar-Data-Extraction/main.py): older standalone Aadhaar prototype script
- [test_extractor.py](/c:/Users/abhin/OneDrive/Documents/Aadhar-Data-Extraction/test_extractor.py): manual Aadhaar smoke-test script
- [test_school_id_extractor.py](/c:/Users/abhin/OneDrive/Documents/Aadhar-Data-Extraction/test_school_id_extractor.py): unit tests for school ID parsing rules
- [requirements.txt](/c:/Users/abhin/OneDrive/Documents/Aadhar-Data-Extraction/requirements.txt): Python dependencies
- [yolov3.cfg](/c:/Users/abhin/OneDrive/Documents/Aadhar-Data-Extraction/yolov3.cfg): YOLO configuration
- `yolov3last2.weights`: trained YOLO weights for Aadhaar field detection

## How To Run

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Start the Streamlit app.

Commands:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## How To Explain This To Your Sir

You can explain the project in this simple way:

"This project takes an image of an Aadhaar card or a school ID card and automatically converts the important information into structured data. The frontend is built in Streamlit. For Aadhaar cards, the system first detects field regions using a YOLO model and then applies OCR on those regions. For school ID cards, the system uses OCR directly on multiple enhanced image versions and then extracts the labeled values using text parsing rules. Finally, the extracted information is shown in a table and can be downloaded as CSV."

## Why Two Different Pipelines Are Used

Two separate extractors are used because the document structures are different.

- Aadhaar cards benefit from region detection, so YOLO helps isolate specific fields before OCR.
- School ID cards behave more like general text documents, so OCR plus regex parsing is simpler and more practical.

## Current Limitations

- Aadhaar accuracy depends on the quality of the trained YOLO weights and the clarity of the image.
- School ID extraction works best when labels are visible and OCR can read them properly.
- Different card designs may require additional parsing rules or retraining.
- The app currently exports CSV but does not save results to a database.

## Possible Future Improvements

- support more document types
- add image rotation and perspective correction
- improve OCR cleanup and post-processing
- store extracted records in a database
- add confidence scores for each extracted field
- retrain or improve the detection model for better Aadhaar accuracy

