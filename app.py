import hashlib

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from aadhaar_extractor import AadhaarExtractor
from school_id_extractor import SchoolIDExtractor


AADHAAR_COLUMNS = ["filename", "name", "dob", "gender", "aadhar_no"]
SCHOOL_ID_COLUMNS = ["filename", "name", "enrollment_no", "programme", "department"]

st.set_page_config(page_title="Document Data Extractor", layout="wide")

st.title("Document Data Extractor")
st.markdown("""
Extract data from both **Aadhaar cards** and **school ID cards**.

- Aadhaar card: **Name**, **DOB**, **Gender**, **Aadhaar Number**
- School ID card: **Name**, **Enrollment No**, **Programme**, **Department**
""")
st.markdown(
    """
    <style>
    div[data-testid="stCameraInput"] {
        max-width: 420px;
        margin: 0 auto 1rem auto;
    }

    div[data-testid="stCameraInput"] video {
        max-height: 320px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_aadhaar_extractor():
    return AadhaarExtractor()


@st.cache_resource
def get_school_id_extractor():
    return SchoolIDExtractor()


def order_columns(dataframe, columns):
    ordered_columns = [column for column in columns if column in dataframe.columns]
    remaining_columns = [column for column in dataframe.columns if column not in ordered_columns]
    return dataframe[ordered_columns + remaining_columns]


def get_file_bytes(file_obj):
    if hasattr(file_obj, "getvalue"):
        return file_obj.getvalue()
    return file_obj.read()


def decode_image(file_obj):
    file_bytes = np.asarray(bytearray(get_file_bytes(file_obj)), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)


def init_camera_state(config):
    st.session_state.setdefault(config["camera_records_key"], [])
    st.session_state.setdefault(config["camera_hash_key"], "")
    st.session_state.setdefault(config["camera_counter_key"], 0)


def clear_camera_state(config):
    st.session_state[config["camera_records_key"]] = []
    st.session_state[config["camera_counter_key"]] = 0


def delete_selected_camera_records(config, selected_indices):
    st.session_state[config["camera_records_key"]] = [
        record
        for index, record in enumerate(st.session_state[config["camera_records_key"]])
        if index not in selected_indices
    ]


def append_camera_capture(extractor, camera_image, config):
    image_bytes = get_file_bytes(camera_image)
    image_hash = hashlib.sha1(image_bytes).hexdigest()

    if image_hash == st.session_state[config["camera_hash_key"]]:
        return None, "duplicate"

    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), 1)
    st.session_state[config["camera_hash_key"]] = image_hash

    if img is None:
        return None, "invalid"

    data = extractor.extract(img)
    st.session_state[config["camera_counter_key"]] += 1
    capture_number = st.session_state[config["camera_counter_key"]]
    data["filename"] = f"{config['camera_filename_prefix']}_{capture_number:03d}.jpg"
    st.session_state[config["camera_records_key"]].append(data)
    return data, "added"


def render_upload_tab(extractor, config):
    uploaded_files = st.file_uploader(
        config["upload_label"],
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=config["upload_key"],
    )

    if not uploaded_files:
        return

    st.write(f"Processing {len(uploaded_files)} images...")
    results = []
    progress_bar = st.progress(0)

    for index, uploaded_file in enumerate(uploaded_files):
        img = decode_image(uploaded_file)

        if img is not None:
            data = extractor.extract(img)
            data["filename"] = uploaded_file.name
            results.append(data)
        else:
            st.warning(f"Could not read image: {uploaded_file.name}")

        progress_bar.progress((index + 1) / len(uploaded_files))

    if not results:
        return

    dataframe = order_columns(pd.DataFrame(results), config["columns"])
    st.subheader("Extracted Data")
    st.dataframe(dataframe)

    csv = dataframe.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=config["upload_csv_name"],
        mime="text/csv",
        key=config["upload_csv_key"],
    )


def render_camera_tab(extractor, config):
    init_camera_state(config)

    left_spacer, camera_column, right_spacer = st.columns([1.2, 2, 1.2])

    with camera_column:
        st.markdown(config["camera_help_text"])
        camera_image = st.camera_input(config["camera_label"], key=config["camera_key"])

    if camera_image is not None:
        with st.spinner("Extracting data from captured image..."):
            _, status = append_camera_capture(extractor, camera_image, config)

        if status == "invalid":
            st.error("Could not process the captured image. Please try again.")
        elif status == "added":
            st.success("Captured image added to the table below.")

    records = st.session_state[config["camera_records_key"]]

    if not records:
        st.info("Captured records will appear in the table below and keep appending with each new photo.")
        return

    dataframe = order_columns(pd.DataFrame(records), config["columns"])
    st.subheader(f"Captured Data ({len(dataframe)})")
    editable_dataframe = dataframe.copy()
    editable_dataframe.insert(0, "delete", False)
    editor_key = f"{config['camera_editor_key']}_{len(dataframe)}"
    edited_dataframe = st.data_editor(
        editable_dataframe,
        hide_index=True,
        use_container_width=True,
        disabled=list(editable_dataframe.columns[1:]),
        column_config={
            "delete": st.column_config.CheckboxColumn(
                "Delete",
                help="Select the rows you want to remove from the captured data.",
                default=False,
            )
        },
        key=editor_key,
    )

    csv = dataframe.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Captured Data as CSV",
        data=csv,
        file_name=config["camera_csv_name"],
        mime="text/csv",
        key=config["camera_csv_key"],
    )

    action_column_1, action_column_2 = st.columns(2)

    with action_column_1:
        if st.button("Delete Selected Rows", key=config["delete_camera_key"]):
            selected_indices = edited_dataframe.index[edited_dataframe["delete"]].tolist()
            if not selected_indices:
                st.warning("Select at least one row to delete.")
            else:
                delete_selected_camera_records(config, set(selected_indices))
                st.rerun()

    with action_column_2:
        if st.button("Clear Captured Data", key=config["clear_camera_key"]):
            clear_camera_state(config)
            st.rerun()


document_type = st.radio(
    "Choose document type",
    ["Aadhaar Card", "School ID Card"],
    horizontal=True,
)

if document_type == "Aadhaar Card":
    config = {
        "extractor_getter": get_aadhaar_extractor,
        "columns": AADHAAR_COLUMNS,
        "upload_label": "Upload Aadhaar Images",
        "upload_key": "aadhaar_upload",
        "upload_csv_name": "aadhaar_extracted_data.csv",
        "upload_csv_key": "aadhaar_upload_csv",
        "camera_label": "Capture Aadhaar Card",
        "camera_key": "aadhaar_camera",
        "camera_help_text": "Point your camera at the Aadhaar card and click **Take Photo**.",
        "camera_csv_name": "aadhaar_camera_data.csv",
        "camera_csv_key": "aadhaar_camera_csv",
        "camera_records_key": "aadhaar_camera_records",
        "camera_hash_key": "aadhaar_camera_hash",
        "camera_counter_key": "aadhaar_camera_counter",
        "camera_filename_prefix": "aadhaar_camera_capture",
        "camera_editor_key": "aadhaar_camera_editor",
        "delete_camera_key": "aadhaar_delete_camera_rows",
        "clear_camera_key": "aadhaar_clear_camera_data",
    }
else:
    config = {
        "extractor_getter": get_school_id_extractor,
        "columns": SCHOOL_ID_COLUMNS,
        "upload_label": "Upload School ID Images",
        "upload_key": "school_id_upload",
        "upload_csv_name": "school_id_extracted_data.csv",
        "upload_csv_key": "school_id_upload_csv",
        "camera_label": "Capture School ID Card",
        "camera_key": "school_id_camera",
        "camera_help_text": "Point your camera at the school ID card and click **Take Photo**.",
        "camera_csv_name": "school_id_camera_data.csv",
        "camera_csv_key": "school_id_camera_csv",
        "camera_records_key": "school_id_camera_records",
        "camera_hash_key": "school_id_camera_hash",
        "camera_counter_key": "school_id_camera_counter",
        "camera_filename_prefix": "school_id_camera_capture",
        "camera_editor_key": "school_id_camera_editor",
        "delete_camera_key": "school_id_delete_camera_rows",
        "clear_camera_key": "school_id_clear_camera_data",
    }

try:
    extractor = config["extractor_getter"]()
    st.success(f"{document_type} extractor loaded successfully!")
except Exception as e:
    st.error(f"Error loading {document_type} extractor: {e}")
    extractor = None

if extractor is not None:
    upload_tab, camera_tab = st.tabs(["Upload Images", "Camera Capture"])

    with upload_tab:
        render_upload_tab(extractor, config)

    with camera_tab:
        render_camera_tab(extractor, config)
