import streamlit as st
import os
from speed_vehical_detector import process_video

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

st.set_page_config(
    page_title="Vehicle Speed Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Speed Detection")
st.write("Upload a video to detect vehicles and estimate their speed.")

uploaded_file = st.file_uploader(
    "Choose a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    input_path = os.path.join("uploads", uploaded_file.name)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    output_path = os.path.join("outputs", "processed_video.mp4")

    st.info("Processing video... Please wait.")

    try:
        with st.spinner("Running YOLOv8 + ByteTrack..."):
            process_video(input_path, output_path)

        st.success("Processing completed!")

        st.subheader("Processed Video")
        st.video(output_path)

        with open(output_path, "rb") as file:
            st.download_button(
                label="⬇ Download Processed Video",
                data=file,
                file_name="vehicle_speed_detection.mp4",
                mime="video/mp4"
            )

    except Exception as e:
        import traceback

        traceback.print_exc()   # Shows full error in Render logs
        st.error(f"Error: {e}")