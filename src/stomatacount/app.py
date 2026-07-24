import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from stomatacount.roboflow_detector import RoboflowStomataDetector
from stomatacount.visualization import draw_predictions


st.set_page_config(
    page_title="StomataCount",
    page_icon="🌿",
    layout="wide",
)

st.title("StomataCount")
st.caption("Automatic stomata counting using a Roboflow object detection model.")

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.30,
    step=0.05,
)

uploaded_files = st.file_uploader(
    "Upload microscopy images",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Analyze images"):
        detector = RoboflowStomataDetector()
        rows = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for uploaded_file in uploaded_files:
                input_path = tmpdir / uploaded_file.name
                output_path = tmpdir / f"{input_path.stem}_annotated{input_path.suffix}"

                input_path.write_bytes(uploaded_file.getbuffer())

                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    analysis = detector.analyze_image(
                        input_path,
                        confidence_threshold=confidence_threshold,
                    )

                    draw_predictions(
                        image_path=input_path,
                        predictions=analysis["predictions"],
                        output_path=output_path,
                        show_labels=True,
                    )

                rows.append(
                    {
                        "image": uploaded_file.name,
                        "total_stomata": analysis["total_stomata"],
                        "mean_confidence": analysis["mean_confidence"],
                        "min_confidence": analysis["min_confidence"],
                        "max_confidence": analysis["max_confidence"],
                    }
                )

                st.subheader(uploaded_file.name)

                col1, col2 = st.columns(2)

                with col1:
                    st.image(str(input_path), caption="Original image", use_container_width=True)

                with col2:
                    st.image(str(output_path), caption="Annotated image", use_container_width=True)

                st.metric("Detected stomata", analysis["total_stomata"])

        results = pd.DataFrame(rows)

        st.subheader("Summary")
        st.dataframe(results, use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="stomatacount_results.csv",
            mime="text/csv",
        )
else:
    st.info("Upload one or more microscopy images to start.")
