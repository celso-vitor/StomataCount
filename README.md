# StomataCount

**StomataCount** is an open-source tool for automated stomatal counting in microscopy images using object detection.

The current version integrates a Roboflow-hosted detection model with batch image processing, annotated image generation, manual-vs-automatic validation, statistical analysis, and timing comparison between manual and automated counting.

Repository: https://github.com/celso-vitor/StomataCount

---

## Current version

**v0.2.1 - Roboflow-based detection and validation**

This version introduces a Roboflow-based workflow for automated stomatal detection and counting.

## Main features

- Automated stomatal detection using a Roboflow object detection model
- Batch processing of microscopy images
- Support for plant/sample-oriented folder structures
- Annotated images with bounding boxes
- CSV summaries by image and by plant/sample
- Manual-vs-automatic count comparison
- Statistical validation of automatic counts
- Timing analysis comparing manual and automated counting
- Streamlit interface for interactive use

---

## Scientific motivation

Manual stomatal counting is time-consuming, repetitive, and subject to inter-observer variability. StomataCount aims to accelerate and standardize stomatal quantification by combining image-based object detection with reproducible data analysis.

The tool is being developed as part of a methodological workflow for plant phenotyping, biotechnology, and biological image analysis.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/celso-vitor/StomataCount.git
cd StomataCount
