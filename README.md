# StomataCount

**StomataCount** is an open-source tool for automated stomatal counting and microscopy image analysis using object detection.

The current stable workflow integrates a Roboflow-hosted detection model with batch image processing, annotated image generation, manual-vs-automatic validation, statistical analysis, and timing comparison between manual and automated counting.

The project is now also being extended into a local **desktop scientific interface** using **PySide6 / Qt**, with the goal of supporting interactive visual review, multi-image analysis, detection inspection, and future morphometric phenotyping of stomata.

Repository: https://github.com/celso-vitor/StomataCount

---

## Current version

**v0.2.1 — Roboflow-based detection and validation**

This version introduces a Roboflow-based workflow for automated stomatal detection and counting.

---

## Main features

### Current command-line and analysis workflow

- Automated stomatal detection using Roboflow object detection models
- Batch processing of microscopy images
- Support for plant/sample-oriented folder structures
- Annotated images with detection boxes
- CSV summaries by image and by plant/sample
- Manual-vs-automatic count comparison
- Statistical validation of automatic counts
- Timing analysis comparing manual and automated counting
- Streamlit interface prototype for interactive use

### Desktop interface under development

- Local desktop interface built with **PySide6 / Qt**
- Image viewer with zoom and pan
- Multi-image loading
- Drag-and-drop image loading planned
- Detection table synchronized with the image viewer planned
- Clickable detections planned
- Ellipse/contour-based annotation planned
- Future support for morphometric measurements

---

## Scientific motivation

Manual stomatal counting is time-consuming, repetitive, and subject to inter-observer variability. StomataCount aims to accelerate and standardize stomatal quantification by combining image-based object detection with reproducible data analysis.

The tool is being developed as part of a methodological workflow for plant phenotyping, biotechnology, and biological image analysis.

The long-term goal is to move beyond simple counting toward **stomatal phenotyping**, including:

- detection review;
- manual correction;
- contour refinement;
- major and minor axis measurements;
- equivalent diameter;
- area and perimeter;
- pixel-to-micrometer calibration;
- per-object and per-image quantitative exports.

---

## Supported model workflows

The current development workflow includes model selection for different plant systems, including:

- Coffee / *Coffea*
- Cassava / *Manihot*

The current models are object detection models. Future versions may include instance segmentation models to provide more accurate stomatal masks and contour-based measurements.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/celso-vitor/StomataCount.git
cd StomataCount
