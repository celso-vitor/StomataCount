# StomataCount (SC)

**StomataCount (SC)** is an open-source image analysis tool developed to automate the quantification of stomata in leaf epidermis images captured through optical microscopy. Designed specifically for *Coffea canephora*, this tool supports high-throughput phenotyping, reduces human error, and enhances consistency in morphological studies.

---

## Project Objective

To apply computer vision techniques for the automated and reproducible analysis of plant microscopic structures, improving data quality and analytical efficiency in plant science research.

---

## Key Features

- **Fully automated image processing pipeline**
  - Grayscale conversion
  - Contrast enhancement using CLAHE
  - Noise reduction via median filtering
  - Otsu’s thresholding for binarization
  - Morphological operations: dilation, erosion, and opening

- **Stomata detection**
  - Circular structure detection via Hough Circle Transform
  - Overlap filtering to avoid double-counting

- **Visual result validation**
  - Circles drawn over detected stomata
  - Side-by-side display of original and processed images

- **Customizable detection parameters**
  ```python
  dp = 1.2            # Inverse resolution ratio
  minDist = 10        # Minimum distance between circle centers
  param1 = 100        # Edge detection threshold
  param2 = 28         # Circle center detection threshold
  minRadius = 5       # Minimum stomata radius
  maxRadius = 30      # Maximum stomata radius

