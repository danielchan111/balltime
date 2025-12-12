# Attribution

## AI-Generated Code

This project was developed with significant assistance from **GitHub Copilot** (Claude Haiku 4.5), an AI coding assistant. The following components were generated or substantially developed with AI assistance:

### Code Components
- **`notebooks/balltime.ipynb`** - Entire notebook structure, including:
  - Data loading and preprocessing pipelines
  - YOLO inference wrapper with batching, FP16 precision, and frame-skipping optimizations
  - Tensor storage and CSV I/O functions
  - CNN model architecture (`BallDetectionCNN`)
  - Training loop with learning rate scheduling and early stopping
  - High-variance dataset sampling class (`HighVarianceDataset`)
  - Test inference pipeline with sliding window prediction
  - Island removal and play interval extraction logic

- **`src/video_objects.py`** - Video processing utilities including:
  - `VideoObject` and `VideoDetectionTensor` data structures
  - Video-to-object conversion functions
  - Binary label generation and alignment
  - CSV serialization functions

- **`src/label_videos.py`** - Interactive CLI for video labeling

### Documentation
- **`README.md`** - Project overview, setup instructions, and evaluation summary
- **`SETUP.md`** - Step-by-step setup and troubleshooting guide
- **`ATTRIBUTION.md`** - This file

---

## External Libraries and Dependencies

### Core Dependencies
- **PyTorch** (`torch`) - Deep learning framework used for CNN training and inference
  - License: BSD
  - URL: https://pytorch.org/

- **Ultralytics YOLO** (`ultralytics`) - Pre-trained ball detection model
  - License: AGPL-3.0
  - URL: https://github.com/ultralytics/ultralytics

- **OpenCV** (`cv2`) - Video frame extraction and image processing
  - License: Apache 2.0
  - URL: https://opencv.org/

- **NumPy** (`numpy==1.26.4`) - Numerical computing and array operations
  - License: BSD
  - URL: https://numpy.org/

- **pandas** - Data manipulation and CSV I/O
  - License: BSD
  - URL: https://pandas.pydata.org/

- **SciPy** (`scipy`) - Scientific computing, specifically `scipy.ndimage` for connected component labeling
  - License: BSD
  - URL: https://scipy.org/

- **tqdm** - Progress bar visualization
  - License: Mozilla Public License 2.0 (MPL 2.0)
  - URL: https://github.com/tqdm/tqdm

- **Matplotlib** - Visualization of training curves
  - License: PSF
  - URL: https://matplotlib.org/

### Development Environment
- **Google Colab** - Cloud-based Jupyter notebook environment with GPU support
  - URL: https://colab.research.google.com/

---

## External Resources and Models

### Ball Detection Model
- **volleyball-ml** - Existing ball detection model used for initial frame analysis
  - URL: https://github.com/masouduut94/volleyball-ml-models
  - Usage: YOLO model accessed via `MLManager` for ball detection inference
  - Note: This project builds upon the existing infrastructure provided by this library

### Datasets
- **Volleyball Video Dataset** - Sample videos provided for training and evaluation
  - Source: Google Drive (internal project folder)
  - Note: Dataset used for proof-of-concept; not publicly available

---

## Methodological References

While not directly used in code, the following concepts informed the project design:

- **Sliding Window Classification** - Standard approach for temporal boundary detection in videos
- **High-Variance Sampling** - Technique to focus training on transitions between classes
- **Island Removal (Connected Component Labeling)** - Post-processing technique to filter noise in predictions
- **Batch Inference with Mixed Precision** - Performance optimization techniques in deep learning

---

## Limitations and Disclaimers

### AI Code Generation Limitations
- AI-generated code was reviewed and modified iteratively based on testing outcomes
- Some functions required debugging and refinement (e.g., dimension mismatches in CNN layers)
- The model's overfitting issue was identified through experimentation and addressed with multiple mitigation strategies

### Model Performance
- This project represents a proof-of-concept with known limitations (see `README.md` Evaluation section)
- The model was not able to achieve production-quality play detection
- Results should not be used for critical applications without additional validation

---

## How to Cite This Project

If you use this project or build upon it, please cite:

```
Chan, Daniel. Volleyball Deadtime Remover. GitHub, 2025. 
https://github.com/danielchan111/balltime

Developed with AI assistance from GitHub Copilot.
```

---

## Contact and Questions

For questions about attribution, AI usage, or project details, please refer to the main `README.md` or contact the project maintainer.

Last Updated: December 12, 2025
