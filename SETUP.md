# Setup Guide

## Prerequisites

- Google Colab account
- Access to Google Drive
- T4 GPU enabled in Colab

## Step-by-Step Setup

### 1. Download Videos
Download the sample volleyball videos from this folder and upload them to your Google Drive:
https://drive.google.com/drive/folders/1PXtGG0hridkFrikvm5QbKXt56ZFVx2v4?usp=sharing

Create the folder structure in your Google Drive:
```
MyDrive/
└── videos/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

### 2. Open Notebook in Colab
- Download `balltime.ipynb` from the repository
- Go to [Google Colab](https://colab.research.google.com/)
- Upload or open the `balltime.ipynb` notebook
- Ensure you have GPU enabled: Runtime → Change runtime type → T4 GPU

### 3. Run the Pipeline

Execute the notebook cells in order:

| Cell | Purpose |
|------|---------|
| 1 | Clone repository and set working directory |
| 2 | Install dependencies from `requirements.txt` |
| 3 | Initialize MLManager for ball detection |
| 4 | Mount Google Drive and configure paths |
| 5 | Import utilities and set inference parameters |
| 6 | Build detection tensors (YOLO inference) |
| 7 | Train CNN model on labeled data |
| 8 | Run inference and extract play intervals |

### 4. Review Results

After the pipeline completes, output files will be generated:

- `tensor_index.csv` - Ball detection confidence scores for all frames
- `video_true_labels.csv` - Ground truth play labels (if provided)
- `best_model.pth` - Trained CNN model weights
- `training_history.csv` - Training and validation metrics
- `training_curves.png` - Loss and accuracy plots
- `test_play_detections.csv` - **Final output with detected play intervals**

The `test_play_detections.csv` file contains:
- `video` - Video filename
- `start_sec` / `end_sec` - Play interval in seconds
- `duration_sec` - Play duration
- `start_frame` / `end_frame` - Frame numbers in original video

## Troubleshooting

**CUDA errors**: The notebook is configured to use CPU if GPU is unavailable. GPU is recommended for performance.

**Videos not found**: Ensure videos are in `MyDrive/videos/` and the folder structure matches exactly.

**Out of memory**: Reduce `BATCH_SIZE` in Cell 5 (default: 64).

**Poor results**: Ensure ground truth labels are properly formatted in `video_true_labels.csv`.
