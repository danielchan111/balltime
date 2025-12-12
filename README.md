# Volleyball Deadtime Remover

A machine learning project that automatically detects and removes dead time from volleyball game film using ball detection and a CNN classifier. This tool helps athletes and coaches save time by automating the manual process of identifying play intervals in video footage.

## What it Does

Our project uses an existing ball detection model (https://github.com/masouduut94/volleyball-ml-models) to detect if the ball is in the top half of frames. It then runs a CNN over the results of the ball detection model to label the center frame of windows as in-play or dead time. The pipeline performs batched inference with frame-skipping and FP16 precision to efficiently process full-length game videos, storing confidence tensors for downstream classification. The CNN is trained on high-variance windows to focus learning on play boundaries rather than uniform regions, and test-time inference includes island removal filtering to produce clean play interval timestamps.

## Quick Start

1. **Environment Setup**: Download the notebook (`balltime.ipynb`) and open it in Google Colab with a T4 GPU kernel enabled.

2. **Prepare Videos**: Download the sample volleyball videos from this folder and upload them to your Google Drive at the path `MyDrive/videos/`: https://drive.google.com/drive/folders/1PXtGG0hridkFrikvm5QbKXt56ZFVx2v4?usp=sharing

3. **Run the Pipeline**: Execute the notebook cells in order:
   - **Cell 1**: Clone the balltime repository and set working directory
   - **Cell 2**: Install dependencies from `requirements.txt`
   - **Cell 3**: Initialize the MLManager for ball detection
   - **Cell 4**: Mount Google Drive and configure video paths
   - **Cell 5**: Import video utilities and set inference parameters
   - **Cell 6**: Build detection tensors by running YOLO inference on all videos
   - **Cell 7**: Train the CNN model on labeled data
   - **Cell 8**: Run inference on test videos and extract play intervals

4. **Review Results**: After inference, check `test_play_detections.csv` for detected play timestamps (start/end times in seconds and frame numbers).

## Video Links

Demo Video: [https://drive.google.com/file/d/15b8x-wLkM0HGUx1z0Itvm6U7liP_AqOX/view?usp=sharing]

Technical Walkthrough: [https://drive.google.com/file/d/1yBCMeGApUO1rT__iv8leAIiEp3RqLe7z/view?usp=sharing]

## Evaluation

Unfortunately, this model was unable to overcome overfitting despite numerous hyperparameter tuning attempts. While the model achieved reasonable accuracy on individual sample windows during training, it failed to generalize to full-length video inference. The primary issue manifested as the model classifying the entire video as a single continuous play, rather than identifying distinct play/dead-time intervals. 

Various mitigation strategies were explored including:
- Implementing high-variance window sampling to focus on boundary frames
- Increasing dropout (0.5) to reduce overfitting
- Reducing model capacity (16 conv channels instead of 32)
- Adjusting positive weight for class imbalance
- Implementing island removal (minimum 5-frame sequences) for post-processing

Despite these efforts, the model continued to struggle with temporal generalization on full sequences, suggesting that the sliding-window approach may fundamentally struggle with the long-range temporal dependencies required for accurate play boundary detection in game film. Future work should explore recurrent architectures or transformer-based approaches to better capture the sequential nature of play patterns.
