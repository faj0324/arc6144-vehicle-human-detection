# Intelligent Visual Surveillance System with Moving Object Recognition

A classical computer-vision surveillance system that detects and classifies **moving people and vehicles** in video, built with OpenCV. Developed as a coursework assignment for **ARC6144 – Machine Vision and Image Processing** (group project, 2024).

The system takes a video, isolates moving objects using background subtraction, cleans up the resulting mask, and draws labelled bounding boxes around each detected object.

## What it does

For every frame of an input video the pipeline:

1. **Background subtraction** — a MOG2 model (`cv2.createBackgroundSubtractorMOG2`) learns the static background and produces a binary foreground mask where moving pixels are white.
2. **Noise removal** — morphological *opening* (erosion → dilation) removes speckle noise and small disconnected blobs from the mask.
3. **Blob detection** — external contours are extracted from the cleaned mask, and tiny regions (area `< 550 px`) are filtered out.
4. **Classification** — each surviving blob is labelled from the aspect ratio of its bounding box (a simple person-vs-vehicle heuristic).
5. **Visualisation & export** — a green bounding box and label are drawn on the frame; both the annotated frame and the binary mask are saved to disk as image sequences.

## Repository layout

```
.
├── src/
│   ├── test01_test_code_for_student.py   # Base detector: background subtraction + mask export
│   ├── car_and_human_test_coding.py      # Extended: adds contours, bounding boxes & labels (Indoor.mp4)
│   └── human_test_coding.py              # Same pipeline, configured for a different clip (i2.mp4)
├── samples/
│   └── outdoor.mp4                       # Small sample clip so the scripts can be run out of the box
├── requirements.txt
├── .gitignore
└── README.md
```

> The extended scripts (`car_and_human_test_coding.py` and `human_test_coding.py`) are near-identical and differ only in which video file they process — they are both kept here to preserve the original submission.

## Getting started

### Prerequisites
- Python 3.8+
- The packages in [`requirements.txt`](requirements.txt) (OpenCV + NumPy)

### Install
```bash
pip install -r requirements.txt
```

### Run
Each script has a `video_to_process` variable near the top that points to the input clip. Point it at a video (e.g. the bundled `samples/outdoor.mp4`) and run:

```bash
cd src
python test01_test_code_for_student.py
```

Two live windows open — the annotated frames and the binary foreground mask — and two folders are created next to the video:

- `<video_name>_frame/` — annotated frames (`Frame0.jpg`, `Frame1.jpg`, …)
- `<video_name>_binary/` — the corresponding binary masks

Press `Esc` (base script) to stop early. These output folders are git-ignored.

## How classification works

The extended scripts assign a label to each blob from the width/height ratio of its bounding box — the intuition being that vehicles tend to be wider than they are tall, while a standing person is taller than they are wide. This is a lightweight heuristic rather than a trained classifier, and it is the main area a modern rewrite would improve (see below).

## Demo videos

The recorded test clips and the annotated result videos are hosted on Google Drive (they're too large to store in the repo):

📁 **[Full dataset & demo folder on Google Drive](https://drive.google.com/drive/folders/132fUkgCWc7iGVwiAMWO8iFYcd4NEzLms?usp=sharing)**

Highlights:
- [Outdoor — detection result](https://drive.google.com/file/d/1o2attThoY0DidKaWx-g1RIFlJkIIUg5o/view)
- [Indoor — detection result](https://drive.google.com/file/d/19mB7__B5ejsEM0cV51wZqCmGgxcn5n8i/view)
- [Screen recording walkthrough](https://drive.google.com/file/d/1QrIOadWS2rzI88VOavSe5OxINohTI6Dn/view)

## Limitations & possible improvements

This is intentionally a *classical* (non-deep-learning) pipeline, and it shows the trade-offs of that approach:

- Aspect-ratio classification is coarse and confuses object types when people and vehicles overlap or partially occlude one another.
- A single global background model struggles with lighting changes, camera shake, and slow-moving objects.
- Detections are per-frame with no tracking, so the same object isn't given a consistent identity across frames.

Natural next steps: swap the heuristic classifier for a trained detector (e.g. YOLO), add object tracking (SORT/DeepSORT) for stable IDs, and tune the morphology/area thresholds per scene.

## Acknowledgements

Coursework for **ARC6144 – Machine Vision and Image Processing**. The moving-object-detection starter approach was provided as part of the assignment; the classification and visualisation on top of it were implemented by the project group.
