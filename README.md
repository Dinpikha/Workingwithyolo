# Working with YOLO 🎬

This repository contains my work with the YOLO (You Only Look Once) object detection framework.

I started by experimenting with the lightweight YOLOv8 nano model (yolov8n) for general object detection, then fine-tuned it to detect 17 celebrities from images.

## What it does
- Detects 17 celebrities in images using a fine-tuned YOLOv8n model
- Auto-downloads the dataset from Kaggle
- Auto-generates bounding box labels using OpenCV face detection
- Trained from scratch with 80/20 train/val split

## Celebrities it can detect
Angelina Jolie, Jennifer Lawrence, Megan Fox, Sandra Bullock, Will Smith,
Brad Pitt, Johnny Depp, Natalie Portman, Scarlett Johansson, Denzel Washington,
Kate Winslet, Nicole Kidman, Tom Cruise, Hugh Jackman, Leonardo DiCaprio,
Robert Downey Jr, Tom Hanks

## Installation
Clone the repository:
\```bash
git clone https://github.com/Dinpikha/Workingwithyolo.git
cd Workingwithyolo
\```

Install dependencies:
\```bash
pip install -r requirements.txt
\```

## How to run

**Step 1 - Generate labeled dataset:**
\```bash
python photodetection.py
\```

**Step 2 - Generate data.yaml:**
\```bash
python createyaml.py
\```

**Step 3 - Train the model:**
\```bash
python train.py
\```

**Step 4 - Test on an image:**
\```bash
python test.py
\```

## Results
- Trained on 100 images per celebrity (~1700 total)