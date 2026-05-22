from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="objdetection/Files/images/YoloDataset/images/val/Angelina_Jolie_95.jpg",
    save=True,
    conf=0.5,
    iou=0.4,
)
print(results)
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        name = model.names[cls_id]
        print(f"Detected: {name} ({conf*100:.1f}% confidence)")