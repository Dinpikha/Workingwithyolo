from ultralytics import YOLO

model= YOLO("yolov8n.pt")

image='/home/Dipika/code/objdetection/Files/reference.jpg'
resultsforipimage=model(image, show=True, conf=0.5)


resultsforipimage[0].show()
# results=model(source=0, show=True, conf=0.5,stream=True)

# for r in results:
    # pass



