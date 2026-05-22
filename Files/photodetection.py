from ultralytics import YOLO
# import mtcnn
from pathlib import Path
import shutil
import cv2
import kagglehub 
model= YOLO("yolov8n.pt")
path = kagglehub.dataset_download("vishesh1412/celebrity-face-image-dataset")

print("Path to dataset files:", path)
# image='Files/reference.jpg'
# resultsforipimage=model(image, show=True, conf=0.5)

for p in Path(path).iterdir():
    print(p.name)
# resultsforipimage[0].show()


cv2facedetector=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# fine tuning
celebrities = [
    'Angelina Jolie', 'Jennifer Lawrence', 'Megan Fox', 'Sandra Bullock',
    'Will Smith', 'Brad Pitt', 'Johnny Depp', 'Natalie Portman',
    'Scarlett Johansson', 'Denzel Washington', 'Kate Winslet', 'Nicole Kidman',
    'Tom Cruise', 'Hugh Jackman', 'Leonardo DiCaprio', 'Robert Downey Jr',
    'Tom Hanks']



sourcedatasetpath=Path(path)/"Celebrity Faces Dataset" 
outputdatasetpath=Path('Files/images/YoloDataset')

for folders in ['images/val','images/train','labels/val','labels/train']:
    (outputdatasetpath/folders).mkdir(parents=True, exist_ok=True)

skipped = 0
total = 0

for class_id , name in enumerate(celebrities):
    folder=sourcedatasetpath/name
    images=list(folder.glob('*.jpg'))+list(folder.glob('*.png'))
    split_idx=int(0.8*len(images))
    for i ,img_path in enumerate(images):
        split='train' if i<split_idx else 'val'
        img=cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            print(f'Could not read image: {img_path}')
            continue
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h,w=img.shape[:2]

        face=cv2facedetector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30))

        if len(face) == 0:
            skipped += 1
            continue
        dest_img=outputdatasetpath/f'images/{split}'/f'{name.replace(" ", "_")}_{i}.jpg'
        shutil.copy(img_path, dest_img)
        label_path=outputdatasetpath/f'labels/{split}'/f'{name.replace(" ", "_")}_{i}.txt'
        with open(label_path, "w") as f:
            for (x, y, bw, bh) in face:
                
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        print('labeled!')




