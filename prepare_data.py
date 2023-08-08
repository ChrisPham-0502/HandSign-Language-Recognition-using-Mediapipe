import cv2
from pathlib import Path
import os

def get_image(class_name, vietnamese_name):
    Class = class_name
    Path('dataset/'+Class).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i = 0    
    while True:
       
        ret, frame = cap.read()

        frame = cv2.flip(frame,1)
        i+= 1
        if i % 5==0:
            try:
                cv2.imwrite(f'dataset/{Class}/{i}.jpg',frame)
                print("Lưu thành công")
            except:
                print("Lưu thất bại")
        cv2.imshow('Gesture', frame)
        if cv2.waitKey(1) == ord('q') or i > 500:
            break
    old_name = f"dataset/{Class}"
    new_name = f'dataset/{vietnamese_name}'
    os.rename(old_name, new_name)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
   get_image("see", "Hen gap lai")
  