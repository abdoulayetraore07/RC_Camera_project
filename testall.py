from camerayolo import Camera
import cv2
import os
import time

camera = Camera()
camera.fromfile = False

folder = 'photos/2/'
loop = True

while loop:
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))

        frame = cv2.flip(img, 0)
        frame = cv2.resize(frame, (576, 325))


        inf_start = time.time()
        ray_pos, ray_label, frame_disp = camera.process_stream(frame)
        inf_end = time.time()


        cv2.imshow('Frame', frame_disp)
        print(filename)
        print("inference time : ", inf_end-inf_start)
        time.sleep(3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            loop = False
            break
    

cv2.destroyAllWindows()