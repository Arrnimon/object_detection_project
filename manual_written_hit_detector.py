import cv2
import math 
import numpy as np 
from ultralytics import YOLO


class mmaProcessing:
    def __init__(self, video_path, model_path):
        self.video_path = video_path
        self.model_path = model_path
        self.model = None
        self.cap = None

        self.model = YOLO(self.model_path)

    def trainModel(self):
        self.model.train(data="mma_data.yaml", epochs=100, imgsz=640, batch=16, name="mma_yolo_model")



    def processVideo(self):
        while(True):
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            cv2.imwrite(name, frame)

            

    def processImageIndividually(self, image_path):
        results = self.model.predict(source=image_path, conf=0.4, save=True, save_txt=True, project="runs/detect", name="mma_yolo_results", exist_ok=True)
        for r in results:
            # Add the tracker statistics from r objects
            pass  # Placeholder to avoid empty loop error
            
        cv2.imshow("Image", r.orig_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def trackStatistics(r):

        head = array[0, 1, 2, 3, 4]
        striking_tools = array[5, 6, 7, 8, 13, 14, 15, 16]

        for person in keypoints:
            #set the first detected person as the main person to be detected
            main_person = person
            if(main_person[0] == 0):
                 other_person = keypoints[1]
            else:
                 other_person = keypoints[0]
            #all the different "face" keypoints are added to an array called head
            for i in head:
                 for j in striking_tools:
                      if main_person[j][2] > 0 and other_person[i][2] > 0:
                           distance = math.sqrt((main_person[j][0] - other_person[i][0])**2 + (main_person[j][1] - other_person[i][1])**2)
                           if distance < 50:
                                print("Hit detected")
                                #increment hit counter
                                hit_counter += 1
                                #draw a line between the two points
                                cv2.line(r.orig_img, (int(main_person[j][0]), int(main_person[j][1])), (int(other_person[i][0]), int(other_person[i][1])), (0, 255, 0), 2)
                                #put the hit counter on the image
                                cv2.putText(r.orig_img, "Hits: " + str(hit_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)









        


