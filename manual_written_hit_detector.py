import cv2
import math 
import numpy as np 
from ultralytics import YOLO
import os
import pathlib


class mmaProcessing:
    def __init__(self, video_path, model_path):
        self.video_path = video_path
        self.model_path = model_path
        self.model = None
        self.cap = None
        
        # Initialize hit counters for the entire video
        self.total_fighter1_hits = 0
        self.total_fighter2_hits = 0

        self.model = YOLO(self.model_path)

    def trainModel(self):
        self.model.train(data="mma_data.yaml", epochs=100, imgsz=640, batch=16, name="mma_yolo_model")

    def validateVideoPath(self, video_path):
        """Validate and normalize the video path"""
        # Convert to Path object for better cross-platform handling
        path = pathlib.Path(video_path)
        
        # Check if file exists
        if not path.exists():
            print(f"Error: Video file does not exist at: {video_path}")
            return None
            
        # Check if it's a file
        if not path.is_file():
            print(f"Error: Path is not a file: {video_path}")
            return None
            
        # Check file extension
        if path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            print(f"Warning: File extension {path.suffix} might not be supported")
            
        # Return absolute path as string
        return str(path.absolute())

    def processVideo(self):
        # Validate video path first
        validated_path = self.validateVideoPath(self.video_path)
        if not validated_path:
            return
            
        print(f"Attempting to open video: {validated_path}")
        
        # Open the video file
        self.cap = cv2.VideoCapture(validated_path)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            print("Possible causes:")
            print("1. Video file is corrupted")
            print("2. Missing video codecs")
            print("3. File path contains special characters")
            print("4. Insufficient permissions")
            return

        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video opened successfully!")
        print(f"FPS: {fps}")
        print(f"Total frames: {frame_count_total}")

        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Run pose detection on the frame

            # Draw keypoints and/or process hits here if needed
            """
            
            for r in results:
                if r.keypoints is not None:
                    for person in r.keypoints.xy:
                        for x, y in person:
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

            """
            if frame_count % 3 == 0:
                # Run pose detection on the frame
                results = self.model.predict(source=frame, conf=0.4, save=False, verbose=False)
                for r in results:
                    
                    fighter1_hits, fighter2_hits = self.trackStatistics(r)
                    # Accumulate total hits
                    self.total_fighter1_hits += fighter1_hits
                    self.total_fighter2_hits += fighter2_hits
                    
                    # Display total hits for the entire video
                    cv2.putText(r.orig_img, f"Total - Fighter 1: {self.total_fighter1_hits}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(r.orig_img, f"Total - Fighter 2: {self.total_fighter2_hits}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    frame = r.orig_img
                
            # Show the frame
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()
        
        # Display final results
        print("\n=== FINAL RESULTS ===")
        print(f"Fighter 1 Total Hits: {self.total_fighter1_hits}")
        print(f"Fighter 2 Total Hits: {self.total_fighter2_hits}")
        
        if self.total_fighter1_hits > self.total_fighter2_hits:
            print("🏆 Fighter 1 wins!")
        elif self.total_fighter2_hits > self.total_fighter1_hits:
            print("🏆 Fighter 2 wins!")
        else:
            print("🤝 It's a tie!")
        print("===================")

    def trackStatistics(self, r):
        """Track hits for each fighter separately"""
        head = [0, 1, 2, 3, 4]
        striking_tools = [5, 6, 7, 8, 13, 14, 15, 16]
        
        # Initialize hit counters for each fighter
        fighter1_hits = 0
        fighter2_hits = 0
        
        # Check if we have keypoints and at least 2 people detected
        if hasattr(r, "keypoints") and hasattr(r.keypoints, "xy") and len(r.keypoints.xy) >= 2:
            fighter1 = r.keypoints.xy[0]  # First fighter
            fighter2 = r.keypoints.xy[1]  # Second fighter
            
            # Check if we have confidence scores (keypoints.conf)
            has_confidence = hasattr(r.keypoints, "conf") and r.keypoints.conf is not None
            
            # Check hits from fighter1 to fighter2 (fighter1 striking fighter2)
            for i in head:  # fighter2's head
                for j in striking_tools:  # fighter1's striking tools
                    # Check if indices are valid
                    if (i < len(fighter2) and j < len(fighter1)):
                        
                        # Check confidence if available, otherwise assume valid keypoints
                        fighter1_conf_valid = True
                        fighter2_conf_valid = True
                        
                        if has_confidence and len(r.keypoints.conf) >= 2:
                            if len(r.keypoints.conf[0]) > j:
                                fighter1_conf_valid = r.keypoints.conf[0][j] > 0
                            if len(r.keypoints.conf[1]) > i:
                                fighter2_conf_valid = r.keypoints.conf[1][i] > 0
                        
                        if fighter1_conf_valid and fighter2_conf_valid:
                            # Check if keypoints are valid (not NaN or zero)
                            fighter1_x, fighter1_y = fighter1[j][0], fighter1[j][1]
                            fighter2_x, fighter2_y = fighter2[i][0], fighter2[i][1]
                            
                            if (not math.isnan(fighter1_x) and not math.isnan(fighter1_y) and 
                                not math.isnan(fighter2_x) and not math.isnan(fighter2_y) and
                                fighter1_x > 0 and fighter1_y > 0 and fighter2_x > 0 and fighter2_y > 0):
                                
                                distance = math.sqrt((fighter1_x - fighter2_x)**2 + (fighter1_y - fighter2_y)**2)
                                if distance < 50:
                                    print("Fighter 1 hit Fighter 2!")
                                    fighter1_hits += 1
                                    # draw a line between the two points (green for fighter1 hits)
                                    cv2.line(r.orig_img, (int(fighter1_x), int(fighter1_y)), (int(fighter2_x), int(fighter2_y)), (0, 255, 0), 2)
            
            # Check hits from fighter2 to fighter1 (fighter2 striking fighter1)
            for i in head:  # fighter1's head
                for j in striking_tools:  # fighter2's striking tools
                    # Check if indices are valid
                    if (i < len(fighter1) and j < len(fighter2)):
                        
                        # Check confidence if available, otherwise assume valid keypoints
                        fighter1_conf_valid = True
                        fighter2_conf_valid = True
                        
                        if has_confidence and len(r.keypoints.conf) >= 2:
                            if len(r.keypoints.conf[0]) > i:
                                fighter1_conf_valid = r.keypoints.conf[0][i] > 0
                            if len(r.keypoints.conf[1]) > j:
                                fighter2_conf_valid = r.keypoints.conf[1][j] > 0
                        
                        if fighter1_conf_valid and fighter2_conf_valid:
                            # Check if keypoints are valid (not NaN or zero)
                            fighter1_x, fighter1_y = fighter1[i][0], fighter1[i][1]
                            fighter2_x, fighter2_y = fighter2[j][0], fighter2[j][1]
                            
                            if (not math.isnan(fighter1_x) and not math.isnan(fighter1_y) and 
                                not math.isnan(fighter2_x) and not math.isnan(fighter2_y) and
                                fighter1_x > 0 and fighter1_y > 0 and fighter2_x > 0 and fighter2_y > 0):
                                
                                distance = math.sqrt((fighter2_x - fighter1_x)**2 + (fighter2_y - fighter1_y)**2)
                                if distance < 50:
                                    print("Fighter 2 hit Fighter 1!")
                                    fighter2_hits += 1
                                    # draw a line between the two points (red for fighter2 hits)
                                    cv2.line(r.orig_img, (int(fighter2_x), int(fighter2_y)), (int(fighter1_x), int(fighter1_y)), (0, 0, 255), 2)
        
        # Display hit counters on the image
        cv2.putText(r.orig_img, f"Fighter 1 Hits: {fighter1_hits}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(r.orig_img, f"Fighter 2 Hits: {fighter2_hits}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return fighter1_hits, fighter2_hits
            
"""
    def processImageIndividually(self, image_path):
        results = self.model.predict(source=image_path, conf=0.4, save=True, save_txt=True, project="runs/detect", name="mma_yolo_results", exist_ok=True)
        for r in results:
            # Add the tracker statistics from r objects
            pass  # Placeholder to avoid empty loop error
            
        cv2.imshow("Image", r.orig_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
"""





def main():
    print("Welcome to the MMA Hit Detector!")
    print("Please enter the path to your video file.")
    print("Examples:")
    print("- C:\\Users\\username\\Videos\\fight.mp4")
    print("- /home/user/videos/fight.mp4")
    print("- fight.mp4 (if in same directory)")
    
    videoPath = input("Enter the path to the video file: ").strip()
    
    # Remove quotes if user accidentally added them
    videoPath = videoPath.strip('"\'')
    
    # Check if file exists before proceeding
    if not os.path.exists(videoPath):
        print(f"Error: File not found at {videoPath}")
        print("Please check the path and try again.")
        return

    model_path = "yolov8n-pose.pt"
    image_path = "path_to_image.jpg"

    mma_processor = mmaProcessing(videoPath, model_path)
    
    # Uncomment the following line to train the model
    #mma_processor.trainModel()
    
    # Process a single image
    #mma_processor.processImageIndividually(image_path)
    
    # Uncomment the following line to process a video
    mma_processor.processVideo()

if __name__ == "__main__":
    main()















        


