import cv2

def UserCam(): 
    # Initialize video capture
    cap = cv2.VideoCapture(2)
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()  
        # Resize the frame to the desired width and height
        frame = cv2.resize(frame, (1000,800))
        # Display the original image with contours and the thresholded mask
        cv2.imshow("Camera User", frame) 
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

UserCam()