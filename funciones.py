import cv2, csv
import matplotlib.pyplot as plt
import numpy as np
global file_name

def detect_color(frame, lower_color, upper_color, color_bgr):
    x=0
    y=0
    frame = brillo(frame)
    # convert to hsv color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # create mask for color
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Apply a series of dilations and erosions to eliminate any small blobs left in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # Find contours and centroid in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame=center_object(frame, contours) # Centroid
    
    if contours:
        # draw bounding box around object
        object_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(object_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
        cv2.putText(frame, str(x) + "," + str(y),
            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 55), 1)
    return frame, x, y

def center_object(frame, contours):
    if len(contours) > 0:
        # Find the largest contour, assuming it is the object
        max_contour = max(contours, key=cv2.contourArea)
        # Calculate the center of mass of the object
        M = cv2.moments(max_contour)
        # Enters if the center is different than 0
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # Draw a circle at the center of the object
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
    return frame

def val_coords(x, y, xval, yval):
    x_in_interval = xval-10 <= x <= xval+10
    y_in_interval = yval-10 <= y <= yval+10
    return x_in_interval and y_in_interval

def gen_graph():
    global file_name
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = []
    y = []
    z = []
    x2 = []
    y2 = []
    z2 = []
    
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
            z.append(float(row[2]))            
            x2.append(float(row[3]))
            y2.append(float(row[4]))
            z2.append(float(row[5]))

    ax.plot(x, y, z, c='blue')
    ax.plot(x2, y2, z2, c='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def brillo(img):
    # Aplicar brillo para reflejar blancos
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    contrast = 1.0
    brightness = 20
    frame[:,:,2] = np.clip(contrast * frame[:,:,2] + brightness, 0, 255)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    return frame

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = []
y = []
z = []
x2 = []
y2 = []
z2 = []

with open("Datos_Sutura\Prueba_2023-05-18_13-12-20.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))
        z.append(float(row[2]))            
        x2.append(float(row[3]))
        y2.append(float(row[4]))
        z2.append(float(row[5]))

ax.plot(x, y, z, c='blue')
ax.plot(x2, y2, z2, c='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
'''

from screeninfo import get_monitors

# Get the screen resolution
screen_info = get_monitors()[0]
screen_width = screen_info.width
screen_height = screen_info.height

# Print the screen resolution
print("Screen resolution: {}x{}".format(screen_width, screen_height))