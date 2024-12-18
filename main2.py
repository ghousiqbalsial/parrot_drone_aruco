from api import ps_drone
import cv2
from cv2 import aruco
import numpy
import time

# Initialize the drone
drone = ps_drone.Drone()

# Connect to the drone
drone.startup()

# Get battery status
print("Battery: " + str(drone.getBattery()[0]) + "%")

# Connect to drone camera
cam = cv2.VideoCapture('tcp://192.168.1.1:5555')

# Define Aruco Dictionary and Parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

# load calibration parameters
ret = numpy.load('cal_vals/ret.npy')
mtx = numpy.load('cal_vals/mtx.npy')
dist = numpy.load('cal_vals/dist.npy')

# hover the drone
drone.takeoff()
time.sleep(5)

# PID Parameters for horizontal, vertical and distance
Kp = [0.1, 0.2, 0.05] 
Ki = [0.0, 0.03, 0.02]
Kd = [0.07, 0.04, 0.02]

# Error in horizontal, vertical and distance
error = [0, 0, 0]

# PID variables
previous_error = [0, 0, 0]
integral = [0, 0, 0]
previous_time = [time.time(), time.time(), time.time()]

# PID Function
def PID(Kp, Ki, Kd, error, previous_error, integral, previous_time):
    dt = time.time() - previous_time
    integral = integral + error * dt
    derivative = (error - previous_error) / dt
    output = Kp * error + Ki * integral + Kd * derivative

    return output

speed_factor = 0.01

# Main loop
while True:

    # Get frame
    ret, frame = cam.read()

    # Break loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Convert image to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Undistort the image
    h,  w = gray.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    # dst = cv2.undistort(gray, mtx, dist, None, mtx)
    # dst_color = cv2.undistort(frame, mtx, dist, None, mtx)
    
    dst = gray
    dst_color = frame
    
    # Crop undistorted image to the same size of the original
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    dst_color = dst_color[y:y+h, x:x+w]
    
    # Detect the marker and get its position
    corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=parameters)
    
    # Draw a rectangle of pixels 50 by 20 around the center of the image
    cv2.rectangle(dst_color, (320-25, 160-20), (320+25, 160+20), (0, 255, 0), 2)

    # If the marker is detected
    if ids is not None:
        aruco.drawDetectedMarkers(dst_color, corners, ids)

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

        # get the height and width of the image
        h, w = dst_color.shape[:2]

        # get the distance of the marker from the image
        distance = numpy.mean(corners[0][0], axis=0) - numpy.array([320, 160])

        # Draw the line from center of the aruco marker to the center of the image
        cv2.line(dst_color, (320, 160), (int(numpy.mean(corners[0][0], axis=0)[0]), int(numpy.mean(corners[0][0], axis=0)[1])), (0, 255, 0), 2)

        # show the frame
        cv2.imshow('frame', dst_color)

        # Calculate the error
        error = [distance[0], distance[1], tvecs[0][0][2]*100]

        # print the error
        print("Distance X: ", error[0])
        print("Distance Y: ", error[1])
        print("Distance Z: ", error[2])

        # PID control
        if abs(error[0]) > 55:
            control_signalX = PID(Kp[0], Ki[0], Kd[0], error[0], previous_error[0], integral[0], previous_time[0])
            print("Control Signal X: ", control_signalX*speed_factor)
        else:
            control_signalX = 0

        if abs(error[1]) > 40:
            control_signalY = PID(Kp[1], Ki[1], Kd[1], error[1], previous_error[1], integral[1], previous_time[1])
            print("Control Signal Y: ", control_signalY*speed_factor)
        else:
            control_signalY = 0

        if abs(error[2]) > 80 or abs(error[2]) < 60:
            control_signalZ = PID(Kp[2], Ki[2], Kd[2], error[2], previous_error[2], integral[2], previous_time[2])
            if abs(error[2]) < 60:
                control_signalZ = -control_signalZ
            print("Control Signal Z: ", control_signalZ*speed_factor)
        else:
            control_signalZ = 0

        # Update previous error
        previous_error = error

        # Update previous time
        previous_time = [time.time(), time.time(), time.time()]

        # Move the drone
        drone.move(control_signalX*speed_factor, control_signalZ*0, -control_signalY*speed_factor, 0)

    else:
        print("Marker not found")
        drone.hover()

    # show the frame
    cv2.imshow('frame', dst_color)    

# Land the drone
drone.land()

# Release the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()

# Get battery status
print("Battery: " + str(drone.getBattery()[0]) + "%")
    