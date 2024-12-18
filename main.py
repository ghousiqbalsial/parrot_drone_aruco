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
time.sleep(10)

# PID parameters horizontal
Kp = 0.2
Ki = 0.0
Kd = 0.07

# PID parameters vertical
KpV = 0.2
KiV = 0.03
KdV = 0.07

# PID parameters distance
KpD = 0.05
KiD = 0.02
KdD = 0.02

# Error in horizontal
error = 0

# PID variables
previous_error = 0
previous_errorV = 0
previous_errorD = 0
integral = 0
integralV = 0
integralD = 0
previous_time = time.time()
previous_timeV = time.time()
previous_timeD = time.time()

# Initial Control
control_signal = 0
control_signalV = 0
control_signalD = 0

# Main loop
while (True):

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
    
    # Overlay the marker border on the frame
    if ids is not None:
        aruco.drawDetectedMarkers(dst_color, corners, ids)
        
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 10, newcameramtx, dist)
        
        # get height and width of the image
        h, w = gray.shape[:2]
        
        # Get the distance of the marker from the center
        distance = numpy.mean(corners[0][0], axis=0) - numpy.array([320, 160])
        
        # Draw line from center of aruco marker to the center of the image
        cv2.line(dst_color, (320, 160), (int(numpy.mean(corners[0][0], axis=0)[0]), int(numpy.mean(corners[0][0], axis=0)[1])), (0, 255, 0), 2)
        

        # Calculate the error
        error = distance[0]
        errorV = distance[1]
        errorD = tvecs[0][0][2]
        
        # Print the distance
        print("Distance X: ", distance[0])
        print("Distance Y: ", distance[1])
        print("TVECS: ", errorD)
        
        # If the error is greater than 55, move the drone horizontally
        if abs(error) > 55:
                
            # Calculate the current time and time difference
            current_time = time.time()
            delta_time = current_time - previous_time
            
            # Calculate the integral of the error
            integral += error * delta_time

            # Calculate the derivative of the error
            derivative = (error - previous_error) / delta_time

            # Calculate the control signal
            control_signal = Kp * error + Ki * integral + Kd * derivative

            # Move the drone based on the control signal
            drone.setSpeed(abs(control_signal)*0.0005)
            print(control_signal*0.005)
            time.sleep(0.01)
            
            # Update the previous error and time
            previous_error = error
            previous_time = current_time
        
        if abs(errorV > 40): # If the error is greater than 15, move the drone vertically
            
            # Calculate the current time and time difference
            current_timeV = time.time()
            delta_timeV = current_timeV - previous_timeV
            
            # Calculate the integral of the error
            integralV += errorV * delta_timeV
            
            # Calculate the derivative of the error
            derivativeV = (errorV - previous_errorV) / delta_timeV
            
            # Calculate the control signal
            control_signalV = KpV * errorV + KiV * integralV + KdV * derivativeV
            
            # Update the previous error and time
            previous_errorV = errorV
            previous_timeV = current_timeV
            
        if errorD > 150: # If the error is greater than 140cm, move the drone forward
            
            # Calculate the current time and time difference
            current_timeD = time.time()
            delta_timeD = current_timeD - previous_timeD
            
            # Calculate the integral of the error
            integralD += errorD * delta_timeD
            
            # Calculate the derivative of the error
            derivativeD = (errorD - previous_errorD) / delta_timeD
            
            # Calculate the control signal
            control_signalD = KpD * errorD + KiD * integralD + KdD * derivativeD
            print("Going forward")
            
        
        elif errorD < 100: # If the error is less than 75cm, move the drone backward
            
            # Calculate the current time and time difference
            current_timeD = time.time()
            delta_timeD = current_timeD - previous_timeD
            
            # Calculate the integral of the error
            integralD += errorD * delta_timeD
            
            # Calculate the derivative of the error
            derivativeD = (errorD - previous_errorD) / delta_timeD
            
            # Calculate the control signal
            control_signalD = KpD * errorD + KiD * integralD + KdD * derivativeD
            control_signalD = -control_signalD
            print("Going back")
            
        if errorD < 100 or errorD > 150:
            # Update the previous error and time
            previous_errorD = errorD
            previous_timeD = current_timeD

            
        # Move the drone according to all the calculations
        drone.move(control_signal*0.005, 0, -control_signalV*0.01, 0)
        if control_signalD > 0:
            drone.moveBackward(control_signalD*0.001)
        else:
            drone.moveForward(control_signalD*0.001)
    
    else:
        print("Not Moving")
        drone.move(0, 0, 0, 0)
    
    # Show the image
    cv2.imshow("Frame", dst_color)

# Release the camera
cam.release()

# Destroy all the windows
cv2.destroyAllWindows()

# Land the drone
drone.land()

# Get battery status
print("Battery: " + str(drone.getBattery()[0]) + "%")
