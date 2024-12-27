import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import websocket
import json
import math
from threading import Thread

current_accel_x, current_accel_y = 0.0, 0.0

current_theta = 0.0
calibration_offsets = {"x": 0.0, "y": 0.0, "z": 0.0}
calibrated = False
calibration_samples = []

THRESHOLD_X = 0.1  # m/s²
THRESHOLD_Y = 0.1  # m/s²

incomputer = True

camera_ang = 14
x_angle = 0
camera_h = 47  # cm
focal = 3.67
h_sensor = 5.7
hfov = 70.0
vfov = 50.0
hfov_rad = np.radians(hfov)
vfov_rad = np.radians(vfov)
obs_circle_default = 1
obs_circle = obs_circle_default
_resize = 0.4
closest_obs_y = 1000000

mapx = 120
mapy = 450  

directionx = 0.0
directiony = 0.0

fig, ax = plt.subplots()

# Camera position
camera_x, camera_y, camera_theta = 0.0, 0.0, 0.0 

smoothed_gyro_z = 0.0
smoothed_gyro_z_rad = 0.0
def on_message(ws, message):
    global current_accel_x, current_accel_y, smoothed_gyro_z, smoothed_gyro_z_rad
    data = json.loads(message)
    values = data.get('values', [])

    if len(values) < 3:
        print("Incomplete IMU data received.")
        return
    
    current_accel_x = values[0]
    current_accel_y = values[1]

    gyro = data.get('gyro', [])
    if len(gyro) >= 3:
        smoothed_gyro_z = 0.98 * smoothed_gyro_z + 0.02 * gyro[2]
        smoothed_gyro_z_rad = math.radians(smoothed_gyro_z)
        print(f"IMU Data: accel_x={current_accel_x}, accel_y={current_accel_y}, gyro_z={smoothed_gyro_z_rad:.4f} rad/s")

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_code, reason):
    print(f"WebSocket closed: {reason}")

def on_open(ws):
    print("WebSocket connection established.")

def connect(url):
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

class KalmanFilter:
    def __init__(self):
        # State: [x, y, theta, vel_x, vel_y, angular_vel]
        self.state = np.zeros(6)
        self.P = np.eye(6) * 1000 

        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= 0.1 
        self.Q[3:6, 3:6] *= 0.1
        self.R_vision = np.eye(2) * 0.5 
        
        self.dt = 1.0/30.0 

    def predict(self, accel_x, accel_y, gyro_z):
        F = np.eye(6)
        F[0, 3] = self.dt  
        F[1, 4] = self.dt 
        F[2, 5] = self.dt 
        
        B = np.array([
            [0.5 * self.dt**2, 0],
            [0, 0.5 * self.dt**2],
            [0, 0],
            [self.dt, 0],
            [0, self.dt],
            [0, self.dt]
        ])
        u = np.array([accel_x, accel_y])
        
        self.state[2] += gyro_z * self.dt
        self.state[5] = gyro_z 

        self.state = F @ self.state + B @ u
        self.P = F @ self.P @ F.T + self.Q

        print(f"[KF] After Prediction: {self.state}")

    def update_vision(self, measurements, landmarks):
        """
        measurements: List of detected landmarks (x, y) in world frame
        landmarks: List of stored landmarks (x, y) in world frame
        """
        for meas in measurements:
            min_dist = float('inf')
            associated_landmark = None
            for lm in landmarks:
                dist = np.linalg.norm(np.array(meas) - np.array(lm))
                if dist < min_dist:
                    min_dist = dist
                    associated_landmark = lm
            if min_dist < 100.0: 
                z = np.array(associated_landmark)
                H = np.zeros((2, 6))
                H[0, 0] = 1  # x position
                H[1, 1] = 1  # y position
                y = z - H @ self.state
                # covariance
                S = H @ self.P @ H.T + self.R_vision
                # Kalman Gain
                K = self.P @ H.T @ np.linalg.inv(S)
                # Update state
                self.state = self.state + K @ y
                # Update covariance
                I = np.eye(6)
                self.P = (I - K @ H) @ self.P

                print(f"[KF] After Update with measurement {z}: {self.state}")
            else:
                print(f"[KF] No association for measurement {meas} (min_dist={min_dist:.2f})")

kf = KalmanFilter()

def integrate_kalman_filter(create_map_func):
    global kf
    
    def wrapper(*args, **kwargs):
        global camera_x, camera_y, camera_theta, current_accel_x, current_accel_y
        
        px, py = create_map_func(*args, **kwargs)
        measurements = list(zip(px, py)) 
        
        # Get IMU measurements from websocket
        try:
            accel_x = current_accel_x  
            accel_y = current_accel_y 
            gyro_z = smoothed_gyro_z_rad  # radians
            
            print(f"[KF] IMU Measurement: accel_x={accel_x}, accel_y={accel_y}, gyro_z={gyro_z:.4f} rad/s")
            
            # Predict with accelerations and gyro
            kf.predict(accel_x, accel_y, gyro_z)
            
        except NameError:
            print("IMU data not available.")
            pass 
        
        if env_memory.is_environment_captured and env_memory.mode == "Navigation":
            landmarks = list(zip(env_memory.stored_px, env_memory.stored_py))
            kf.update_vision(measurements, landmarks)
        
        camera_x = kf.state[0]
        camera_y = kf.state[1]
        camera_theta = np.degrees(kf.state[2])
        
        print(f"[KF] State: x={camera_x:.2f}, y={camera_y:.2f}, theta={camera_theta:.2f} degrees")
        update_plot(px, py, camera_x, camera_y, camera_theta)
        
        return px, py
    
    return wrapper

class EnvironmentMemory:
    def __init__(self):
        self.stored_px = []
        self.stored_py = []
        self.is_environment_captured = False
        self.camera_offset_x = 0.0
        self.camera_offset_y = 0.0
        self.mode = "Mapping" 

env_memory = EnvironmentMemory()

def update_plot(px, py, camera_x, camera_y, camera_theta, color="red"):
    ax.clear()
    ax.set_xlim(-mapx, mapx)
    ax.set_ylim(-mapy, mapy)
    ax.set_aspect('equal')
    if env_memory.is_environment_captured:
        ax.scatter(
            env_memory.stored_px, 
            env_memory.stored_py, 
            color="blue", 
            s=10, 
            label="Landmarks"
        )

    if env_memory.mode == "Mapping":
        transformed_current_px = []
        transformed_current_py = []

        for x, y in zip(px, py):
            rel_x = x * np.cos(np.radians(camera_theta)) - y * np.sin(np.radians(camera_theta))
            rel_y = x * np.sin(np.radians(camera_theta)) + y * np.cos(np.radians(camera_theta))
            world_x = rel_x + camera_x
            world_y = rel_y + camera_y
            transformed_current_px.append(world_x)
            transformed_current_py.append(world_y)

        ax.scatter(transformed_current_px, transformed_current_py, color=color, s=10, label="Current Detections")

    if env_memory.mode == "Navigation" and env_memory.is_environment_captured:
        ax.scatter(camera_x, camera_y, color="black", s=100, marker="o", label="Camera Position")

        arrow_length = 20
        arrow_x = camera_x + arrow_length * np.cos(np.radians(camera_theta))
        arrow_y = camera_y + arrow_length * np.sin(np.radians(camera_theta))
        ax.arrow(camera_x, camera_y, arrow_x - camera_x, arrow_y - camera_y,
                 head_width=5, head_length=5, fc='black', ec='black')

    if env_memory.mode == "Navigation" and env_memory.is_environment_captured:
        ax.legend(loc='upper right')

    plt.draw()
    plt.pause(0.001)

if incomputer:
    def empty(v):
        pass
    cv2.namedWindow("track_bar")
    cv2.resizeWindow("track_bar", 640,320)
    cv2.createTrackbar("x_angle", "track_bar",   0 , 360, empty)
    cv2.createTrackbar("z_angle", "track_bar",   0 , 360, empty)
    cv2.createTrackbar("hfov", "track_bar",   0 , 100, empty)
    cv2.createTrackbar("vfov", "track_bar",   0 , 100, empty)
    cv2.setTrackbarPos("x_angle", "track_bar", 180)
    cv2.setTrackbarPos("z_angle", "track_bar", 180+int(camera_ang))
    cv2.setTrackbarPos("hfov", "track_bar", int(hfov))
    cv2.setTrackbarPos("vfov", "track_bar", int(vfov))
    cap = cv2.VideoCapture(2)  # 1.webcam 2.android

def rotate_point(point, center, angle):
    angle_rad = np.radians(angle)
    translated_point = (point[0] - center[0], point[1] - center[1])
    
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    rotated_point = np.dot(rotation_matrix, translated_point)
    
    final_point = (rotated_point[0] + center[0], rotated_point[1] + center[1])
    
    return final_point

def recalibrate_if_needed():
    global calibration_offsets, calibration_samples, calibrated
    if calibrated and not env_memory.is_environment_captured:
        if abs(current_accel_x) < 0.01 and abs(current_accel_y) < 0.01 and abs(current_theta) < 0.01:
            calibrate_sensor(current_accel_x, current_accel_y, 0)
            print("Recalibrated offsets:", calibration_offsets)

def create_map(data):
    global camera_x, camera_y, camera_theta 

    if incomputer:
        try:
            ret, oimg = cap.read()
            if not ret:
                print("Failed to grab frame")
                return [], []
        except Exception as e:
            print("no pic:", e)
            return [], []

    global obs_circle
    oimg = cv2.GaussianBlur(oimg, (5, 5), 0)
    top, bottom, left, right = 1, 1, 1, 1
    oimg = cv2.copyMakeBorder(oimg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_w = oimg.shape[1]
    img_h = oimg.shape[0]

    hsv_img = cv2.cvtColor(oimg, cv2.COLOR_BGR2HSV)
    
    px = []
    py = []
    list_x = []
    list_y = []
    cl = ["b", "y"]
    lower = [np.array([100, 150, 0]), np.array([20, 100, 100])]  # Blue and Yellow lower bounds
    upper = [np.array([140, 255, 255]), np.array([30, 255, 255])]  # Blue and Yellow upper bounds

    for _ in range(0, len(lower)):
        output = cv2.inRange(hsv_img, lower[_], upper[_])
        output = cv2.Laplacian(output, -1, 1, 1)
        ret, output = cv2.threshold(output, 30, 255, cv2.THRESH_BINARY)
        
        if not incomputer:
            __, contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        for i in range(1, len(contour), 10):
            stapx, stapy = None, None 

            try:
                if cl[_] == 'y':
                    stapx = contour[i][0][0]
                    stapy = contour[i][0][1]
                elif cl[_] == 'b':
                    incontour = cv2.pointPolygonTest(
                        contour, 
                        (int(contour[i][0][0]), int(contour[i][0][1] - 5)), 
                        False
                    )
                    if incontour > 0:
                        stapx = contour[i][0][0]
                        stapy = contour[i][0][1]

                if stapx is not None and stapy is not None:
                    list_x.append(stapx)
                    list_y.append(stapy)

            except Exception as e:
                print(f"Error processing contour point: {e}")
                pass



        for i in range(len(list_x)):
            if list_x[i] < 100000:
                color_cv = (255, 255, 0) if _ == 1 else (255, 0, 0)  # Yellow for index 1, Blue for index 0
                oimg = cv2.circle(oimg, (list_x[i], list_y[i]), 2, color_cv, 3)

    hfov_rad = np.radians(hfov)
    vfov_rad = np.radians(vfov)

    for i in range(len(list_x)):
        ztheta = np.degrees(np.arctan((list_y[i] - (img_h / 2)) * np.tan(np.radians(vfov / 2)) / (img_h / 2))) + camera_ang
        y = (camera_h / np.sin(np.radians(ztheta))) * np.sin(np.radians(90 - ztheta))

        rotate_ang = np.radians(abs(camera_ang - ztheta))
        rotate_y = y / np.cos(np.radians(ztheta))
        hfov_rotate = 2 * np.arcsin(np.tan(hfov_rad / 2) / np.sqrt(np.tan(rotate_ang) ** 2 + (1 / (np.cos(hfov_rad / 2))) ** 2))
        hfov_rotate = np.degrees(hfov_rotate)

        xtheta = np.degrees(np.arctan((list_x[i] - (img_w / 2)) * np.tan(np.radians(hfov_rotate / 2)) / (img_w / 2)))
        x = rotate_y * np.tan(np.radians(xtheta))

        x_rotate = rotate_point((x, y), (0, 0), -x_angle)
        x = x_rotate[0]
        y = x_rotate[1]

        px.append(x)
        py.append(y)

    print(f"Detected Points: px={len(px)}, py={len(py)}")
    update_plot(px, py, camera_x, camera_y, camera_theta)

    cv2.imshow("Video Feed", oimg)
    cv2.setWindowProperty("Video Feed", cv2.WND_PROP_TOPMOST, 1)  # Keep video feed window on top

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # 'c' to capture environment
        if not env_memory.is_environment_captured:
            print("Environment capture triggered.")
            env_memory.stored_px = px.copy()
            env_memory.stored_py = py.copy()
            env_memory.is_environment_captured = True
            env_memory.mode = "Mapping"
            # if env_memory.stored_px and env_memory.stored_py:
            #     camera_x = np.mean(env_memory.stored_px)
            #     camera_y = np.mean(env_memory.stored_py)
            #     camera_theta = 0.0  # Initial orientation
            #     kf.state[0] = camera_x
            #     kf.state[1] = camera_y
            #     kf.state[2] = np.radians(camera_theta)
            #     kf.state[3:6] = 0.0  # Reset velocities
            #     kf.P = np.eye(6) * 10  # Reduce uncertainty
                # print(f"Captured environment: x={camera_x}, y={camera_y}, theta={camera_theta}")
    elif key == ord('n'):  # 'n' to start navigation
        if env_memory.is_environment_captured:
            env_memory.mode = "Navigation"
            print("Entered Navigation Mode!")
            env_memory.stored_px = px.copy()
            env_memory.stored_py = py.copy()
            env_memory.is_environment_captured = True
            if env_memory.stored_px and env_memory.stored_py:
                camera_theta = 0.0 
                
                kf.state[0] = camera_x
                kf.state[1] = camera_y
                kf.state[2] = np.radians(camera_theta)  # Convert to radians
                kf.state[3] = 0.0  # Initial velocity x
                kf.state[4] = 0.0  # Initial velocity y
                kf.state[5] = 0.0  # Initial angular velocity

                kf.P = np.eye(6) * 10  # Lower uncertainty
                
                print(f"Camera initialized based on vision: x={camera_x:.2f}, y={camera_y:.2f}, theta={camera_theta:.2f} degrees")
            print("Environment captured! Stored points:", len(env_memory.stored_px))
    elif key == 27:  # ESC key to exit
        print("ESC key pressed. Exiting...")
        return [], []

    return px, py

create_map = integrate_kalman_filter(create_map)

lower_colors = [np.array([100, 150, 0]), np.array([20, 100, 100])]  # Blue and Yellow lower bounds
upper_colors = [np.array([140, 255, 255]), np.array([30, 255, 255])]  # Blue and Yellow upper bounds

def main():
    plt.axis('square')
    plt.xlim(-mapx, mapx)
    plt.ylim(-mapy, mapy) 

    while True:
        px, py = create_map(0)
        global camera_ang, x_angle, hfov, vfov, hfov_rad, vfov_rad
        x_angle = cv2.getTrackbarPos("x_angle", "track_bar") - 180
        camera_ang = cv2.getTrackbarPos("z_angle", "track_bar") - 180
        hfov = cv2.getTrackbarPos("hfov", "track_bar")
        vfov = cv2.getTrackbarPos("vfov", "track_bar")
        hfov_rad = np.radians(hfov)
        vfov_rad = np.radians(vfov)

        time.sleep(0.01)

if __name__ == '__main__':
    try:
        accel = Thread(target=connect, args=("ws://10.1.1.39:8080/sensor/connect?type=android.sensor.accelerometer",), daemon=True)
        gyro = Thread(target=connect, args=("ws://10.1.1.39:8080/sensor/connect?type=android.sensor.gyroscope",), daemon=True)
        
        accel.start()
        gyro.start()
        recalibrate_if_needed()

        main()
    except KeyboardInterrupt:
        print("Program interrupted")
    finally:
        if incomputer:
            cap.release()
        cv2.destroyAllWindows()
        print("Resources released")