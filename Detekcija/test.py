import snap7
import cv2
import cv2.aruco as aruco
import numpy as np
import struct
import threading
import time
from collections import deque

# === PLC parametri ===
PLC_IP = '192.168.1.3'
RACK = 0
SLOT = 1
DB_NUMBER = 17
START_ADDRESS = 0

client = snap7.client.Client()
client.connect(PLC_IP, RACK, SLOT)
if client.get_connected():
    print("[INFO] Povezan s PLC-om ✅")
else:
    print("[ERROR] Neuspješno spajanje s PLC-om ❌")
    exit()

def write_real_async(db_number, start, value):
    def _send():
        data = struct.pack('>f', value)
        client.db_write(db_number, start, data)
    threading.Thread(target=_send).start()

# === Kamera kalibracija ===
cameraMatrix = np.load("camera_matrix.npy")
distCoeffs = np.load("dist_coeffs.npy")

# === ArUco setup ===
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_length = 0.045  # u metrima

# === VideoCapture setup ===
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("[ERROR] Kamera se ne može otvoriti.")
    exit()

# === Frame buffer ===
frame_buffer = deque(maxlen=1)
stop_threads = False

def camera_reader():
    while not stop_threads:
        ret, frame = cap.read()
        if ret:
            frame_buffer.append(frame)

cam_thread = threading.Thread(target=camera_reader, daemon=True)
cam_thread.start()

def get_corner_position(rvec, tvec, marker_length, corner_index):
    half_length = marker_length / 2
    local_corners = np.array([
        [-half_length, half_length, 0],
        [half_length, half_length, 0],
        [half_length, -half_length, 0],
        [-half_length, -half_length, 0]
    ])
    corner_local = local_corners[corner_index]
    R, _ = cv2.Rodrigues(rvec)
    corner_world = tvec + np.dot(R, corner_local)
    return corner_world

# === Inicijalizacija ===
prev_position = None
last_valid_position = None
tolerance = 0.1  # mm
frame_count = 0
DETECT_EVERY = 4

# === Glavna petlja ===
while True:
    t_start = time.time()
    frame_count += 1

    if not frame_buffer:
        time.sleep(0.005)
        continue

    frame = frame_buffer[-1].copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_positions = {}

    if frame_count % DETECT_EVERY == 0:
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, cameraMatrix, distCoeffs)

            for i, marker_id in enumerate(ids.flatten()):
                tvec = tvecs[i][0]
                rvec = rvecs[i][0]

                position = get_corner_position(rvec, tvec, marker_length, 0) if marker_id in [0, 2] else tvec
                position_mm = position * 1000
                marker_positions[marker_id] = position_mm

                # Debug
                # cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.03)
                # cv2.putText(frame, f"ID {marker_id}: {position_mm.round(1)} mm",
                #             (10, 60 + 30 * int(marker_id)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if all(k in marker_positions for k in [0, 1, 2]):
                rub_lijevi = marker_positions[0]
                rub_desni = marker_positions[2]
                marker = marker_positions[1]

                duzina_staze = np.linalg.norm(rub_desni - rub_lijevi)
                vektor_staze = rub_desni - rub_lijevi
                vektor_objekta = marker - rub_lijevi
                projekcija = np.dot(vektor_objekta, vektor_staze) / np.linalg.norm(vektor_staze)
                pozicija_markera = projekcija
                relativna_pozicija = np.clip(pozicija_markera / duzina_staze, 0, 1)

                last_valid_position = pozicija_markera

    if last_valid_position is not None:
        if prev_position is None or abs(last_valid_position - prev_position) > tolerance:
            write_real_async(DB_NUMBER, START_ADDRESS, float(last_valid_position))
            prev_position = last_valid_position

        cv2.putText(frame, f"Pozicija: {last_valid_position:.1f} mm ", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"Duzina staze: {duzina_staze:.1f} mm ", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    total = (time.time() - t_start) * 1000
    cv2.putText(frame, f"Ciklus: {total:.1f} ms", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1)

    cv2.imshow("Aruco Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Čišćenje ===
stop_threads = True
cam_thread.join()
cap.release()
cv2.destroyAllWindows()
client.disconnect()
