import snap7
from snap7.util import *
from snap7.type import *
import cv2
import cv2.aruco as aruco
import numpy as np
import struct

# === Parametri PLC-a ===
PLC_IP = '192.168.1.3'
RACK = 0
SLOT = 1
DB_NUMBER = 17     # Data Block u PLC-u gdje se šalje pozicija
START_ADDRESS = 0 # Offset unutar DB-a

# === Poveži se s PLC-om ===
client = snap7.client.Client()
client.connect(PLC_IP, RACK, SLOT)

if client.get_connected():
    print("[INFO] Povezan s PLC-om ✅")
else:
    print("[ERROR] Neuspješno spajanje s PLC-om ❌")
    exit()

# === Funkcija za slanje real broja (float) u PLC ===
def write_real(db_number, start, value):
    data = struct.pack('>f', value)  # Siemens koristi Big Endian za REAL
    client.db_write(db_number, start, data)

# === Učitaj kalibracijske podatke kamere ===
cameraMatrix = np.load("camera_matrix.npy")
distCoeffs = np.load("dist_coeffs.npy")

# === Podesi dictionary i detektor ===
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# === Veličina markera u metrima ===
marker_length = 0.045  # 45 mm

# === Pokreni kameru ===
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("[ERROR] Kamera se ne može otvoriti.")
    exit()

# === Funkcija za izračun 3D pozicije određenog ugla markera ===
def get_corner_position(rvec, tvec, marker_length, corner_index):
    half_length = marker_length / 2
    local_corners = np.array([
        [-half_length, half_length, 0],   # top-left
        [half_length, half_length, 0],    # top-right
        [half_length, -half_length, 0],   # bottom-right
        [-half_length, -half_length, 0]   # bottom-left
    ])
    corner_local = local_corners[corner_index]
    R, _ = cv2.Rodrigues(rvec)
    corner_world = tvec + np.dot(R, corner_local)
    return corner_world

# === Glavna petlja ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, cameraMatrix, distCoeffs)
        marker_positions = {}

        for i, marker_id in enumerate(ids.flatten()):
            tvec = tvecs[i][0]
            rvec = rvecs[i][0]

            if marker_id in [0, 2]:
                position = get_corner_position(rvec, tvec, marker_length, corner_index=0)
            else:
                position = tvec

            position_mm = position * 1000
            marker_positions[marker_id] = position_mm

            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.03)
            cv2.putText(frame, f"ID {marker_id}: {position_mm.round(1)} mm",
                        (10, 60 + 30 * int(marker_id)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # === Ako su detektovani svi markeri (0,1,2) izračunaj poziciju ===
        if all(k in marker_positions for k in [0, 1, 2]):
            rub_lijevi = marker_positions[0]
            rub_desni = marker_positions[2]
            marker = marker_positions[1]

            duzina_staze = np.linalg.norm(rub_desni - rub_lijevi)
            vektor_staze = rub_desni - rub_lijevi
            vektor_objekta = marker - rub_lijevi
            projekcija = np.dot(vektor_objekta, vektor_staze) / np.linalg.norm(vektor_staze)
            pozicija_markera = projekcija  # u mm
            relativna_pozicija = np.clip(pozicija_markera / duzina_staze, 0, 1)

            # === Pošalji poziciju u PLC ===
            write_real(DB_NUMBER, START_ADDRESS, float(pozicija_markera))

            cv2.putText(frame, f"Marker ID 1: {pozicija_markera:.1f} mm / {duzina_staze:.1f} mm  ({relativna_pozicija * 100:.1f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Aruco Tracker (mm)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.disconnect()