import cv2
import cv2.aruco as aruco
import numpy as np

# === Učitaj kalibracijske podatke ===
cameraMatrix = np.load("camera_matrix.npy")
distCoeffs = np.load("dist_coeffs.npy")
calib_size = np.load("calib_resolution.npy")  # npr. [1280, 720]

# === Pokreni kameru ===
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("[ERROR] Kamera se ne može otvoriti.")
    exit()

# === Provjera stvarne rezolucije kamere ===
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Stvarna rezolucija kamere: {frame_width}x{frame_height}")

# === Skaliraj cameraMatrix ako rezolucija nije ista ===
if (frame_width, frame_height) != tuple(calib_size):
    scale_x = frame_width / calib_size[0]
    scale_y = frame_height / calib_size[1]
    cameraMatrix[0, 0] *= scale_x  # fx
    cameraMatrix[0, 2] *= scale_x  # cx
    cameraMatrix[1, 1] *= scale_y  # fy
    cameraMatrix[1, 2] *= scale_y  # cy
    print("[INFO] Skalirana cameraMatrix na novu rezoluciju.")

# === ArUco detekcija ===
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

marker_length = 0.045  # u metrima (npr. 45 mm)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Sharpness filter (izoštravanje slike) ===
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel_sharp)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, cameraMatrix, distCoeffs
        )

        marker_positions = {}
        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]
            marker_positions[marker_id] = tvec  # spremi u metrima

            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.03)
            pos_mm = tvec * 1000
            cv2.putText(frame, f"ID {marker_id}: {pos_mm.round(1)} mm",
                        (10, 40 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

        # === Udaljenost između ID 0 i ID 2 ===
        if 0 in marker_positions and 2 in marker_positions:
            dist_02 = np.linalg.norm(marker_positions[0] - marker_positions[2]) * 1000
            cv2.putText(frame, f"Udaljenost ID0-ID2: {dist_02:.1f} mm",
                        (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 0), 2)

        # === X koordinata ID 1 u odnosu na ID 0 (lijevo-desno) ===
        if 1 in marker_positions and 0 in marker_positions:
            rel_10 = (marker_positions[1] - marker_positions[0]) * 1000  # mm
            x_offset = rel_10[0]  # lijevo/desno pomak

            # Prag za "nultu točku" (±2 mm)
            if abs(x_offset) <= 2:
                cv2.putText(frame, "ID1 na nultoj tocki pozicioniranja",
                            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"ID1 X od ID0: {x_offset:.1f} mm",
                            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)

    cv2.imshow("Aruco Detekcija", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
