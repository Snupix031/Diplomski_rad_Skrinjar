import cv2
import numpy as np

# === Parametri checkerboard uzorka ===
CHECKERBOARD = (9, 6)  # broj unutarnjih kutova (ne kvadrata!)
square_size = 25  # mm – veličina kvadrata

# Priprema 3D točaka (npr. (0,0,0), (25,0,0), (50,0,0), ...)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Liste za spremanje točaka
objpoints = []  # 3D točke u stvarnom prostoru
imgpoints = []  # 2D točke u slici

# === Otvaranje kamere ===
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 30)  # brže otvaranje kamere na Windowsu

if not cap.isOpened():
    print("[ERROR] Ne mogu otvoriti kameru.")
    exit()

# Opcionalno: postavi nižu rezoluciju radi brzine
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"[INFO] Stvarna rezolucija: {w} x {h}")
img_count = 0

print("📸 Pritisni 's' za spremanje slike, 'q' za kraj...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Problem s čitanjem iz kamere.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pronađi kutove checkerboard uzorka
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret_corners:
        # Poboljšaj preciznost
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Prikaži kutove na slici
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret_corners)

    cv2.imshow('Kalibracija kamere', frame)
    key = cv2.waitKey(1)

    if key == ord('s') and ret_corners:
        print(f"[INFO] Spremljena slika: kalib_{img_count}.png")
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        cv2.imwrite(f"kalib_{img_count}.png", frame)
        img_count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === KALIBRACIJA ===
print("[INFO] Kalibriram kameru na temelju {} slika...".format(len(objpoints)))

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("[INFO]  Kalibracija završena.")
print("Camera matrix:\n", cameraMatrix)
print("Distortion coefficients:\n", distCoeffs)

# Spremi rezultate
np.save("camera_matrix.npy", cameraMatrix)
np.save("dist_coeffs.npy", distCoeffs)
