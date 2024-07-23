import cv2 as cv
import numpy as np
from pioneer_sdk import Camera
from pioneer_sdk import Pioneer


def load_coefficients(path):
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    camera_matrix = cv_file.getNode("Camera_Matrix").mat()
    dist_coeffs = cv_file.getNode("Distortion_Coefficients").mat()

    cv_file.release()
    return camera_matrix, dist_coeffs


# Dictionary of aruco-markers2
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
# Parameters for marker detection
aruco_params = cv.aruco.DetectorParameters()
aruco_detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

# Load camera matrix and distortion coefficients from file
camera_matrix, dist_coeffs = load_coefficients("out_camera_data.yml")

size_of_marker = 0.05  # side length in meters

# Coordinates of marker corners
points_of_marker = np.array(
    [
        (size_of_marker / 2, -size_of_marker / 2, 0),
        (-size_of_marker / 2, -size_of_marker / 2, 0),
        (-size_of_marker / 2, size_of_marker / 2, 0),
        (size_of_marker / 2, size_of_marker / 2, 0),
    ]
)

# Загрузка накладываемого изоюражения
logo = cv.imread('logo.png', cv.IMREAD_UNCHANGED)
logo_height, logo_width = logo.shape[:2]
# Преобразуйте изображение в формат BGRA (синий, зеленый, красный, альфа)
logo = cv.cvtColor(logo, cv.COLOR_BGR2BGRA)
# Установите альфа-канал для всего изображения
logo[:, :, 3] = 255


if __name__ == "__main__":
    # Создание и полёт дрона
    dron = Pioneer(ip='127.0.0.1', mavlink_port=8000)
    dron.arm()
    dron.takeoff()
    # dron.go_to_local_point(x=1, y=0, z=1, yaw=0)
    dron.set_manual_speed_body_fixed(vx=0, vy=1, vz=0, yaw_rate=1)

    # Connect to the drone camera
    camera = Camera(ip='127.0.0.1', port=18000)
    while True:
        frame = camera.get_cv_frame()  # Get frame (already decoded)
        if frame is not None:
            # Detect markers
            corners, ids, rejected = aruco_detector.detectMarkers(frame)
            if corners:
                cv.aruco.drawDetectedMarkers(frame, corners, ids)

                # Определение координат углов первого обнаруженного маркера
                pts1 = np.float32(corners[0].reshape(4, 2))

                # Определение координат углов логотипа в его собственном пространстве
                pts2 = np.float32([[0, 0], [logo_width, 0],
                                   [logo_width, logo_height],
                                   [0, logo_height]])

                # Создание матрицы перспективного преобразования
                M = cv.getPerspectiveTransform(pts2, pts1)
                # Применение перспективного преобразования к логотипу
                warp_logo = cv.warpPerspective(logo, M, (frame.shape[1], frame.shape[0]))

                # Наложение логотипа на изображение с камеры
                alpha_channel = warp_logo[:, :, 3] / 255  # convert from 0-255 to 0.0-1.0
                overlay_colors = warp_logo[:, :, :3]
                alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
                h, w = warp_logo.shape[:2]
                background_subsection = frame[0:h, 0:w]
                composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask

                # overwrite the section of the background image that has been updated
                frame[0:h, 0:w] = composite


                # Calculate pose for first detected marker
                success, rvecs, tvecs = cv.solvePnP(
                    points_of_marker, corners[0], camera_matrix, dist_coeffs
                )
                coordinates = [tvecs.item(0), tvecs.item(1), tvecs.item(2)]
                # Draw axes
                cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.1)


            else:
                pass

            cv.imshow("video", frame)  # Show an image on the screen

        if cv.waitKey(1) == 27:  # Exit if the ESC key is pressed
            break

    cv.destroyAllWindows()  # Close all opened openCV windows
