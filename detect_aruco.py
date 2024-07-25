import math

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


def get_centre_aruco(corners):
    # Find center of first aruco marker on screen
    x_center = int(
        (
                corners[0][0]
                + corners[1][0]
                + corners[2][0]
                + corners[3][0]
        )
        // 4
    )
    y_center = int(
        (
                corners[0][1]
                + corners[1][1]
                + corners[2][1]
                + corners[3][1]
        )
        // 4
    )
    return x_center, y_center


def get_dist(corners, x=640/2, y=480/2):
    x_center, y_center = get_centre_aruco(corners)
    return ((x_center - x)**2 + (y_center - y)**2)**0.5


def get_next_goal(goal, ids):
    if max(ids) <= goal:
        return max(ids)
    else:
        return min([id for id in ids if id > goal])

def search_aruco(dron, time_without_goal, corners_of_last_goal=None, waiting_time=100):
    time_without_goal += 1
    if time_without_goal > waiting_time:
        # Search new aruco
        dron.set_manual_speed_body_fixed(vx=0, vy=0, vz=0, yaw_rate=0.3)
    elif time_without_goal > waiting_time/10 and corners_of_last_goal is not None:
        # Search last aruco
        go_to_aruco(dron, corners_of_last_goal, speed=1 / math.log10(time_without_goal + 10))
    return time_without_goal

def go_to_aruco(dron, corners, speed=1., wide=640, high=480):
    x_center, y_center = get_centre_aruco(corners)

    yaw_rate = 0
    # set a non-zero rotation speed if the center of the marker is on the side of the screen
    if x_center < wide / 2:
        yaw_rate = -3
    elif x_center > wide / 2:
        yaw_rate = 3
    yaw_rate *= 0.005 + abs(x_center - wide / 2) / frame.shape[1] / 2

    v_y = 0
    if y_center < high / 2:
        v_y = 3
    elif y_center > high / 2:
        v_y = -1
    v_y *= 0.01 + abs(y_center - high / 2) / high / 2

    dron.set_manual_speed_body_fixed(vx=0, vy=v_y*speed, vz=0, yaw_rate=yaw_rate*speed)


# Dictionary of aruco-markers2
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_1000)
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
    dron = Pioneer(ip='127.0.0.1',
                   mavlink_port=8000,
                   logger=False,
                   log_connection=False)
    dron.arm()
    dron.takeoff()
    dron.go_to_local_point(x=0, y=0, z=1.8, yaw=0)
    # dron.set_manual_speed_body_fixed(vx=0, vy=1, vz=0, yaw_rate=1)

    # Connect to the drone camera
    camera = Camera(ip='127.0.0.1',
                    port=18000,
                    log_connection=False)
    goal_id = goal_index = None
    corners_of_goal = None
    # last_info_about_aruco = {}
    visited_aruco_ids = set()
    # time_out_of_frame = {}
    time_without_goal = 0
    while True:
        frame = camera.get_cv_frame()  # Get frame (already decoded)
        if frame is not None:
            # Auto contrast
            alpha = 1.8 # Contrast control (1.0-3.0)
            beta = 20  # Brightness control (0-100)
            frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)

            # denoising of image saving it into dst image
            frame = cv.fastNlMeansDenoisingColored(frame, None,
                                                   10, 10, 7, 5)

            # Detect markers
            corners, ids, rejected = aruco_detector.detectMarkers(frame)
            if corners:
                cv.aruco.drawDetectedMarkers(frame, corners, ids)
                ids, corners = zip(*sorted(zip(ids, corners), key=lambda x: x[0]))
                ids = [id.tolist()[0] for id in ids]
                # for id, corner in zip(ids, corners):
                #     last_info_about_aruco[id] = corner
                #     if id not in time_out_of_frame:
                #         time_out_of_frame[id] = 0
                # for id in time_out_of_frame:
                #     if id not in ids:
                #         time_out_of_frame[id] += 1
                #     else:
                #         time_out_of_frame[id] = 0
                unsettled_goals = [id for id in ids if id not in visited_aruco_ids]
                if len(unsettled_goals) != 0:
                    goal_id = min(unsettled_goals)
                    goal_index = ids.index(goal_id)
                    corners_of_goal = corners[goal_index][0]


                    time_without_goal = 0

                    # Определение координат углов первого обнаруженного маркера
                    pts1 = np.float32(corners_of_goal)

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

                    x_center, y_center = get_centre_aruco(corners_of_goal)
                    dot_size = 5
                    # Draw red square in center of aruco marker
                    frame[
                        y_center - dot_size: y_center + dot_size,
                        x_center - dot_size: x_center + dot_size,
                    ] = [255, 0, 255]
                    # Draw markers

                    # Calculate pose for first detected marker
                    success, rvecs, tvecs = cv.solvePnP(
                        points_of_marker, corners[0], camera_matrix, dist_coeffs
                    )
                    coordinates = [tvecs.item(0), tvecs.item(1), tvecs.item(2)]

                    # Draw axes
                    cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.1)

                    if get_dist(corners_of_goal) <= 50:
                        print(f'Success! visited aruco with the id {goal_id}')
                        visited_aruco_ids.add(goal_id)
                        goal_id = goal_index = None
                    else:
                        go_to_aruco(dron, corners_of_goal)
                else:
                    time_without_goal = search_aruco(dron, time_without_goal, corners_of_goal)
            else:
                time_without_goal = search_aruco(dron, time_without_goal, corners_of_goal)


            cv.imshow("video", frame)  # Show an image on the screen

        if cv.waitKey(1) == 27:  # Exit if the ESC key is pressed
            break

    cv.destroyAllWindows()  # Close all opened openCV windows
