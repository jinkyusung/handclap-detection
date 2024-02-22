import cv2
import mediapipe as mp

import metric
import misc


def pattern(cap, alpha, beta):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    distances = []
    included_angles = []

    left_fingers_mean = []
    right_fingers_mean = []

    left_z_directs = []
    right_z_directs = []

    left_y_directs = []
    right_y_directs = []

    left_x_directs = []
    right_x_directs = []


    with mp_hands.Hands(
        min_detection_confidence=alpha,
        min_tracking_confidence=beta) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                left_hand_landmark_0 = None
                right_hand_landmark_0 = None
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    if handedness == 'Left':
                        # Left-hand drawing options
                        landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=4)
                        connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)

                        # Distances
                        left_hand_landmark_0 = hand_landmarks.landmark[0]
                        
                        # Degree of grasping
                        left_each_angles = metric.each_angles(hand_landmarks.landmark)
                        left_mean = metric.mean(left_each_angles)
                        left_fingers_mean.append(left_mean)

                        # Left palm direction by the normal vector
                        left_normal_vector = metric.normal_vector(hand_landmarks.landmark)
                        left_z_direct = metric.included_angle(left_normal_vector, (0, 0, 1))
                        left_y_direct = metric.included_angle(left_normal_vector, (0, 1, 0))
                        left_x_direct = metric.included_angle(left_normal_vector, (1, 0, 0))

                        left_z_directs.append(left_z_direct)
                        left_y_directs.append(left_y_direct)
                        left_x_directs.append(left_x_direct)

                    elif handedness == 'Right':
                        # Right-hand drawing options
                        landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4)
                        connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)

                        # Distances
                        right_hand_landmark_0 = hand_landmarks.landmark[0]

                        # Degree of grasping
                        right_each_angles = metric.each_angles(hand_landmarks.landmark)
                        right_mean = metric.mean(right_each_angles)
                        right_fingers_mean.append(right_mean)

                        # Right palm direction by the normal vector
                        right_normal_vector = (-1) * metric.normal_vector(hand_landmarks.landmark)
                        right_z_direct = metric.included_angle(right_normal_vector, (0, 0, 1))
                        right_y_direct = metric.included_angle(right_normal_vector, (0, 1, 0))
                        right_x_direct = metric.included_angle(right_normal_vector, (1, 0, 0))

                        right_z_directs.append(right_z_direct)
                        right_y_directs.append(right_y_direct)
                        right_x_directs.append(right_x_direct)

                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec, connection_drawing_spec)

                if left_hand_landmark_0 and right_hand_landmark_0:
                    distance = metric.distance(left_hand_landmark_0, right_hand_landmark_0)
                    distances.append(distance)
                    included_angle = metric.included_angle(left_normal_vector, right_normal_vector)
                    included_angles.append(included_angle)
                    misc.echo('both', distance)
                
                if not right_hand_landmark_0:
                    distances.append(0)
                    included_angles.append(0)
                    right_fingers_mean.append(0)
                    right_z_directs.append(0)
                    right_y_directs.append(0)
                    right_x_directs.append(0)
                    misc.echo('right-miss')
                
                if not left_hand_landmark_0:
                    distances.append(0)
                    included_angles.append(0)
                    left_fingers_mean.append(0)
                    left_z_directs.append(0)
                    left_y_directs.append(0)
                    left_x_directs.append(0)
                    misc.echo('left-miss')
            else:
                distances.append(0)
                included_angles.append(0)
                right_fingers_mean.append(0)
                left_fingers_mean.append(0)
                right_z_directs.append(0)
                right_y_directs.append(0)
                right_x_directs.append(0)
                left_z_directs.append(0)
                left_y_directs.append(0)
                left_x_directs.append(0)
                misc.echo('miss')

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                return distances, included_angles, left_fingers_mean, right_fingers_mean, \
                    left_x_directs, left_y_directs, left_z_directs, right_x_directs, right_y_directs, right_z_directs

    cap.release()
    cv2.destroyAllWindows()
    return distances, included_angles, left_fingers_mean, right_fingers_mean, \
        left_x_directs, left_y_directs, left_z_directs, right_x_directs, right_y_directs, right_z_directs


def outliers(distances, included_angles, left_fingers_mean, right_fingers_mean, left_x_directs, left_y_directs, left_z_directs, right_x_directs, right_y_directs, right_z_directs, t_dist, t_angle, t_curv, t_dir):
    outliers = []
    for i in range(len(distances)):
        distances_window = window(i, i+30, distances)
        angles_window = window(i-30, i+30, included_angles)
        if Checker.distances(distances_window, t_dist) and Checker.included_angles(angles_window, t_angle) \
            and (Checker.included_angles(angles_window, t_angle)):
            outliers.append((i, i+10))
    return outliers


def window(start, end, target):
    return target[max(0, start): min(len(target)-1, end)]


class Checker:
    @staticmethod
    def distances(window, t_dist):
        if not len(window):
            return False
        moving_avg = sum(window) / len(window)
        difference = abs(window[-1] - moving_avg)
        return difference > t_dist

    @staticmethod
    def included_angles(window, t_angle):
        if not len(window):
            return False
        max_angle = max(window)
        return max_angle > t_angle
    