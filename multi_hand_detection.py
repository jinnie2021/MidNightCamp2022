import mediapipe as mp
import cv2
import numpy as np
import socket

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# Set webcam resolution
# WIDTH, HEIGHT, Z_CONVERSION = 1280, 720, 10
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
# Check webcam resolution
WIDTH, HEIGHT = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
Z_CONVERSION = 10
print("Width: {}, Height: {}, Z_conversion: {}".format(WIDTH, HEIGHT, Z_CONVERSION))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 8052)

gesture_dict = {0: 'Grab (Drag and Drop)', # key point coordinate: landmark #9
                1: 'swipe', # key point coordinate: landmark #12
                2: 'zoom', # key point coordinate: landmark #8 for left and right hand
                3: 'click'} # key point coordinate: landmark #8

def fingers_up(hand_Lms):
    # index, middle, ring, pinky finger
    # False: when fingers are folded
    # True: when fingers are up (not folded)
    up_fingers = [False, False, False, False]
    up_fingers[0] = hand_Lms.landmark[8].y < hand_Lms.landmark[6].y
    up_fingers[1] = hand_Lms.landmark[12].y < hand_Lms.landmark[10].y
    up_fingers[2] = hand_Lms.landmark[16].y < hand_Lms.landmark[14].y
    up_fingers[3] = hand_Lms.landmark[20].y < hand_Lms.landmark[17].y
    return up_fingers

# To show hands in Unity window
# Method1 ] convert scale first with function scale_conversion (which will return the coordinates of webcam window),
# then, adjust coordinates in unity script
# Method2 ] send coordinate in mediapipe format (normalized in [0.0, 1.0])
# then, adjust coordinates in unity script
# I tried both, Method2 is more manageable

# Hand landmark scale conversion depending on resolution of the webcam
# def scale_conversion(hand_landmark_list):
#     hand_array = np.array(hand_landmark_list) # convert to array
#     hand_array = hand_array.reshape(-1, 3) # to split x, y, z easily
#
#     # convert x, y, z in webcam's WIDTH, HEIGHT and assigned Z_CONVERSION
#     # Caution! slicing processed in shallow copy, which means the original value doesn't change
#     hand_array_x = hand_array[:, 0] * WIDTH
#     hand_array_y = hand_array[:, 1] * HEIGHT
#     hand_array_z = hand_array[:, 2] * Z_CONVERSION
#
#     # convert into original format
#     hand_array_xyz = np.column_stack([hand_array_x.T, hand_array_y.T, hand_array_z.T]) # convert each x, y, z in 1 column and stack in column
#     converted_landmarks = hand_array_xyz.flatten() # convert to 1 row
#     converted_landmarks = converted_landmarks.tolist() # convert to list
#     return converted_landmarks


# -------------------------------------------------------------------
# The program starts from here!
with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, image = cap.read()
        img_h, img_w = image.shape[:2]

        if not ret:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        gesture_detection_result = [0, 0, 0, 0] # detection result of each gesture
        data = [] # [[left hand landmarks], [right hand landmarks], [gesture detection result]]
        if results.multi_hand_landmarks:
            # multiple hand detection
            data.append([]) # to save left hand landmarks
            data.append([]) # to save right hand landmarks
            prev_Left, prev_Right = False, False # save previous position to detect both hands
            for cnt, handLms in enumerate(results.multi_hand_landmarks):
                # Draw hands
                mp_drawing.draw_landmarks(
                    image, handLms, mp_hands.HAND_CONNECTIONS
                )


                # 각 손가락이 펴졌는지 접혔는지 확인
                fingers_unfolded = fingers_up(handLms)
                # print(fingers_unfolded)

                # Grab (Drag and Drop)
                grab_thumb_x, grab_thumb_y = handLms.landmark[4].x, handLms.landmark[4].y
                grab_middle_x, grab_middle_y = handLms.landmark[11].x, handLms.landmark[11].y
                dist = np.sqrt((grab_thumb_x - grab_middle_x) ** 2 + (grab_thumb_y - grab_middle_y) ** 2)
                # print(dist)
                if dist <= 0.1:
                    gesture_detection_result[0] = 1
                    # print('Grabbed')
                else:
                    gesture_detection_result[0] = 0
                    # print('Not grabbed')

                # Swipe
                if fingers_unfolded[0] and fingers_unfolded[1] and fingers_unfolded[2] and fingers_unfolded[3]:
                    gesture_detection_result[1] = 1
                else:
                    gesture_detection_result[1] = 0

                # Click
                if fingers_unfolded[0] and fingers_unfolded[1] and not fingers_unfolded[2] and not fingers_unfolded[3]:
                    v_finger_x, v_finger_y, v_finger_z = handLms.landmark[8].x - handLms.landmark[5].x, \
                                                         handLms.landmark[8].y - handLms.landmark[5].y, \
                                                         handLms.landmark[8].z - handLms.landmark[5].z
                    v_finger = np.array([v_finger_x, v_finger_y, v_finger_z])
                    v_palm_x, v_palm_y, v_palm_z = handLms.landmark[5].x - handLms.landmark[0].x, \
                                                   handLms.landmark[5].y - handLms.landmark[0].y, \
                                                   handLms.landmark[5].z - handLms.landmark[0].z
                    v_palm = np.array([v_palm_x, v_palm_y, v_palm_z])

                    # Get angle from vectors
                    # v_finger: from index finger to first top knuckle in palm
                    # v_palm: from first top knuckle in palm to wrist
                    angle = np.arccos(np.dot(v_finger, v_palm))
                    # print(v_finger)
                    # print(v_palm)
                    # print(angle)

                    if angle > 1.505:
                        gesture_detection_result[3] = 1
                        # print('Clicked')
                    else:
                        gesture_detection_result[3] = 0
                        # print('Not Clicked')


                # Zoom In
                # 왼손인지 오른손인지 확인
                Left = handLms.landmark[4].x > handLms.landmark[20].x
                Right = handLms.landmark[4].x < handLms.landmark[20].x
                if Left: # save left hand landmarks
                    hand = []
                    for idx, lm in enumerate(handLms.landmark):
                        hand.extend([lm.x, 1 - lm.y, lm.z])
                    data[0].extend(hand)
                elif Right: # save right hand landmarks
                    hand = []
                    for idx, lm in enumerate(handLms.landmark):
                        hand.extend([lm.x, 1 - lm.y, lm.z])
                    data[1].extend(hand)

                # Both hands are detected
                if (Left and prev_Right) or (prev_Left and Right):
                    # left hand thumb, index
                    l_thumb_x, l_thumb_y = data[0][4*3] * img_w, data[0][4*3 + 1] * img_h
                    l_index_x, l_index_y = data[0][8*3] * img_w, data[0][8*3 + 1] * img_h
                    # right hand thumb, index
                    r_thumb_x, r_thumb_y = data[1][4*3] * img_w, data[1][4*3 + 1] * img_h
                    r_index_x, r_index_y = data[1][8*3] * img_w, data[1][8*3 + 1] * img_h

                    # distance between thumb and index finger
                    dist_left = int(np.sqrt((l_thumb_x - l_index_x) ** 2 + (l_thumb_y - l_index_y) ** 2))
                    dist_right = int(np.sqrt((r_thumb_x - r_index_x) ** 2 + (r_thumb_y - r_index_y) ** 2))
                    # print(dist_left, dist_right)

                    # the index and middle fingers are close enough for both hands
                    if dist_left < 20 and dist_right < 20:
                        gesture_detection_result[2] = 1
                        # print('zoom in')
                    else:
                        gesture_detection_result[2] = 0
                        # print('zoom out')

                prev_Left, prev_Right = Left, Right # save current hand

            # Hand landmark scale conversion
            # data[0] = scale_conversion(data[0])
            # data[1] = scale_conversion(data[1])

            # Add detection result to data which will be transmitted
            data.append(gesture_detection_result)


        # print(gesture_detection_result)
        print(data)

        # Send gesture detection result with UDP
        sock.sendto(str.encode(str(data)), serverAddressPort)

        # Show window
        # image = cv2.resize(image, (0, 0), None, 0.5, 0.5) # Optional: Show window in smaller 1/4 size to see with unity window
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('image',  image)

        # Close window with esc keystroke
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
