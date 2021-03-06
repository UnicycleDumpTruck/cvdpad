from datetime import datetime
from time import sleep
import cv2
import pandas
from resize import ResizeWithAspectRatio
import numpy as np
from button import Button
from pyautogui import keyDown, keyUp

first_frame = None
status_list = [None, None]
time_stamp = [datetime]
df = pandas.DataFrame(columns=["Start", "End"])

video = cv2.VideoCapture(0)
sleep(2)

up_button_rect = Button(214, 0, 214, 240, "w", "Up")
left_button_rect = Button(0, 240, 214, 240, "a", "Left")
right_button_rect = Button(428, 240, 214, 240, "d", "Right")
down_button_rect = Button(214, 480, 214, 240, "s", "Down")
a_button_rect = Button(1066, 480, 214, 240, "z", "A")
b_button_rect = Button(1066, 0, 214, 240, "x", "B")

buttons = [
    up_button_rect,
    left_button_rect,
    right_button_rect,
    down_button_rect,
    a_button_rect,
    b_button_rect,
]

while True:
    check, color_frame = video.read()
    status = 0
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    for button in buttons:
        cv2.rectangle(
            color_frame, button.upper_left, button.lower_right, (0, 255, 0), 4
        )

    # cv2.rectangle(color_frame, (214, 0), (428, 240), (0, 255, 0), 4)  # up
    # cv2.rectangle(color_frame, (0, 240), (214, 480), (0, 255, 0), 4)  # left
    # cv2.rectangle(color_frame, (428, 240), (640, 480), (0, 255, 0), 4)  # right
    # cv2.rectangle(color_frame, (214, 480), (428, 720), (0, 255, 0), 4)  # down
    # cv2.rectangle(color_frame, (1066, 480), (1280, 720), (0, 255, 0), 4)  # b
    # cv2.rectangle(color_frame, (1066, 0), (1280, 240), (0, 255, 0), 4)  # a

    delta_frame = cv2.absdiff(first_frame, gray)
    # print(delta_frame[1])
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=3)

    (cnts, _) = cv2.findContours(
        thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if cnts:
        keyUp("s")
        keyDown("w")
    else:
        keyUp("w")
        keyDown("s")

    # for contour in cnts:
    #     if cv2.contourArea(contour) < 10000:
    #         continue
    #     # print(contour)
    #     status = 1
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #     cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     contour_center_x = x + int(w / 2)
    #     contour_center_y = y + int(h / 2)

    # for button in buttons:
    #     # if button.containsPoint(contour_center_x, contour_center_y):
    #     if button.intersectsRect(x, y, w, y):
    #         button.pressButton()
    #         break

    # for loop ends here

    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        time_stamp.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        time_stamp.append(datetime.now())

    # cv2.imshow("Gray Frame", ResizeWithAspectRatio(gray, width=640))
    # cv2.imshow("Delta Frame", ResizeWithAspectRatio(delta_frame, width=640))
    cv2.imshow("Threshold Frame", ResizeWithAspectRatio(thresh_frame, width=640))
    # cv2.imshow("Color Frame", ResizeWithAspectRatio(color_frame, width=640))

    # Wait for quit
    key = cv2.waitKey(1)
    if key == ord("q"):
        if status == 1:
            time_stamp.append(datetime.now())
        break

# while loop ends here

# print(status_list)

for i in range(0, len(time_stamp) - 1, 2):
    df = df.append(
        {"Start": time_stamp[i], "End": time_stamp[i + 1]}, ignore_index=True
    )

df.to_csv("All_Time_Stamp.csv")
video.release()
cv2.destroyAllWindows()
