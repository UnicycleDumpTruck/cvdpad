from time import sleep
import cv2

from resize import ResizeWithAspectRatio
import numpy as np
from button import Button


first_frame = None

video = cv2.VideoCapture(0)
sleep(2)

up_button = Button(214, 0, 214, 240, "w", "Up")
left_button = Button(0, 240, 214, 240, "a", "Left")
right_button = Button(428, 240, 214, 240, "d", "Right")
down_button = Button(214, 480, 214, 240, "s", "Down")
a_button = Button(1066, 480, 214, 240, "z", "A")
b_button = Button(1066, 0, 214, 240, "x", "B")

buttons = [
    up_button,
    left_button,
    right_button,
    down_button,
    a_button,
    b_button,
]

while True:
    check, color_frame = video.read()
    # status = 0
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    for button in buttons:
        cv2.rectangle(
            color_frame, button.upper_left, button.lower_right, (0, 255, 0), 4
        )

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=3)

    (cnts, _) = cv2.findContours(
        thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # contour_center_x = x + int(w / 2)
        # contour_center_y = y + int(h / 2)

        for button in buttons:
            # if button.containsPoint(contour_center_x, contour_center_y):
            if button.intersectsRect(x, y, w, y):
                button.pressButton()
                break

    # for loop ends here

    # cv2.imshow("Gray Frame", ResizeWithAspectRatio(gray, width=640))
    # cv2.imshow("Delta Frame", ResizeWithAspectRatio(delta_frame, width=640))
    # cv2.imshow("Threshold Frame", ResizeWithAspectRatio(thresh_frame, width=640))
    cv2.imshow("Color Frame", ResizeWithAspectRatio(color_frame, width=640))

    # Wait for quit
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# while loop ends here


video.release()
cv2.destroyAllWindows()
