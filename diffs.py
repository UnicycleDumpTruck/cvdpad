from datetime import datetime
from time import sleep
import cv2
import pandas
from resize import ResizeWithAspectRatio
import numpy as np


first_frame = None
status_list = [None, None]
time_stamp = [datetime]
df = pandas.DataFrame(columns=["Start", "End"])

video = cv2.VideoCapture(0)
sleep(2)


while True:
    check, color_frame = video.read()
    status = 0
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    # print(delta_frame[1])
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=3)

    (cnts, _) = cv2.findContours(
        thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        # print(contour)
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        contour_center_x = x + int(w / 2)
        contour_center_y = y + int(h / 2)

        if contour_center_x <= 640 and contour_center_y <= 360:
            # Upper Left
            print("Upper left")
        elif contour_center_x > 640 and contour_center_y <= 360:
            # Upper Right
            print("Upper Right")
        elif contour_center_x <= 640 and contour_center_y > 360:
            # Lower Left
            print("Lower Left")
        elif contour_center_x > 640 and contour_center_y > 360:
            # Lower Right
            print("Lower Right")

    # for loop ends here

    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        time_stamp.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        time_stamp.append(datetime.now())

    cv2.imshow("Gray Frame", ResizeWithAspectRatio(gray, width=640))
    cv2.imshow("Delta Frame", ResizeWithAspectRatio(delta_frame, width=640))
    cv2.imshow("Threshold Frame", ResizeWithAspectRatio(thresh_frame, width=640))
    cv2.imshow("Color Frame", ResizeWithAspectRatio(color_frame, width=640))

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
