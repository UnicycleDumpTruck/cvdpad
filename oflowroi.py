import cv2 as cv  # type: ignore
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import mode
from argparse import ArgumentParser

from pyautogui import keyDown, keyUp

from resize import resizeWithAspectRatio

current_key_down = None

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "-rec", "--record", default=False, action="store_true", help="Record?"
    )
    ap.add_argument(
        "-pscale",
        "--pyr_scale",
        default=0.5,
        type=float,
        help="Image scale (<1) to build pyramids for each image",
    )
    ap.add_argument(
        "-l", "--levels", default=3, type=int, help="Number of pyramid layers"
    )
    ap.add_argument(
        "-w", "--winsize", default=15, type=int, help="Averaging window size"
    )
    ap.add_argument(
        "-i",
        "--iterations",
        default=3,
        type=int,
        help="Number of iterations the algorithm does at each pyramid level",
    )
    ap.add_argument(
        "-pn",
        "--poly_n",
        default=5,
        type=int,
        help="Size of the pixel neighborhood used to find polynomial expansion in each pixel",
    )
    ap.add_argument(
        "-psigma",
        "--poly_sigma",
        default=1.1,
        type=float,
        help="Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion",
    )
    ap.add_argument(
        "-th",
        "--threshold",
        default=5.0,
        type=float,
        help="Threshold value for magnitude",
    )
    ap.add_argument(
        "-p", "--plot", default=False, action="store_true", help="Plot accumulators?"
    )
    ap.add_argument(
        "-rgb", "--rgb", default=False, action="store_true", help="Show RGB mask?"
    )
    ap.add_argument(
        "-s",
        "--size",
        default=10,
        type=int,
        help="Size of accumulator for directions map",
    )

    args = vars(ap.parse_args())

    directions_map = np.zeros([args["size"], 5])

    cap = cv.VideoCapture(0)
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    # if args['record']:
    #     h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    #     w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    #     codec = cv.VideoWriter_fourcc(*'MPEG')
    #     out = cv.VideoWriter('out.avi', codec, 10.0, (w, h))

    # if args['plot']:
    #     plt.ion()

    frame_previous = resizeWithAspectRatio(cap.read()[1], width=320)
    gray_previous = cv.cvtColor(frame_previous, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame_previous)
    hsv[:, :, 1] = 255
    param = {
        "pyr_scale": args["pyr_scale"],
        "levels": args["levels"],
        "winsize": args["winsize"],
        "iterations": args["iterations"],
        "poly_n": args["poly_n"],
        "poly_sigma": args["poly_sigma"],
        "flags": cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
    }

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        gray = cv.cvtColor(resizeWithAspectRatio(frame, width=320), cv.COLOR_BGR2GRAY)
        # print(int(4/5*h), w, h) #[0:int(4/5*h), w:h]

        flow = cv.calcOpticalFlowFarneback(gray_previous, gray, None, **param)

        # optical_flow = cv.optflow.DualTVL1OpticalFlow_create(nscales=1, epsilon=0.05, warps=1)
        # flow = optical_flow.calc(gray_previous, gray, None)

        mag, ang = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
        ang_180 = ang / 2
        gray_previous = gray

        move_sense = ang[mag > args["threshold"]]
        move_mode = mode(move_sense)[0]

        if 10 < move_mode <= 100:
            directions_map[-1, 0] = 1
            directions_map[-1, 1:] = 0
            directions_map = np.roll(directions_map, -1, axis=0)
        elif 100 < move_mode <= 190:
            directions_map[-1, 1] = 1
            directions_map[-1, :1] = 0
            directions_map[-1, 2:] = 0
            directions_map = np.roll(directions_map, -1, axis=0)
        elif 190 < move_mode <= 280:
            directions_map[-1, 2] = 1
            directions_map[-1, :2] = 0
            directions_map[-1, 3:] = 0
            directions_map = np.roll(directions_map, -1, axis=0)
        elif 280 < move_mode or move_mode < 10:
            directions_map[-1, 3] = 1
            directions_map[-1, :3] = 0
            directions_map[-1, 4:] = 0
            directions_map = np.roll(directions_map, -1, axis=0)
        else:
            directions_map[-1, -1] = 1
            directions_map[-1, :-1] = 0
            directions_map = np.roll(directions_map, 1, axis=0)

        if args["plot"]:
            plt.clf()
            plt.plot(directions_map[:, 0], label="Down")
            plt.plot(directions_map[:, 1], label="Right")
            plt.plot(directions_map[:, 2], label="Up")
            plt.plot(directions_map[:, 3], label="Left")
            plt.plot(directions_map[:, 4], label="Waiting")
            plt.legend(loc=2)
            plt.pause(1e-5)
            plt.show()

        loc = directions_map.mean(axis=0).argmax()
        if loc == 0:
            text = "Down"
            if current_key_down != "s":
                if current_key_down:
                    keyUp(current_key_down)
                current_key_down = "s"
                keyDown("s")
        elif loc == 1:
            text = "Right"
            if current_key_down != "d":
                if current_key_down:
                    keyUp(current_key_down)
                current_key_down = "d"
                keyDown("d")
        elif loc == 2:
            text = "Up"
            if current_key_down != "w":
                if current_key_down:
                    keyUp(current_key_down)
                current_key_down = "w"
                keyDown("w")
        elif loc == 3:
            text = "Left"
            if current_key_down != "a":
                if current_key_down:
                    keyUp(current_key_down)
                current_key_down = "a"
                keyDown("a")
        else:
            text = ""
            if current_key_down:
                keyUp(current_key_down)
                current_key_down = None

        hsv[:, :, 0] = ang_180
        hsv[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        # rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        gray = cv.flip(gray, 1)
        cv.putText(
            gray,
            text,
            (30, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            frame.shape[1] / 500,
            (0, 0, 255),
            2,
        )

        k = cv.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        # if args['record']:
        #     out.write(frame)
        # if args['rgb']:
        #    cv.imshow('Mask', rgb)
        cv.imshow("Gray", gray)
        k = cv.waitKey(1) & 0xFF
        if k == ord("q"):
            break

    cap.release()
    # if args["record"]:
    #     out.release()
    if args["plot"]:
        plt.ioff()
    cv.destroyAllWindows()
