from time import sleep
from pyautogui import press, keyDown, keyUp


class Button:
    def __init__(self, x, y, w, h, keypress, name):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.keypress = keypress
        self.name = name

    @property
    def upper_left(self):
        """Return tuple (x,y) of rectangle's upper left corner"""
        return (self.x, self.y)

    @property
    def lower_right(self):
        """Return tuple (x,y) of rectangle's lower right corner"""
        return (self.x + self.w, self.y + self.h)

    def intersectsRect(self, x, y, w, h):
        """Returns true if self rectangle intersects passed rectangle"""
        overlap_x = max(self.x, x)
        overlap_y = max(self.y, y)
        overlap_w = min(self.x + self.w, x + w) - overlap_x
        overlap_h = min(self.y + self.h, y + h) - overlap_y
        if overlap_w < 0 or overlap_h < 0:
            return False
        return True

    def containsPoint(self, x, y):
        """Returns true if self rectangle contains passed point"""
        if (self.x < x < (self.x + self.w)) and (self.y < y < (self.y + self.h)):
            return True
        return False

    def pressButton(self):
        """Conveys button's keypress to system, wherever system focus is."""
        press(self.keypress)
        print(f"{self.keypress} pressed")
        sleep(0)
