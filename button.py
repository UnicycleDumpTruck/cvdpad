class button:
    def __init__(self, x, y, w, h, keypress, name):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.keypress = keypress
        self.name = name

    @property
    def upper_left(self):
        return (self.x, self.y)

    @property
    def lower_right(self):
        return (self.x + self.w, self.y + self.h)

