import math
class Point:
    """ Point class represents and manipulates x,y coords. """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "x: {}, y: {}".format(self.x, self.y)

    def __repr__(self):
        return str(self)

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)