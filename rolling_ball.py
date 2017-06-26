import math
import numpy
from numba import jit
from parakeet import jit as jitp
from skimage.morphology import disk
from skimage.filter import rank

"""
Ported to Python from ImageJ's Background Subtractor.
Only works for 8-bit greyscale images currently.
Does not perform shrinking/enlarging for larger radius sizes.

Based on the concept of the rolling ball algorithm described
in Stanley Sternberg's article,
"Biomedical Image Processing", IEEE Computer, January 1983.

Imagine that the 2D grayscale image has a third (height) dimension by the image
value at every point in the image, creating a surface. A ball of given radius
is rolled over the bottom side of this surface; the hull of the volume
reachable by the ball is the background.

http://rsbweb.nih.gov/ij/developer/source/ij/plugin/filter/BackgroundSubtracter.java.html
"""

def rolling_ball_background(array, radius,
                            smoothing=False, jit=True):
    """
    Calculates and subtracts background from array.

    Arguments:
    array - numpy array representing image
    radius - radius of the rolling ball creating the background
    smoothing - Whether the image should be smoothed before creating the
                background.
    """
    ball = RollingBall(radius)
    float_array = rolling_ball_float_background(array, radius,
                                                smoothing, ball, jit)
    offset = 0.5
    final = array - float_array + offset
    #Is this correct?
    final[final < 0 ] = 0
    final[final > 0xffff] = 0xffff
    return final.astype('uint16')


def rolling_ball_float_background(float_array, radius, smoothing,
                                  ball, jit):
    """
    Create background for a float image by rolling a ball over the image
    """
    shrink = ball.shrink_factor > 1

    if smoothing:
        float_array = rank.mean(float_array, selem=disk(3))
    pixels = roll_ball(ball, float_array.astype('float32'), jit)
    return numpy.reshape(pixels, float_array.shape)

@jit('f4[:](f4[:],i2,f4[:,:],f4[:],f4[:])', nopython=True)
def run_balln(z_ball, ball_width, array, pixels, cache):
    height, width = array.shape
    radius = ball_width / 2

    for y in range(-radius, height + radius):
        next_line_to_write_in_cache = (y + radius) % ball_width
        next_line_to_read = y + radius
        if next_line_to_read < height:
            src = next_line_to_read * width
            dest = next_line_to_write_in_cache * width
            for i in range(dest, dest+width):
                cache[i] = pixels[src]
                src += 1
            p = next_line_to_read * width
            for x in range(width):
                pixels[p] = -float(1.7*10**10)
                p += 1
        y_0 = y - radius
        if y_0 < 0:
            y_0 = 0
        y_ball_0 = y_0 - y + radius
        y_end = y + radius
        if y_end >= height:
            y_end = height - 1
        for x in range(-radius, width + radius):
            z = float(1.7*10**10)
            x_0 = x - radius
            if x_0 < 0:
                x_0 = 0
            x_ball_0 = x_0 - x + radius
            x_end = x + radius
            if x_end >= width:
                x_end = width - 1
            y_ball = y_ball_0
            for yp in range(y_0, y_end + 1):
                cache_pointer = (yp % ball_width) * width + x_0
                bp = x_ball_0 + y_ball * ball_width
                for xp in range(x_0, x_end + 1):
                    z_reduced = cache[cache_pointer] - z_ball[bp]
                    if z > z_reduced:
                        z = z_reduced
                    cache_pointer += 1
                    bp += 1
                y_ball += 1

            y_ball = y_ball_0
            for yp in range(y_0, y_end + 1):
                p = x_0 + yp * width
                bp = x_ball_0 + y_ball * ball_width
                for xp in range(x_0, x_end + 1):
                    z_min = z + z_ball[bp]
                    if pixels[p] < z_min:
                        pixels[p] = z_min
                    p += 1
                    bp += 1
                y_ball += 1

    return pixels

@jitp
def run_ballp(z_ball, ball_width, array, pixels, cache):
    height, width = array.shape
    radius = ball_width / 2

    for y in range(-radius, height + radius):
        next_line_to_write_in_cache = (y + radius) % ball_width
        next_line_to_read = y + radius
        if next_line_to_read < height:
            src = next_line_to_read * width
            dest = next_line_to_write_in_cache * width
            for i in range(dest, dest+width):
                cache[i] = pixels[src]
                src += 1
            p = next_line_to_read * width
            for x in range(width):
                pixels[p] = -float(1.7*10**10)
                p += 1
        y_0 = y - radius
        if y_0 < 0:
            y_0 = 0
        y_ball_0 = y_0 - y + radius
        y_end = y + radius
        if y_end >= height:
            y_end = height - 1
        for x in range(-radius, width + radius):
            z = float(1.7*10**10)
            x_0 = x - radius
            if x_0 < 0:
                x_0 = 0
            x_ball_0 = x_0 - x + radius
            x_end = x + radius
            if x_end >= width:
                x_end = width - 1
            y_ball = y_ball_0
            for yp in range(y_0, y_end + 1):
                cache_pointer = (yp % ball_width) * width + x_0
                bp = x_ball_0 + y_ball * ball_width
                for xp in range(x_0, x_end + 1):
                    z_reduced = cache[cache_pointer] - z_ball[bp]
                    if z > z_reduced:
                        z = z_reduced
                    cache_pointer += 1
                    bp += 1
                y_ball += 1

            y_ball = y_ball_0
            for yp in range(y_0, y_end + 1):
                p = x_0 + yp * width
                bp = x_ball_0 + y_ball * ball_width
                for xp in range(x_0, x_end + 1):
                    z_min = z + z_ball[bp]
                    if pixels[p] < z_min:
                        pixels[p] = z_min
                    p += 1
                    bp += 1
                y_ball += 1

    return pixels


def roll_ball(ball, array, jit=""):
    """
    Rolls a filtering object over an image in order to find the
    image's smooth continuous background.  For the purpose of explaining this
    algorithm, imagine that the 2D grayscale image has a third (height)
    dimension defined by the intensity value at every point in the image.  The
    center of the filtering object, a patch from the top of a sphere having
    radius 'radius', is moved along each scan line of the image so that the
    patch is tangent to the image at one or more points with every other point
    on the patch below the corresponding (x,y) point of the image.  Any point
    either on or below the patch during this process is considered part of the
    background.
    """
    height, width = array.shape
    pixels = numpy.float32(array.flatten())
    z_ball = ball.data
    ball_width = ball.width
    radius = ball_width / 2
    cache = numpy.zeros(width * ball_width,).astype('float32')
    if jit.lower() in 'parakeet':
        pixels = run_ballp(z_ball, ball_width, array, pixels, cache)
        return numpy.reshape(pixels, array.shape)
    elif jit.lower() in 'numba':
        pixels = run_balln(z_ball, ball_width, array, pixels, cache)
        return numpy.reshape(pixels, array.shape)
    else:
        print("No JIT")
    for y in range(-radius, height + radius):
        next_line_to_write_in_cache = (y + radius) % ball_width
        next_line_to_read = y + radius
        if next_line_to_read < height:
            src = next_line_to_read * width
            dest = next_line_to_write_in_cache * width
            cache[dest:dest + width] = pixels[src:src + width]
            p = next_line_to_read * width
            for x in range(width):
                pixels[p] = -float('inf')
                p += 1
        y_0 = y - radius
        if y_0 < 0:
            y_0 = 0
        y_ball_0 = y_0 - y + radius
        y_end = y + radius
        if y_end >= height:
            y_end = height - 1
        for x in range(-radius, width + radius):
            z = float('inf')
            x_0 = x - radius
            if x_0 < 0:
                x_0 = 0
            x_ball_0 = x_0 - x + radius
            x_end = x + radius
            if x_end >= width:
                x_end = width - 1
            y_ball = y_ball_0
            for yp in range(y_0, y_end + 1):
                cache_pointer = (yp % ball_width) * width + x_0
                bp = x_ball_0 + y_ball * ball_width
                for xp in range(x_0, x_end + 1):
                    z_reduced = cache[cache_pointer] - z_ball[bp]
                    if z > z_reduced:
                        z = z_reduced
                    cache_pointer += 1
                    bp += 1
                y_ball += 1

            y_ball = y_ball_0
            for yp in range(y_0, y_end + 1):
                p = x_0 + yp * width
                bp = x_ball_0 + y_ball * ball_width
                for xp in range(x_0, x_end + 1):
                    z_min = z + z_ball[bp]
                    if pixels[p] < z_min:
                        pixels[p] = z_min
                    p += 1
                    bp += 1
                y_ball += 1

    return numpy.reshape(pixels, array.shape)


class RollingBall(object):
    """
    A rolling ball (or actually a square part thereof).
    """
    def __init__(self, radius):
        if radius <= 10:
            self.shrink_factor = 1
            arc_trim_per = 24
        elif radius <= 30:
            self.shrink_factor = 2
            arc_trim_per = 24
        elif radius <= 100:
            self.shrink_factor = 4
            arc_trim_per = 32
        else:
            self.shrink_factor = 8
            arc_trim_per = 40
        self.build(radius, arc_trim_per)

    def build(self, ball_radius, arc_trim_per):
        small_ball_radius = ball_radius / self.shrink_factor
        if small_ball_radius < 1:
            small_ball_radius = 1
        rsquare = small_ball_radius * small_ball_radius
        xtrim = int(arc_trim_per * small_ball_radius) / 100
        half_width = int(round(small_ball_radius - xtrim))
        self.width = (2 * half_width) + 1
        self.data = [0.0] * (self.width * self.width)

        p = 0
        for y in range(self.width):
            for x in range(self.width):
                xval = x - half_width
                yval = y - half_width
                temp = rsquare - (xval * xval) - (yval * yval)

                if temp > 0:
                    self.data[p] = float(math.sqrt(temp))
                p += 1
        self.data = numpy.array(self.data, dtype='float32')
