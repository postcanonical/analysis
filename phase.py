import math
import numpy as np

def get_amp_phase(data, position, width, is_calibrated=False, calibration=None, lookup_size=100):
    """
    Calculate the amplitude and phase of the Lissajous curve.

    :param data: List of tuples [(x1, y1), (x2, y2), ...] representing the data points.
    :param position: Central position in the data to analyze.
    :param width: Window width around the position.
    :param is_calibrated: Whether to apply calibration to amplitude.
    :param calibration: Dictionary with calibration factors {'voltage': float, 'amplitude': float}.
    :param lookup_size: Size of the lookup range for filtering distant points.
    :return: Tuple (amplitude, phase) where amplitude is the distance and phase is in degrees.
    """
    n_points = len(data)
    p1 = max(0, position - width // 2)
    p2 = min(n_points - 1, position + width // 2)

    # Initialize indices and min/max values
    A = B = C = D = p1
    min_x = max_x = data[p1][0]
    min_y = max_y = data[p1][1]

    # Find extreme points based on X values
    for i in range(p1 + 1, p2 + 1):
        if data[i][0] < min_x:
            min_x = data[i][0]
            A = i
        if data[i][0] > max_x:
            max_x = data[i][0]
            B = i

    # Find extreme points based on Y values
    for i in range(p1 + 1, p2 + 1):
        if data[i][1] < min_y:
            min_y = data[i][1]
            C = i
        if data[i][1] > max_y:
            max_y = data[i][1]
            D = i

    # Adjust ranges for lookup
    if C < A:
        A, C = C, A
    if D < B:
        B, D = D, B

    if (C - A) > lookup_size:
        tA = (A + C - lookup_size) // 2
        tC = (A + C + lookup_size) // 2
        A, C = tA, tC
    if (D - B) > lookup_size:
        tB = (B + D - lookup_size) // 2
        tD = (B + D + lookup_size) // 2
        B, D = tB, tD

    # Find the most distant points
    max_dist = 0
    q1 = q2 = None
    for s1 in range(A, C + 1):
        for s2 in range(B, D + 1):
            dx = data[s1][0] - data[s2][0]
            dy = data[s1][1] - data[s2][1]
            dist = dx**2 + dy**2
            if dist > max_dist:
                max_dist = dist
                q1, q2 = s1, s2

    # Ensure indices are valid
    q1 = min(q1, n_points - 1)
    q2 = min(q2, n_points - 1)

    # Swap if necessary to maintain order
    if q1 < q2:
        q1, q2 = q2, q1

    # Calculate amplitude and phase
    x1, y1 = data[q1]
    x2, y2 = data[q2]
    amplitude = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if is_calibrated and calibration:
        amplitude *= calibration['voltage'] / calibration['amplitude']

    dx = x1 - x2
    dy = y2 - y1
    phase = math.degrees(math.atan2(dy, dx))
    phase = phase % 360  # Normalize phase to [0, 360)

    return amplitude, phase

