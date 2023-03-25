from xml.dom import minidom

import matplotlib.pyplot as plt
import numpy as np
from svg.path import parse_path, Move, Line
import pickle

with open("cat.svg") as f:
    svg_string = f.read()

xml_object = minidom.parseString(svg_string)

points = []
lines = []

width = 500
height = 420
scale_x = 120
scale_y = 150
offset_x = -30


for element in xml_object.getElementsByTagName("path"):
    for path in parse_path(element.getAttribute("d")):
        if isinstance(path, Move):
            # points.append([path.start.real, path.start.imag])
            points.append([])
            pass
        elif isinstance(path, Line):
            points[-1].append([path.start.real, path.start.imag])
            points[-1].append([path.end.real, path.end.imag])
            lines.append(
                [[path.start.real, path.start.imag], [path.end.real, path.end.imag]]
            )

curves = []
for curve in points:
    matrix = np.array(curve)
    matrix[:, 0] = scale_x * matrix[:, 0] / width
    matrix[:, 1] = scale_y * ((height - matrix[:, 1]) / height)

    curves.append(matrix)

# import ipdb; ipdb.set_trace()
xml_object.unlink()

lines = np.array(lines)
# import ipdb; ipdb.set_trace()
lines[:, :, 0] = offset_x + scale_x * lines[:, :, 0] / width
lines[:, :, 1] = scale_y * ((height - lines[:, :, 1]) / height)


with open("cat_lines.pkl", "wb") as f:
    pickle.dump(lines, f)


lines = lines.reshape(-1, 2)

plt.figure()
# plt.scatter(points[:, 0], 500 - points[:, 1])
# plt.scatter(points[:, 0], -points[:, 1])
# for curve in curves:
#     plt.plot(curve[:, 0], curve[:, 1])

plt.plot(lines[:, 0], lines[:, 1])

plt.show()
