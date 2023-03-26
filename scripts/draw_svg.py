import pickle
import sys
from pathlib import Path
from xml.dom import minidom

import matplotlib.pyplot as plt
import numpy as np
from svg.path import Line, parse_path

filename = sys.argv[1]

with open(filename) as f:
    svg_string = f.read()

xml_object = minidom.parseString(svg_string)

lines = []

# cat.svg
# width = 500
# height = 420
# scale_x = 120
# scale_y = 150
# offset_x = -30
# offset_y = 0

# cat2.svg
# width = 25.2
# height = 28.8
# scale_x = 55
# scale_y = 85
# offset_x = -130
# offset_y = 250

# dog.svg
width = 25.2
height = 28.8
scale_x = 30
scale_y = 40
offset_x = 20
offset_y = 60


for element in xml_object.getElementsByTagName("path"):
    for path in parse_path(element.getAttribute("d")):
        if isinstance(path, Line):
            lines.append(
                [[path.start.real, path.start.imag], [path.end.real, path.end.imag]]
            )

xml_object.unlink()

lines = np.array(lines)
lines[:, :, 0] = offset_x + scale_x * lines[:, :, 0] / width
lines[:, :, 1] = offset_y + scale_y * ((height - lines[:, :, 1]) / height)


name = Path(filename).stem

with open(f"data/{name}_lines.pkl", "wb") as f:
    pickle.dump(lines, f)


# Format for rust
def format_point(point) -> str:
    return f"({point[0]:.2f}, {point[1]:.2f})"


for line in lines:
    print(f"({format_point(line[0])}, {format_point(line[1])}),")


plt.figure()
# Line is of shape (n, 2, 2)
# lines = lines.reshape(-1, 2)
# plt.plot(lines[:, 0], lines[:, 1])
for line in lines:
    plt.plot(line[:, 0], line[:, 1])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.tight_layout()

plt.show()
