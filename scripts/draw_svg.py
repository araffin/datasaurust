"""
Utility script to retrieves lines from an svg.
"""
import pickle
import sys
from pathlib import Path
from xml.dom import minidom
import yaml

import matplotlib.pyplot as plt
import numpy as np
from svg.path import Line, parse_path

filename = Path(sys.argv[1])

with open(filename) as f:
    svg_string = f.read()

xml_object = minidom.parseString(svg_string)

lines = []

name = filename.stem

# with open(f"data/{name}_lines.pkl", "wb") as f:
#     pickle.dump(lines, f)


# Read scaling config from yaml file
with open(Path(__file__).parent / "config.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)[name]
    # Convert each key to a float
    config = {key: float(value) for key, value in config.items()}
    width = config["width"]
    height = config["height"]
    scale_x = config["scale_x"]
    scale_y = config["scale_y"]
    offset_x = config["offset_x"]
    offset_y = config["offset_y"]

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
