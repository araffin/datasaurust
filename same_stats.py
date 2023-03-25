"""
Usage:
    same_stats.py run <shape_start> <shape_end> [<iters>][<decimals>][<frames>]

    This is code created for the paper:
    Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics through Simulated Annealing
    Justin Matejka and George Fitzmaurice
    ACM CHI 2017

    The paper, video, and associated code and datasets can be found on the Autodesk Research website: http://www.autodeskresearch.com/papers/samestats
    For any questions, please contact Justin Matejka (Justin.Matejka@Autodesk.com)

    The most basic way to try this out is to run a command like this from the command line:
    python same_stats.py dino circle

    That will start with the Dinosaurus dataset, and morph it into a circle.


    I have stripped out some of the functionality for some examples in the paper, for the time being, to make
    the code easier to follow. If you would like the dirty, perhaps unrunnable for you, code, contact me and
    I can get it to you. I will be adding all that functionality back in, in a more reasonable way shortly, and
    will have the project hosted on GitHub so it is easier to share.
"""


# warnings.simplefilter(action="ignore", category=FutureWarning)
# warnings.simplefilter(action="ignore", category=UserWarning)

import itertools
import math
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytweening
import seaborn as sns
from docopt import docopt

# from tqdm import tnrange, trange
from tqdm.rich import tqdm
import pickle

# setting up the style for the charts
sns.set_style("darkgrid")
mpl.rcParams["font.size"] = 12.0
mpl.rcParams["font.family"] = "monospace"
mpl.rcParams["font.weight"] = "normal"
mpl.rcParams["font.sans-serif"] = (
    "Helveitca",
    "Bitstream Vera Sans",
    "Lucida Grande",
    "Verdana",
    "Geneva",
    "Lucid",
    "Arial",
    "Avant Garde",
    "sans-serif",
)
# mpl.rcParams["font.monospace"] = (
#     "Decima Mono",
#     "Bitstream Vera Sans Mono",
#     "Andale Mono",
#     "Nimbus Mono L",
#     "Courier New",
#     "Courier",
#     "Fixed",
#     "Terminal",
#     "monospace",
# )
mpl.rcParams["text.color"] = "#222222"
mpl.rcParams["pdf.fonttype"] = 42

line_shapes = [
    "x",
    "h_lines",
    "v_lines",
    "wide_lines",
    "high_lines",
    "slant_up",
    "slant_down",
    "center",
    "star",
    "down_parab",
    "cat_lines",
]
all_targets = list(line_shapes)
all_targets.extend(["circle", "bullseye", "dots", "cat"])
initial_datasets = ["dino", "rando", "slant", "big_slant"]

cat_data = np.load("cat.npz")
cat_x = cat_data["x"][::4]
cat_y = cat_data["y"][::4]

with open("cat_lines.pkl", "rb") as f:
    cat_lines = pickle.load(f)


#
# these are the initial datasets which are used in the paper
#
def load_dataset(name="dino"):
    if name == "dino":
        df = pd.read_csv(
            "seed_datasets/Datasaurus_data.csv", header=None, names=["x", "y"]
        )
    elif name == "rando":
        df = pd.read_csv("seed_datasets/random_cloud.csv")
        df = df[["x", "y"]]
    elif name == "slant":
        df = pd.read_csv("seed_datasets/slanted_less.csv")
        df = df[["x", "y"]]
    elif name == "big_slant":
        df = pd.read_csv("seed_datasets/less_angled_blob.csv")
        df = df[["x", "y"]]
        df = df.clip(1, 99)

    return df.copy()


#
# This function calculates the summary statistics for the given set of points
#
def get_values(df):
    xm = df.x.mean()
    ym = df.y.mean()
    xsd = df.x.std()
    ysd = df.y.std()
    pc = df.corr().x.y

    # return [xm, ym, xsd, ysd, pc]
    return [xm, ym, xsd, ysd]


#
# checks to see if the statistics are still within the acceptable bounds
# with df1 as the original dataset, and df2 as the one we are testing
#
def is_error_still_ok(df1, df2, decimals=2):
    r1 = get_values(df1)
    r2 = get_values(df2)

    # check each of the error values to check if they are the same to the correct number of decimals
    r1 = [math.floor(r * 10**decimals) for r in r1]
    r2 = [math.floor(r * 10**decimals) for r in r2]

    # we are good if r1 and r2 have the same numbers
    er = np.subtract(r1, r2)
    er = [abs(n) for n in er]

    return np.max(er) == 0


def lineMagnitude(x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude


#
# This function calcualtes the minimum distance between a point and a line, used
# to determine if the points are getting closer to the target
#
def DistancePointLine(px, py, x1, y1, x2, y2):
    # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = ((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        # // closest point does not fall within the line segment, take the shorter distance
        # // to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine


# save the plot to an image file
def save_scatter(df, iter_num, dp=72):
    show_scatter(df)
    plt.savefig("logs/" + str(iter_num) + ".png", dpi=dp)
    plt.clf()
    plt.cla()
    plt.close()


def save_scatter_and_results(
    df, iter_num, dp=72, labels=["X Mean", "Y Mean", "X SD", "Y SD", "Corr."]
):
    show_scatter_and_results(df, labels=labels)
    plt.savefig("logs/" + str(iter_num) + ".png", dpi=dp)
    plt.clf()
    plt.cla()
    plt.close()


# create a plot of the data
def show_scatter(
    df, xlim=(-5, 105), ylim=(-5, 105), color="black", marker="o", reg_fit=False
):
    sns.regplot(
        x="x",
        y="y",
        data=df,
        ci=None,
        fit_reg=reg_fit,
        marker=marker,
        scatter_kws={"s": 50, "alpha": 0.7, "color": color},
        line_kws={"linewidth": 4, "color": "red"},
    )
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()


# create a plot which shows both the plot, and the text of the summary statistics
def show_scatter_and_results(df, labels=["X Mean", "Y Mean", "X SD", "Y SD", "Corr."]):
    # plt.figure(figsize=(12, 5))
    plt.figure(figsize=(7, 5))
    sns.regplot(
        x="x",
        y="y",
        data=df,
        ci=None,
        fit_reg=False,
        scatter_kws={"s": 50, "alpha": 0.7, "color": "black"},
    )
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.tight_layout()

    # res = get_values(df)
    # fs = 30
    # y_off = -5
    # max_label_length = max([len(l) for l in labels])

    # plt.text(
    #     110,
    #     y_off + 80,
    #     labels[0].ljust(max_label_length) + ": " + format(res[0], "0.9f")[:-2],
    #     fontsize=fs,
    #     alpha=0.3,
    # )
    # plt.text(
    #     110,
    #     y_off + 65,
    #     labels[1].ljust(max_label_length) + ": " + format(res[1], "0.9f")[:-2],
    #     fontsize=fs,
    #     alpha=0.3,
    # )
    # plt.text(
    #     110,
    #     y_off + 50,
    #     labels[2].ljust(max_label_length) + ": " + format(res[2], "0.9f")[:-2],
    #     fontsize=fs,
    #     alpha=0.3,
    # )
    # plt.text(
    #     110,
    #     y_off + 35,
    #     labels[3].ljust(max_label_length) + ": " + format(res[3], "0.9f")[:-2],
    #     fontsize=fs,
    #     alpha=0.3,
    # )
    # plt.text(
    #     110,
    #     y_off + 20,
    #     labels[4].ljust(max_label_length) + ": " + format(res[4], "+.9f")[:-2],
    #     fontsize=fs,
    #     alpha=0.3,
    # )

    # plt.text(
    #     110,
    #     y_off + 80,
    #     labels[0].ljust(max_label_length) + ": " + format(res[0], ".9f")[:-7],
    #     fontsize=fs,
    #     alpha=1,
    # )
    # plt.text(
    #     110,
    #     y_off + 65,
    #     labels[1].ljust(max_label_length) + ": " + format(res[1], "0.9f")[:-7],
    #     fontsize=fs,
    #     alpha=1,
    # )
    # plt.text(
    #     110,
    #     y_off + 50,
    #     labels[2].ljust(max_label_length) + ": " + format(res[2], "0.9f")[:-7],
    #     fontsize=fs,
    #     alpha=1,
    # )
    # plt.text(
    #     110,
    #     y_off + 35,
    #     labels[3].ljust(max_label_length) + ": " + format(res[3], "0.9f")[:-7],
    #     fontsize=fs,
    #     alpha=1,
    # )
    # plt.text(
    #     110,
    #     y_off + 20,
    #     labels[4].ljust(max_label_length) + ": " + format(res[4], "+.9f")[:-7],
    #     fontsize=fs,
    #     alpha=1,
    # )
    # plt.tight_layout(rect=[0, 0, 0.57, 1])


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def average_location(pairs):
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return [np.mean(xs), np.mean(ys)]


#
# These are the hardcoded shapes which we perturb towards. It would useful to have
# a tool for drawing these shapes instead
#
def get_points_for_shape(line_shape):
    lines = []
    if line_shape == "x":
        l1 = [[20, 0], [100, 100]]
        l2 = [[20, 100], [100, 0]]
        lines = [l1, l2]
    elif line_shape == "h_lines":
        lines = [[[0, y], [100, y]] for y in [10, 30, 50, 70, 90]]
    elif line_shape == "v_lines":
        lines = [[[x, 0], [x, 100]] for x in [10, 30, 50, 70, 90]]
    elif line_shape == "wide_lines":
        l1 = [[10, 0], [10, 100]]
        l2 = [[90, 0], [90, 100]]
        lines = [l1, l2]
    elif line_shape == "high_lines":
        l1 = [[0, 10], [100, 10]]
        l2 = [[0, 90], [100, 90]]
        lines = [l1, l2]
    elif line_shape == "slant_up":
        l1 = [[0, 0], [100, 100]]
        l2 = [[0, 30], [70, 100]]
        l3 = [[30, 0], [100, 70]]
        l4 = [[50, 0], [100, 50]]
        l5 = [[0, 50], [50, 100]]
        lines = [l1, l2, l3, l4, l5]
    elif line_shape == "slant_down":
        l1 = [[0, 100], [100, 0]]
        l2 = [[0, 70], [70, 0]]
        l3 = [[30, 100], [100, 30]]
        l4 = [[0, 50], [50, 0]]
        l5 = [[50, 100], [100, 50]]
        lines = [l1, l2, l3, l4, l5]
    elif line_shape == "center":
        cx = 54.26
        cy = 47.83
        l1 = [[cx, cy], [cx, cy]]
        lines = [l1]
    elif line_shape == "star":
        star_pts = [
            10,
            40,
            40,
            40,
            50,
            10,
            60,
            40,
            90,
            40,
            65,
            60,
            75,
            90,
            50,
            70,
            25,
            90,
            35,
            60,
        ]
        pts = [star_pts[i : i + 2] for i in range(0, len(star_pts), 2)]
        pts = [[p[0] * 0.8 + 20, 100 - p[1]] for p in pts]
        pts.append(pts[0])
        lines = [pts[i : i + 2] for i in range(0, len(pts) - 1, 1)]
    elif line_shape == "down_parab":
        curve = [[x, -(((x - 50) / 4) ** 2) + 90] for x in np.arange(0, 100, 3)]
        lines = [curve[i : i + 2] for i in range(0, len(curve) - 1, 1)]
    elif line_shape == "cat_lines":
        lines = cat_lines

    return lines


#
# This is the function which does one round of perturbation
# df: is the current dataset
# initial: is the original dataset
# target: is the name of the target shape
# shake: the maximum amount of movement in each iteration
#
def perturb(
    df,
    initial,
    target="circle",
    line_error=1.5,
    shake=0.1,
    allowed_dist=2,
    temp=0,
    x_bounds=[0, 100],
    y_bounds=[0, 100],
    custom_points=None,
):
    # take one row at random, and move one of the points a bit
    row = np.random.randint(0, len(df))
    i_xm = df["x"][row]
    i_ym = df["y"][row]

    # this is the simulated annealing step, if "do_bad", then we are willing to
    # accept a new state which is worse than the current one
    do_bad = np.random.random_sample() < temp

    while True:
        xm = i_xm + np.random.randn() * shake
        ym = i_ym + np.random.randn() * shake

        if target == "circle":
            # info for the circle
            cx = 54.26
            cy = 47.83
            r = 30

            dc1 = dist([df["x"][row], df["y"][row]], [cx, cy])
            dc2 = dist([xm, ym], [cx, cy])

            old_dist = abs(dc1 - r)
            new_dist = abs(dc2 - r)

        elif target == "bullseye":
            # info for the bullseye
            cx = 54.26
            cy = 47.83
            rs = [18, 37]

            dc1 = dist([df["x"][row], df["y"][row]], [cx, cy])
            dc2 = dist([xm, ym], [cx, cy])

            old_dist = np.min([abs(dc1 - r) for r in rs])
            new_dist = np.min([abs(dc2 - r) for r in rs])

        elif target == "dots":
            # create a grid of "cluster points" and move if you are getting closer
            # (or are already close enough)
            xs = [25, 50, 75]
            ys = [20, 50, 80]

            old_dist = np.min(
                [
                    dist([x, y], [df["x"][row], df["y"][row]])
                    for x, y in itertools.product(xs, ys)
                ]
            )
            new_dist = np.min(
                [dist([x, y], [xm, ym]) for x, y in itertools.product(xs, ys)]
            )
        elif target == "cat":
            # create a grid of "cluster points" and move if you are getting closer
            # (or are already close enough)
            xs = cat_x
            ys = cat_y

            old_dist = np.min(
                [dist([x, y], [df["x"][row], df["y"][row]]) for x, y in zip(xs, ys)]
            )
            new_dist = np.min([dist([x, y], [xm, ym]) for x, y in zip(xs, ys)])

        elif target in line_shapes:
            lines = get_points_for_shape(target)

            # calculate how far the point is from the closest one of these
            old_dist = np.min(
                [
                    DistancePointLine(i_xm, i_ym, l[0][0], l[0][1], l[1][0], l[1][1])
                    for l in lines
                ]
            )
            new_dist = np.min(
                [
                    DistancePointLine(xm, ym, l[0][0], l[0][1], l[1][0], l[1][1])
                    for l in lines
                ]
            )

        # check if the new distance is closer than the old distance
        # or, if it is less than our allowed distance
        # or, if we are do_bad, that means we are accpeting it no matter what
        # if one of these conditions are met, jump out of the loop
        if (
            (new_dist < old_dist or new_dist < allowed_dist or do_bad)
            and ym > y_bounds[0]
            and ym < y_bounds[1]
            and xm > x_bounds[0]
            and xm < x_bounds[1]
        ):
            break

    # set the new data point, and return the set
    df["x"][row] = xm
    df["y"][row] = ym
    return df


def s_curve(v):
    return pytweening.easeInOutQuad(v)


import sys


def is_kernel():
    if "IPython" not in sys.modules:
        # IPython hasn't been imported, definitely not
        return False
    from IPython import get_ipython

    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), "kernel", None) is not None


#
# this is the main fucntion, for taking one datset and perturbing it into a target shape
# df: the initial dataset
# target: the shape we are aiming for
# iters: how many iterations to run the algorithm for
# num_frames: how many frames to save to disk (for animations)
# decimals: how many decimal points to keep fixed
# shake: the maximum movement for a single interation
#
def run_pattern(
    df,
    target,
    iters=100000,
    num_frames=100,
    decimals=2,
    shake=0.2,
    max_temp=0.4,
    min_temp=0,
    ramp_in=False,
    ramp_out=False,
    freeze_for=0,
    labels=["X Mean", "Y Mean", "X SD", "Y SD", "Corr."],
    reset_counts=False,
    custom_points=False,
):
    global frame_count
    global it_count

    if reset_counts:
        it_count = 0
        frame_count = 0

    r_good = df.copy()

    # this is a list of frames that we will end up writing to file
    write_frames = [
        int(round(pytweening.linear(x) * iters))
        for x in np.arange(0, 1, 1 / (num_frames - freeze_for))
    ]

    if ramp_in and not ramp_out:
        write_frames = [
            int(round(pytweening.easeInSine(x) * iters))
            for x in np.arange(0, 1, 1 / (num_frames - freeze_for))
        ]
    elif ramp_out and not ramp_in:
        write_frames = [
            int(round(pytweening.easeOutSine(x) * iters))
            for x in np.arange(0, 1, 1 / (num_frames - freeze_for))
        ]
    elif ramp_out and ramp_in:
        write_frames = [
            int(round(pytweening.easeInOutSine(x) * iters))
            for x in np.arange(0, 1, 1 / (num_frames - freeze_for))
        ]

    extras = [iters] * freeze_for
    write_frames.extend(extras)

    # this gets us the nice progress bars in the notbook, but keeps it from crashing
    # looper = trange

    # this is the main loop, were we run for many iterations to come up with the pattern
    # for i in looper(iters + 1, leave=True, ascii=True, desc=target + " pattern"):
    for i in tqdm(range(iters + 1)):
        t = (max_temp - min_temp) * s_curve((iters - i) / iters) + min_temp

        if target in all_targets:
            test_good = perturb(r_good.copy(), initial=df, target=target, temp=t)
        else:
            raise Exception("bah, that's not a proper type of pattern")

        # here we are checking that after the purturbation, that the statistics are still within the allowable bounds
        if is_error_still_ok(df, test_good, decimals):
            r_good = test_good

        # save this chart to the file
        for x in range(write_frames.count(i)):
            save_scatter_and_results(
                r_good,
                target + "-image-" + format(int(frame_count), "05"),
                150,
                labels=labels,
            )
            # save_scatter(r_good, target + "-image-"+format(int(frame_count), '05'), 150)
            r_good.to_csv(
                "logs/" + target + "-data-" + format(int(frame_count), "05") + ".csv"
            )

            frame_count = frame_count + 1
    return r_good


#
# function to load a dataset, and then perturb it
# start_dataset is a string, and one of ['dino', 'rando', 'slant', 'big_slant']
#


def do_single_run(start_dataset, target, iterations=100000, decimals=2, num_frames=100):
    global it_count
    global frame_count
    it_count = 0
    frame_count = 0

    df = load_dataset(start_dataset)
    temp = run_pattern(df, target, iters=iterations, num_frames=num_frames)
    return temp


def print_stats(df):
    print("N: ", len(df))
    print("X mean: ", df.x.mean())
    print("X SD: ", df.x.std())
    print("Y mean: ", df.y.mean())
    print("Y SD: ", df.y.std())
    print("Pearson correlation: ", df.corr().x.y)


# run <shape_start> <shape_end> [<iters>][<decimals>]
if __name__ == "__main__":
    arguments = docopt(__doc__, version="Same Stats 1.0")
    if arguments["run"]:
        it = 100000
        de = 2
        frames = 100
        if arguments["<iters>"]:
            it = int(arguments["<iters>"])
        if arguments["<decimals>"]:
            de = int(arguments["<decimals>"])
        if arguments["<decimals>"]:
            frames = int(arguments["<frames>"])

        shape_start = arguments["<shape_start>"]
        shape_end = arguments["<shape_end>"]

        if shape_start in initial_datasets and shape_end in all_targets:
            do_single_run(
                shape_start, shape_end, iterations=it, decimals=de, num_frames=frames
            )
        else:
            print("************* One of those shapes isn't correct:")
            print("shape_start must be one of ", initial_datasets)
            print("shape_end must be one of ", all_targets)
