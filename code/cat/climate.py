import numpy as np


def cat_weights(source, target, clip=None):
    ppt = np.abs(source["PPT"] - np.mean(target["PPT"])) / np.std(target["PPT"])
    gdd = np.abs(source["GDD"] - np.mean(target["GDD"])) / np.std(target["GDD"])
    if clip is not None:
        ppt, gdd = np.clip(ppt, None, clip), np.clip(gdd, None, clip)
    w = 1 / np.sqrt(np.exp(ppt) ** 2 + np.exp(gdd) ** 2)
    return np.asarray(w / np.nanmax(w))
