from numba import njit, prange, cuda
import numpy as np
from numba.typed import List
from skimage.measure import find_contours


@cuda.jit
def blend_overlay_cuda(pred, img, boundmask, landmask, alpha, out):
    y, x = cuda.grid(2)
    h, w = pred.shape[:2]
    beta = 1 - alpha

    if y < h and x < w:
        if landmask[y, x]:
            for c in range(3):
                out[y, x, c] = 255
        elif boundmask[y, x]:
            for c in range(3):
                out[y, x, c] = pred[y, x, c]
        else:
            for c in range(3):
                out[y, x, c] = alpha * pred[y, x, c] + beta * img[y, x, c]

@njit(parallel=True)
def blend_overlay(pred, img, boundmask, landmask, local_boundmask, alpha):
    h, w, c = pred.shape
    out = np.empty((h, w, c), dtype=np.float32)

    beta = 1 - alpha

    for y in prange(h):
        for x in range(w):
            if landmask[y, x]:
                out[y, x, 0] = img[y, x, 0]
                out[y, x, 1] = img[y, x, 1]
                out[y, x, 2] = img[y, x, 2]
            elif boundmask[y, x]:
                out[y, x, :] = pred[y, x, :]
            elif local_boundmask is not None and local_boundmask[y, x]:
                out[y, x, :] = 255
            else:
                for ch in range(c):
                    out[y, x, ch] = alpha * pred[y, x, ch] + beta * img[y, x, ch]

    return out

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Optimize this
def generate_boundaries(lbl):
    boundmask = np.zeros_like(lbl, dtype=bool)   
    for lvl in np.unique(lbl):
        level_ctrs = find_contours(lbl, level=lvl)
        for c in level_ctrs:
            try:
                contours = np.concatenate((contours, c), axis=0)
            except:
                contours = c
    if 'contours' in locals():
        contours = np.uint16(contours)
        boundmask[contours[:,0], contours[:,1]] = True
    return boundmask

@njit(parallel=True)
def apply_brightness(image, nan_mask, brightness=0.0, clip=True):
    # Adjust brightness
    h, w, c = image.shape
    adjusted = np.empty((h, w, c), dtype=np.float32)

    for y in prange(h):
        for x in range(w):
            for ch in range(c):
                if nan_mask[y, x]:
                    adjusted[y, x, ch] = image[y, x, ch]
                else:
                    adjusted[y, x, ch] = image[y, x, ch] + brightness * 128

    if clip:
        for y in prange(h):
            for x in range(w):
                for ch in range(c):
                    if adjusted[y, x, ch] < 0:
                        adjusted[y, x, ch] = 0
                    elif adjusted[y, x, ch] > 255:
                        adjusted[y, x, ch] = 255

    return adjusted.astype(np.uint8)

def ds_to_src_pixel(row_ds, col_ds, src_h, src_w, dst_h, dst_w):
    x_scale = src_w / dst_w
    y_scale = src_h / dst_h
    col_src = (col_ds + 0.5) * x_scale
    row_src = (row_ds + 0.5) * y_scale
    return row_src, col_src

import numpy as np

def build_tiepoint_grid_interpolator(tie_points):
    """
    tie_points: list of dicts with keys row, col, latitude, longitude
    Returns:
      pix2ll(row, col) -> (lat, lon)
      plus the grid arrays (rows, cols, lat_grid, lon_grid) if you want them
    """
    rows = np.array(sorted({tp["row"] for tp in tie_points}), dtype=float)
    cols = np.array(sorted({tp["col"] for tp in tie_points}), dtype=float)

    r_index = {r: i for i, r in enumerate(rows)}
    c_index = {c: j for j, c in enumerate(cols)}

    lat_grid = np.full((len(rows), len(cols)), np.nan, dtype=float)
    lon_grid = np.full((len(rows), len(cols)), np.nan, dtype=float)

    for tp in tie_points:
        i = r_index[tp["row"]]
        j = c_index[tp["col"]]
        lat_grid[i, j] = tp["latitude"]
        lon_grid[i, j] = tp["longitude"]

    # Sanity: ensure full grid
    if np.isnan(lat_grid).any() or np.isnan(lon_grid).any():
        raise ValueError("Tie-point grid is missing some row/col combinations.")

    def bilinear(x, y, xs, ys, v):
        # x=col, y=row
        x = float(np.clip(x, xs[0], xs[-1]))
        y = float(np.clip(y, ys[0], ys[-1]))

        j = np.searchsorted(xs, x) - 1
        i = np.searchsorted(ys, y) - 1
        j = int(np.clip(j, 0, len(xs) - 2))
        i = int(np.clip(i, 0, len(ys) - 2))

        x0, x1 = xs[j], xs[j + 1]
        y0, y1 = ys[i], ys[i + 1]

        q00 = v[i,     j]
        q01 = v[i,     j + 1]
        q10 = v[i + 1, j]
        q11 = v[i + 1, j + 1]

        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

        return (q00 * (1 - tx) * (1 - ty) +
                q01 * (tx)     * (1 - ty) +
                q10 * (1 - tx) * (ty) +
                q11 * (tx)     * (ty))

    def pix2ll(row, col):
        lat = bilinear(col, row, cols, rows, lat_grid)
        lon = bilinear(col, row, cols, rows, lon_grid)
        return float(lat), float(lon)

    return pix2ll, (rows, cols, lat_grid, lon_grid)
