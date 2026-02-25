'''
Segmentation related functions

Last modified: Jan 2026
'''

import numpy as np
from shapely import polygons
from skimage.measure import find_contours, label
from skimage.segmentation import find_boundaries
from magic_py.magic_rag import magic_rag

def get_segment_contours(pred, x, y):
    target_rgb = pred[x, y]
    mask = np.all(pred == target_rgb, axis=-1)
    labeled = label(mask, connectivity=2)
    region_label = labeled[x, y]

    if region_label == 0:
        return []

    segment_mask = labeled == region_label

    # Get list of contours — each a Nx2 array of [row, col]
    contours = find_contours(segment_mask.astype(np.uint8), level=0.5)

    return contours, segment_mask

# Segmentation on a local zoomed area, applied to cropped image only
def Map_labels(labels):
    # map labels to [0, 1, 2,..., n_labels-1]
    id = np.unique(labels)
    aux_id = -1 * np.ones_like(labels)     
    c = 0
    for i in id:                                
        if i != -1:
            aux_id[labels == i] = c
            c += 1
    return aux_id

def IRGS(img, n_classes, n_iter, mask=None):
    # --- RUN IRGS --- #
    rag = None
    if mask is None:
        rag = magic_rag(img, msk=None, N_class=n_classes, verbose=True)
    else:
        rag = magic_rag(img, msk=mask, N_class=n_classes, verbose=True)
    print("Initializing k-means with", n_classes, "classes")
    rag.initialize_kmeans()
    print("Performing", str(n_iter), "IRGS iterations...")
    # for j in tqdm(range(n_iter), ncols=50):
    for j in range(n_iter):
        # rag.irgs_step(beta1=beta1, current_iter=i+1)
        rag.irgs_step(current_iter=j+1)
    irgs_output = rag.result_image
    # irgs_output = rag.result_image_with_boundaries
    # boundaries = np.int16(rag.bmp == -2) # Not consistent with rag.result_image_with_boundaries
    boundaries = np.int16(rag.result_image_with_boundaries != -2)
    boundaries[boundaries == 0] = -1
    boundaries[irgs_output < 0] = -1
    irgs_output[irgs_output < 0] = -1           # background and boundaries.
                                                # IRGS returns an aditional class with 
                                                # label -2 for landmask and boundaries\
    irgs_output = Map_labels(irgs_output)
    return irgs_output, boundaries

def remove_edge_touching_polygons(irgs_output):
    # Get only the enclosed polygons that don't touch the edges of the selected area
    rows, cols = len(irgs_output), len(irgs_output[0])
    polygons = irgs_output.copy()
    
    if rows == 0 or cols == 0:
        return irgs_output, np.ones_like(irgs_output, dtype=np.int8)  # No valid area, return empty boundaries

    visited = set()
    stack = []

    def push_if_border(r, c):
        stack.append((r, c))

    # add all border cells
    for c in range(cols):
        push_if_border(0, c)
        push_if_border(rows-1, c)
    for r in range(rows):
        push_if_border(r, 0)
        push_if_border(r, cols-1)

    # flood-fill from each border cell
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue

        val = polygons[r, c]
        # If it's already -1 from a previous fill, treat it as "already erased"
        if val == -1:
            continue

        # DFS this component of same 'val'
        comp_stack = [(r, c)]
        component_cells = []

        while comp_stack:
            x, y = comp_stack.pop()
            if (x, y) in visited:
                continue
            if polygons[x, y] != val:
                continue

            visited.add((x, y))
            component_cells.append((x, y))

            if x > 0:     comp_stack.append((x-1, y))
            if x < rows-1:   comp_stack.append((x+1, y))
            if y > 0:     comp_stack.append((x, y-1))
            if y < cols-1:   comp_stack.append((x, y+1))

        # This entire component is reachable from the border so set to -1
        for x, y in component_cells:
            polygons[x, y] = -1

    boundaries = find_boundaries(polygons, mode='outer', connectivity=1, background=-1).astype(np.int8)
    boundaries[boundaries == 1] = -1
    boundaries[boundaries == 0] = 1

    return polygons, boundaries
