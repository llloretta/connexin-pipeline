import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import regionprops_table
from scipy.spatial import Delaunay
from itertools import combinations



def get_nuclei_centerpoints(nuclei: np.ndarray) -> pd.DataFrame:
    """
    Label a binary nuclei mask and compute centerpoints (centroids)
    of each connected component (nucleus).

    Parameters
    ----------
    nuclei : np.ndarray
        3D binary mask (z, y, x) with values 0/255 (or 0/1).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['label', 'z', 'y', 'x'].
    """
    binary_mask = nuclei > 0
    labeled, num_features = ndimage.label(binary_mask)

    print(f"Found {num_features} connected components")

    props = regionprops_table(
        labeled,
        properties=("label", "centroid")
    )

    df = pd.DataFrame(props)
    df = df.rename(columns={"centroid-0": "z", "centroid-1": "y", "centroid-2": "x"})

    return df[["label", "z", "y", "x"]]



def build_nuclei_edges(nuclei_coords, spacing=(0.766, 0.325, 0.325), max_edge_distance=None):
    """
    Connect neighboring nuclei with a 3D Delaunay triangulation.

    Each edge of the triangulation is treated as a candidate "cell-cell contact"
    between two nuclei. Long edges (nuclei that are actually far apart, but got
    connected because they sit on the outer boundary of the point cloud) are
    removed using a maximum distance cutoff.

    Parameters
    ----------
    nuclei_coords : array, shape (n_nuclei, 3)
        Nuclei centers as [z, y, x], in voxel units.
    spacing : tuple (z, y, x)
        Physical voxel size in micrometers. Needed because triangulating on raw
        voxel coordinates would treat 1 voxel of z as equal to 1 voxel of x/y,
        which is wrong when the voxels are not cubic.
    max_edge_distance : float or None
        Maximum allowed distance (micrometers) between two connected nuclei.
        Edges longer than this are discarded. Use None to keep all edges.

    Returns
    -------
    edges : DataFrame with columns [nucleus_1, nucleus_2, distance_um]
        nucleus_1 / nucleus_2 are row indices into nuclei_coords.
    nuclei_coords_um : array, shape (n_nuclei, 3)
        Same nuclei centers, converted to physical micrometers.
    """
    nuclei_coords = np.asarray(nuclei_coords, dtype=float)
    spacing = np.asarray(spacing, dtype=float)

    # convert voxel indices -> real physical distances (micrometers)
    nuclei_coords_um = nuclei_coords * spacing

    # triangulate in 3D: every nucleus is connected to its natural geometric neighbors
    triangulation = Delaunay(nuclei_coords_um)

    # collect every unique pair of nuclei that share a triangulation cell (simplex)
    unique_pairs = set()
    for simplex in triangulation.simplices:
        for nucleus_a, nucleus_b in combinations(simplex, 2):
            pair = (min(nucleus_a, nucleus_b), max(nucleus_a, nucleus_b))
            unique_pairs.add(pair)

    # compute the real distance for each pair, and drop pairs that are too far apart
    edge_rows = []
    for nucleus_1, nucleus_2 in unique_pairs:
        distance_um = np.linalg.norm(nuclei_coords_um[nucleus_1] - nuclei_coords_um[nucleus_2])
        if max_edge_distance is None or distance_um <= max_edge_distance:
            edge_rows.append({
                'nucleus_1': nucleus_1,
                'nucleus_2': nucleus_2,
                'distance_um': distance_um
            })

    edges = pd.DataFrame(edge_rows)

    print(f"  Nuclei: {len(nuclei_coords)}")
    print(f"  Candidate neighbor pairs found: {len(unique_pairs)}")
    print(f"  Kept after distance filter: {len(edges)}")

    return edges, nuclei_coords_um


def match_connexin_to_nuclei_pairs(locations, edges, nuclei_coords_um,
                                    spacing=(0.766, 0.325, 0.325),
                                    distance_threshold=None):
    """
    Assign each connexin region to the nuclei pair (candidate cell-cell contact)
    it lies closest to.

    For every connexin region, this checks the distance to every nuclei-pair
    midpoint (the ~intercalated-disc position, halfway between the two nuclei)
    and keeps the closest one. If the region is farther than distance_threshold
    from every midpoint, it is left unassigned.

    Parameters
    ----------
    locations : DataFrame
        Connexin region table (e.g. from get_region_properties_3d), must
        contain columns ['z', 'y', 'x'] with region centroids in voxel units.
    edges : DataFrame
        Nuclei-pair table from build_nuclei_edges.
    nuclei_coords_um : array, shape (n_nuclei, 3)
        Nuclei centers in physical micrometers, from build_nuclei_edges.
    spacing : tuple (z, y, x)
        Physical voxel size in micrometers, used to convert connexin region
        centroids (voxel units) into the same micrometer units as the nuclei.
    distance_threshold : float or None
        Maximum distance (micrometers) between a connexin region and its
        nearest nuclei-pair line for the match to be accepted. Regions farther
        than this from every line are left unassigned. Use None to always
        assign to the closest line, regardless of distance.

    Returns
    -------
    matched : DataFrame
        Copy of `locations` with four new columns:
        - nucleus_1, nucleus_2 : the matched nuclei pair (NaN if unassigned)
        - nucleus_distance_um  : distance between that pair of nuclei
        - region_to_line_um    : distance from the region centroid to that
                                 pair's midpoint
    """
    spacing = np.asarray(spacing, dtype=float)

    # convert connexin region centroids from voxel units -> micrometers
    region_coords_um = locations[['z', 'y', 'x']].to_numpy() * spacing

    # for every edge, take its midpoint (the ~intercalated-disc position,
    # halfway between the two nuclei) as the single target point to match against
    nucleus_1_pos = nuclei_coords_um[edges['nucleus_1'].to_numpy()]
    nucleus_2_pos = nuclei_coords_um[edges['nucleus_2'].to_numpy()]
    edge_midpoints = (nucleus_1_pos + nucleus_2_pos) / 2.0

    matched_nucleus_1 = []
    matched_nucleus_2 = []
    matched_nucleus_distance = []
    matched_region_to_line = []

    for region_point in region_coords_um:

        # distance from the region to every edge midpoint
        distance_to_edge = np.linalg.norm(region_point[None, :] - edge_midpoints, axis=1)

        # keep only the single closest edge for this region
        closest_edge_index = np.argmin(distance_to_edge)
        closest_distance = distance_to_edge[closest_edge_index]

        if distance_threshold is not None and closest_distance > distance_threshold:
            # too far from every nuclei pair -> leave unassigned
            matched_nucleus_1.append(np.nan)
            matched_nucleus_2.append(np.nan)
            matched_nucleus_distance.append(np.nan)
        else:
            matched_nucleus_1.append(edges['nucleus_1'].iloc[closest_edge_index])
            matched_nucleus_2.append(edges['nucleus_2'].iloc[closest_edge_index])
            matched_nucleus_distance.append(edges['distance_um'].iloc[closest_edge_index])

        matched_region_to_line.append(closest_distance)

    matched = locations.copy()
    matched['nucleus_1'] = matched_nucleus_1
    matched['nucleus_2'] = matched_nucleus_2
    matched['nucleus_distance_um'] = matched_nucleus_distance
    matched['region_to_line_um'] = matched_region_to_line

    n_assigned = matched['nucleus_1'].notna().sum()
    print(f"  Connexin regions assigned to a nuclei pair: {n_assigned} / {len(matched)}")

    return matched


def align_connexin_z_to_full_stack(locations, z_start_slide):
    """
    Shift connexin region z-coordinates from local crop indices to their
    true position in the full z-stack.

    The connexin regions were detected on a small z-crop of the full stack
    (e.g. only slides 150-160), so their 'z' column currently holds LOCAL
    indices (0, 1, 2, ...) relative to the crop, not the real z-position in
    the full-depth stack that the nuclei were detected in. This adds the
    crop's starting slide number back on, so connexin and nuclei coordinates
    live in the same z reference frame before any distance-based matching.

    Parameters
    ----------
    locations : DataFrame
        Connexin region table (e.g. from get_region_properties_3d), with a
        'z' column in local crop voxel units.
    z_start_slide : int
        Index of the crop's first slide within the full z-stack.
        e.g. if the crop is slides 150-160 of the full stack, z_start_slide = 150.

    Returns
    -------
    locations_aligned : DataFrame
        Copy of `locations` with 'z' replaced by the full-stack z-index.
        All other columns are unchanged.
    """
    locations_aligned = locations.copy()
    locations_aligned['z'] = locations_aligned['z'] + z_start_slide

    print(f"  Shifted connexin z-coordinates by {z_start_slide} slides")
    print(f"  New z range: {locations_aligned['z'].min():.2f} – {locations_aligned['z'].max():.2f}")

    return locations_aligned