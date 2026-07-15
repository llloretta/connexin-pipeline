from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
import numpy as np
import matplotlib.pyplot as plt
from threshold import adapted_sauvola_threshold
from skimage import measure


# tbd: adjust this functions to fit the localization matrix ! OR have both 

# --- functions to use on image data 
def count_connexin_plaques_within_plaque(img_float, binary, min_distance=2):
    
    # count plaques
    # -> this counts 2 peaks when 2 plaques are touching
    coordinates = peak_local_max(
        img_float,
        min_distance=min_distance,
        threshold_abs=0.05,
        labels=binary
    )
    n_plaques = len(coordinates)
    print(f"Plaques with min distance {min_distance} detected: {n_plaques}")

    return coordinates, n_plaques

# def measure_connexin_regions(binary, img_float):
#     # label connected regions for shape/size info
#     labeled = label(binary)
#     props = regionprops(labeled, intensity_image=img_float)

#     return props, labeled

# -- functions to use on binary dataframe 

def count_connexin_plaques(binary):
    # this function counts the number of connected components in a binary mask
    labeled_array = measure.label(binary)  # label the connected components in the binary mask
    num_regions = labeled_array.max()  # or len(np.unique(labeled_array)) - 
    
    print(f"Number of connected components (plaques) detected: {num_regions}")

    return num_regions, labeled_array






