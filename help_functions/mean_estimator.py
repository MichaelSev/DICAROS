# Step 1
from help_functions.align_functions import *
import pandas as pd
import numpy as np

# Function 
def make_species_mean_common(landmarks,species_all,idxs,d,mean_shape_save=None,output_dir=None,filename_mean=None):

    # Select all points for one species (e.g., first species in the dataset)
    reference_list = []
    species_list = []
    count_list = []

    for species in species_all.unique():
        # select all rows where the species is the current species
        species_idx = np.where(species_all == species)

        # Take all idx from landmarks
        landmarks_subset = landmarks.iloc[species_idx]

        shapes,_ = align_species_landmarks(landmarks_subset,d)

        # make shapes into a matrix, where each row is a shape, and it has d*n_landmarks columns
        n_landmarks = shapes.shape[1]
        shapes = shapes.reshape(-1,d*n_landmarks)

        reference = np.mean(shapes, axis=0)

        count = len(species_idx)

        species_list.append(species)
        reference_list.append(reference)
        count_list.append(count)


    # Realign all species to the mean shape
    if idxs is not None:
        print("Three")
        reference_list_output,_ = align_species_landmarks_with_idx(reference_list,idxs,d)
    else: 
        print("Four")
        reference_list_output,_ = align_species_landmarks(reference_list,d)



    # Flatten to dd array 
    reference_list_output = reference_list_output.reshape(-1,d*n_landmarks)



    # Combine all the reference shapes into one matrix, with count and species name as the first two columns
    reference_matrix = np.column_stack((species_list,count_list, reference_list_output))

    # make pandas dataframe
    reference_df = pd.DataFrame(reference_matrix)

    # save file
    if mean_shape_save is not None: 
        reference_df.to_csv(f'{output_dir}/{filename_mean}', index=False)



    return reference_df
