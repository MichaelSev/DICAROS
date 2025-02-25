

import numpy as np
# Code for do GPA

def translate_to_origin(shape):
    centroid = np.mean(shape, axis=0)
    shape_translated = shape - centroid
    return shape_translated, centroid

def rotate_and_scale(A, B):
    # Compute the covariance matrix
    H = B.T @ A
    
    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Compute the optimal rotation matrix
    rotation_matrix = U @ Vt
    
    # Rotate B
    B_rotated = B @ rotation_matrix
    
    # Scale B
    scale_factor = np.sqrt(np.sum(A**2) / np.sum(B_rotated**2))
    B_scaled = B_rotated * scale_factor
    
    return B_scaled, rotation_matrix, scale_factor

def compute_mean_shape(shapes):
    return np.mean(shapes, axis=0)

def align_shapes(shapes, max_iterations=9, tolerance=1e-4):
    num_shapes = len(shapes)
    aligned_shapes = shapes.copy()
    
    for iteration in range(max_iterations):
        # Compute the mean shape of all shapes
        mean_shape = compute_mean_shape(aligned_shapes)
        
        # Translate mean shape to origin
        mean_shape_translated, _ = translate_to_origin(mean_shape)
        
        # Align each shape to the mean shape
        for i in range(num_shapes):
            shape_translated, _ = translate_to_origin(aligned_shapes[i])
            aligned_shape, _, _ = rotate_and_scale(mean_shape_translated, shape_translated)
            aligned_shapes[i] = aligned_shape
        
        # Recompute the mean shape after alignment
        new_mean_shape = compute_mean_shape(aligned_shapes)
        
        # Check for convergence

        if np.sum(np.sqrt((new_mean_shape - mean_shape)**2)) < tolerance:
            break
    
    # Translate all aligned shapes back to the original centroid of the mean shape
    final_aligned_shapes = [shape + np.mean(mean_shape, axis=0) for shape in aligned_shapes]

    return final_aligned_shapes, new_mean_shape



