from PIL import Image
import numpy as np
# Deefine image 
from PIL import Image
import glob
import os

def add_padding_and_center(image_path, padding=100):
    # Load the original image
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Calculate new size
    new_height = image_array.shape[0] + 2 * padding
    new_width = image_array.shape[1] + 2 * padding
    
    # Create a new image with white background
    new_image_array = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
    
    # Calculate offset to center the original image
    offset_y = padding
    offset_x = padding
    
    # Place the original image in the center of the new image
    new_image_array[offset_y:offset_y+image_array.shape[0], offset_x:offset_x+image_array.shape[1]] = image_array
    
    # Save the new image
    new_image = Image.fromarray(new_image_array)
    #new_image_path = "centered_" + image_path
    #new_image.save(new_image_path)
    
    return new_image_array, (offset_x, offset_y)

# Example usage
#image_path = leaf.name
#new_image ,offset = add_padding_and_center(image_path, padding=100)
#new_image_array =  np.array(new_image)




############### jax geometry #########################


from jaxgeometry.manifolds.landmarks import *   
from jaxgeometry.Riemannian import metric
from jaxgeometry.dynamics import Hamiltonian
from jaxgeometry.stochastics import Brownian_coords
from jaxgeometry.Riemannian  import Log
from jaxgeometry.dynamics import flow_differential

# flow arbitrary points of N
def ode_Hamiltonian_advect(c,y):
    t,x,chart = c
    qp, = y
    q = qp[0]
    p = qp[1]

    dxt = jnp.tensordot(M.K(x,q),p,(1,0)).reshape((-1,M.m))
    return dxt

# flow arbitrary points of backwards
def ode_Hamiltonian_advect_rev(c,y):
    t,x,chart = c
    qp, = y
    q = qp[0]
    p = qp[1]

    dxt = -jnp.tensordot(M.K(x,q),p,(1,0)).reshape((-1,M.m))
    return dxt



##########################
# Align two sets of points
##########################


import numpy as np
from scipy.optimize import minimize


def align_A_to_B(startpoints, endpoints, inputpoints,transform_params=None):
    # Convert 1D arrays to 2D (x, y) format
    startpoints = np.column_stack([startpoints[::2], startpoints[1::2]])
    endpoints = np.column_stack([endpoints[::2], endpoints[1::2]])
    inputpoints = np.column_stack([inputpoints[::2], inputpoints[1::2]])

    if transform_params is None:
        # Compute centroids
        centroid_start = np.mean(startpoints, axis=0)
        centroid_end = np.mean(endpoints, axis=0)

        def objective_function(params):
            """ Minimize the error between transformed startpoints and endpoints """
            tx, ty, scale, theta = params

            # Create transformation matrix
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

            # Apply transformation: scale → rotate → translate
            transformed_points = ((startpoints - centroid_start) * scale) @ rotation + centroid_start
            transformed_points += np.array([tx, ty])

            # Compute alignment error
            return np.mean(np.linalg.norm(transformed_points - endpoints, axis=1))

        # Initial guesses: translation = centroid difference, scale = 1, rotation = 0
        initial_params = [centroid_end[0] - centroid_start[0], 
                        centroid_end[1] - centroid_start[1], 
                        1.0, 
                        0.0]

        # Optimize transformation parameters and increase sensitivity 
        result = minimize(objective_function, initial_params, method='Nelder-Mead', options={'maxiter': 10000})

        # Extract optimized parameters
        tx, ty, scale, theta = result.x

        # Compute final transformation matrix
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    else: 
        (tx,ty), scale, theta,centroid_start,centroid_end = transform_params["translation"], transform_params["scale"], transform_params["rotation_angle"], transform_params["centroid_start"], transform_params["centroid_end"]
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

    def transform_forward(points):
        """ Apply forward transformation (start to end) """
        transformed = ((points - centroid_start) * scale) @ rotation + centroid_start
        transformed += np.array([tx, ty])
        return transformed

    def transform_reverse(points):
        """ Apply reverse transformation (end to start) """
        # First subtract translation
        points_translated = points - np.array([tx, ty])
        # Reverse rotation and scale
        rotation_reverse = rotation.T  # Transpose for inverse rotation
        transformed = ((points_translated - centroid_start) / scale) @ rotation_reverse + centroid_start
        return transformed

    # Transform points in both directions
    transformed_startpoints_forward = transform_forward(startpoints)
    transformed_inputpoints_forward = transform_forward(inputpoints)
    transformed_startpoints_reverse = transform_reverse(endpoints)
    transformed_inputpoints_reverse = transform_reverse(inputpoints)

    # Return both forward and reverse transformations, along with transformation parameters
    transform_params = {
        'translation': (tx, ty),
        'scale': scale,
        'rotation_angle': theta,
        'centroid_start': centroid_start,
        'centroid_end': centroid_end
    }
    
    return (transformed_startpoints_forward.flatten(), transformed_inputpoints_forward.flatten(),
            transformed_startpoints_reverse.flatten(), transformed_inputpoints_reverse.flatten(),
            transform_params)


##########################
# Apply interpolatoin on two images
##########################

def interpolate_image(startpoints,endpoints, moving_points, iterative_image,image_coords_moved,transform_params=None):

    if transform_params is None:
        _,_,_,transformed_inputpoints,_ = align_A_to_B(endpoints,startpoints,moving_points)
    else:
        _,_,_,transformed_inputpoints,_ = align_A_to_B(endpoints,startpoints,moving_points,transform_params)


    x_coords = transformed_inputpoints[::2]
    y_coords = transformed_inputpoints[1::2]

    height, width = iterative_image.shape[:2]
    # Clip coordinates to image boundaries
    x_coords = np.clip(x_coords, 0, width-1)
    y_coords = np.clip(y_coords, 0, height-1)

    # Get integer coordinates
    x_base = np.floor(x_coords).astype(int)
    y_base = np.floor(y_coords).astype(int)
    
    # Calculate fractional offsets
    x_frac = x_coords - x_base
    y_frac = y_coords - y_base

    # Initialize interpolation result
    interpolated = np.zeros((len(x_coords), 3))

    # Define 2x2 bilinear interpolation function
    @jax.jit
    def interpolate_2x2(x_base, y_base, x_frac, y_frac, iterative_image, width, height):
        # Create meshgrid for 2x2 neighborhood
        i_range = jnp.arange(2)
        j_range = jnp.arange(2)
        i_grid, j_grid = jnp.meshgrid(i_range, j_range)
        
        # Calculate indices for all points at once
        x_idx = jnp.clip(x_base[:, None, None] + i_grid[None, :, :], 0, width-1)
        y_idx = jnp.clip(y_base[:, None, None] + j_grid[None, :, :], 0, height-1)
        
        # Calculate weights for all points
        x_dist = jnp.abs(x_frac[:, None, None] - i_grid[None, :, :])
        y_dist = jnp.abs(y_frac[:, None, None] - j_grid[None, :, :])
        weights = jnp.maximum(0, (1 - x_dist) * (1 - y_dist))
        
        # Gather pixel values and apply weights
        pixels = iterative_image[y_idx, x_idx]
        weighted_sum = jnp.sum(pixels * weights[..., None], axis=(1,2))
        
        # Calculate normalization factor
        valid_weights = jnp.where((jnp.abs(i_grid) < 1)[:, :, None] & 
                                (jnp.abs(j_grid) < 1)[:, :, None],
                                (1 - jnp.abs(i_grid))[:, :, None] * 
                                (1 - jnp.abs(j_grid))[:, :, None],
                                0)
        norm_factor = jnp.sum(valid_weights)
        
        return weighted_sum / norm_factor

    # Use displacement to choose interpolation method
    interpolated= interpolate_2x2(x_base, y_base, x_frac, y_frac, iterative_image, width, height)
        
    # Convert to integers in RGB range (0-255)
    interpolated = np.clip(interpolated.round(), 0, 255).astype(np.uint8)

    # Ensure the number of points matches the image dimensions
    if interpolated.size != height * width * 3:
        # Reshape interpolated to match number of grid points
        interpolated = interpolated.reshape(-1)[:height * width * 3]

    return interpolated.reshape(height, width, 3)



