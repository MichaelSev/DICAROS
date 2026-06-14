


from scipy.optimize import minimize
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes


import numpy as np



def align_species_landmarks(landmarks_list,shape_dim, dim=None):
    """
    Align landmark coordinates for a given species using Procrustes analysis
    
    Args:
        species_subset: DataFrame subset containing only one species
        first_species: Name of the species being processed
        
    Returns:
        aligned_shapes: Array of aligned landmark shapes
        reference: Mean shape used as reference
    """
    # Extract landmark coordinates for the species


    # Convert landmarks to numpy arrays and reshape to (n_shapes, n_landmarks, dim)
    landmarks_arrays = np.array(landmarks_list)
    n_shapes = landmarks_arrays.shape[0]

    n_landmarks = landmarks_arrays.shape[1] // shape_dim
    landmarks_nd = landmarks_arrays.reshape(n_shapes, n_landmarks, shape_dim)

    # Align all shapes using Procrustes
    from scipy.spatial import procrustes
    # Use first shape as initial reference
    reference = np.mean(landmarks_nd, axis=0)
    aligned_shapes = []

    # Repeat alignment 10 times
    for iteration in range(10):
        # Align each shape to current reference
        aligned_shapes = []
        for shape in landmarks_nd:
            _, transformed, _ = procrustes(reference, shape)
            aligned_shapes.append(transformed)
        
        # Update reference as mean of aligned shapes

        aligned_shapes = np.array(aligned_shapes)
        reference = np.mean(aligned_shapes, axis=0)
        
    return aligned_shapes, reference



def align_species_landmarks_with_idx(landmarks_list,idxs,shape_dim, dim=None):
    """
    Align landmark coordinates for a given species using Procrustes analysis
    
    Args:
        species_subset: DataFrame subset containing only one species
        idxs: Indices of landmarks to use for alignment
        
    Returns:
        aligned_shapes: Array of aligned landmark shapes
        reference: Mean shape used as reference
    """
    # Extract landmark coordinates for the species


    # Convert landmarks to numpy arrays and reshape to (n_shapes, n_landmarks, dim)
    landmarks_arrays = np.array(landmarks_list)
    print("landmarks_list", landmarks_arrays.shape)
    n_shapes = landmarks_arrays.shape[0]
    n_landmarks = landmarks_arrays.shape[1] // shape_dim
    landmarks_nd = landmarks_arrays.reshape(n_shapes, n_landmarks, shape_dim)
    print("landmarks_nd", landmarks_nd.shape)
    # Align all shapes using Procrustes
    from scipy.spatial import procrustes
    # Use mean shape as initial reference
    reference = np.mean(landmarks_nd, axis=0)
    aligned_shapes = np.copy(landmarks_nd)

    # Repeat alignment, using only idxs as "anchors" for the transformation fit,
    # but applying the transformation to the full shape in each iteration.
    for iteration in range(2):
        new_aligned_shapes = []
        for shape in aligned_shapes:
            # Fit using only the anchor points
            anchors_shape = shape[idxs, :]
            anchors_ref = reference[idxs, :]

            # Get transform from anchors
            _, R, s = None, None, None
    
            # Assume procrustes_wrs returns (mtx1, mtx2, disp, R, s)
            _, _, _, R, s = procrustes_wrs(anchors_ref, anchors_shape)
      
            # Center the shape so that the transform is correctly applied
            # (mirroring what procrustes does with translation)
            # We'll translate shape to 0, apply rot+scale, then translate to reference mean

            
            shape_center = np.mean(anchors_shape, axis=0)
            ref_center = np.mean(anchors_ref, axis=0)
            shape_centered = shape - shape_center
            shape_transformed = np.dot(shape_centered, R.T) * s + ref_center

            new_aligned_shapes.append(shape_transformed)
        
        aligned_shapes = np.array(new_aligned_shapes)
        reference = np.mean(aligned_shapes, axis=0)


   
    return aligned_shapes, reference






def procrustes_wrs(data1, data2):
    r"""Procrustes analysis, a similarity test for two data sets.
    includes rotation matrix, scale factor


    This module provides functions to perform full Procrustes analysis.

    This code was originally written by Justin Kucynski and ported over from
    scikit-bio by Yoshiki Vazquez-Baeza.

    This code is modified to  output the rotation matrix, the scale factor 
    This code is ported over from scipy.spatial.procrustes by Michael Lind Severinsen 

 

    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both such that:

    - :math:`tr(AA^{T}) = 1`.

    - Both sets of points are centered around the origin.

    Procrustes ([1]_, [2]_) then applies the optimal transform to the second
    matrix (including scaling/dilation, rotations, and reflections) to minimize
    :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
    pointwise differences between the two input datasets.

    This function was not designed to handle datasets with different numbers of
    datapoints (rows).  If two data sets have different dimensionality
    (different number of columns), simply add columns of zeros to the smaller
    of the two.

    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).

    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The orientation of `data2` that best fits `data1`. Centered, but not
        necessarily :math:`tr(AA^{T}) = 1`.
    disparity : float
        :math:`M^{2}` as defined above.

    Raises
    ------
    ValueError
        If the input arrays are not two-dimensional.
        If the shape of the input arrays is different.
        If the input arrays have zero columns or zero rows.

    See Also
    --------
    scipy.linalg.orthogonal_procrustes
    scipy.spatial.distance.directed_hausdorff : Another similarity test
      for two data sets

    Notes
    -----
    - The disparity should not depend on the order of the input matrices, but
      the output matrices will, as only the first output matrix is guaranteed
      to be scaled such that :math:`tr(AA^{T}) = 1`.

    - Duplicate data points are generally ok, duplicating a data point will
      increase its effect on the procrustes fit.

    - The disparity scales as the number of points per input matrix.

    References
    ----------
    .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".
    .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial import procrustes

    The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
    ``a`` here:

    >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
    >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
    >>> mtx1, mtx2, disparity = procrustes(a, b)
    >>> round(disparity)
    0

    """
    mtx1 = np.array(data1, dtype=np.float64, copy=True)
    mtx2 = np.array(data2, dtype=np.float64, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R, s