"""Procrustes alignment utilities.

Two alignment modes are supported, mirroring the original pipeline:

* **Full Generalised Procrustes Analysis (GPA)** -- every landmark contributes
  to the fit (``align_shapes`` with ``idxs=None``). Uses ``scipy.spatial.procrustes``
  (centring + unit-norm scaling + optimal rotation/reflection).
* **Anchored alignment** -- only a chosen subset of landmarks (``idxs``) drives
  the rotation/scale fit, which is then applied to the whole shape
  (``align_shapes`` with an ``idxs`` list). Useful when a few homologous
  landmarks should define the common frame (e.g. wing hinge points).

The low-level :func:`procrustes_with_transform` is a port of
``scipy.spatial.procrustes`` that additionally returns the rotation matrix and
scale factor, originally adapted by Michael Lind Severinsen.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

__all__ = [
    "procrustes_with_transform",
    "align_shapes",
    "to_nd",
    "to_flat",
]


def to_nd(shapes, d):
    """Reshape ``(n_shapes, n_landmarks * d)`` (or a single flat shape) to
    ``(n_shapes, n_landmarks, d)``."""
    arr = np.asarray(shapes, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    n_shapes = arr.shape[0]
    n_landmarks = arr.shape[1] // d
    return arr.reshape(n_shapes, n_landmarks, d)


def to_flat(shapes_nd):
    """Reshape ``(n_shapes, n_landmarks, d)`` back to ``(n_shapes, n_landmarks*d)``."""
    arr = np.asarray(shapes_nd, dtype=float)
    return arr.reshape(arr.shape[0], -1)


def procrustes_with_transform(data1, data2):
    r"""Procrustes analysis returning the transform.

    Standardises ``data1`` and ``data2`` (centre + unit Frobenius norm), then
    finds the rotation ``R`` and scale ``s`` that best map ``data2`` onto
    ``data1`` minimising :math:`\sum (data1 - data2 R^T s)^2`.

    Returns
    -------
    mtx1, mtx2 : ndarray
        Standardised ``data1`` and the transformed ``data2``.
    disparity : float
        Sum of squared pointwise differences.
    R : ndarray
        Optimal orthogonal transform (rotation, possibly with reflection).
    s : float
        Optimal scale factor.
    """
    mtx1 = np.array(data1, dtype=np.float64, copy=True)
    mtx2 = np.array(data2, dtype=np.float64, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    mtx1 /= norm1
    mtx2 /= norm2

    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s
    disparity = np.sum(np.square(mtx1 - mtx2))
    return mtx1, mtx2, disparity, R, s


def _gpa_full(landmarks_nd, n_iter=10):
    """Full GPA: iteratively align every shape to the running mean using all
    landmarks. Returns ``(aligned (n, k, d), reference (k, d))``."""
    reference = np.mean(landmarks_nd, axis=0)
    aligned = landmarks_nd
    for _ in range(n_iter):
        aligned = np.array([procrustes(reference, shape)[1] for shape in landmarks_nd])
        reference = np.mean(aligned, axis=0)
    return aligned, reference


def _gpa_anchored(landmarks_nd, idxs, n_iter=2):
    """Anchored GPA: only ``idxs`` landmarks drive the rotation/scale fit; the
    resulting transform is applied to the full shape. Returns
    ``(aligned (n, k, d), reference (k, d))``."""
    idxs = np.asarray(idxs, dtype=int)
    reference = np.mean(landmarks_nd, axis=0)
    aligned = np.copy(landmarks_nd)
    for _ in range(n_iter):
        new_aligned = []
        for shape in aligned:
            anchors_shape = shape[idxs, :]
            anchors_ref = reference[idxs, :]
            _, _, _, R, s = procrustes_with_transform(anchors_ref, anchors_shape)
            shape_center = np.mean(anchors_shape, axis=0)
            ref_center = np.mean(anchors_ref, axis=0)
            transformed = np.dot(shape - shape_center, R.T) * s + ref_center
            new_aligned.append(transformed)
        aligned = np.array(new_aligned)
        reference = np.mean(aligned, axis=0)
    return aligned, reference


def align_shapes(shapes, d, idxs=None, n_iter=None):
    """Align a set of shapes by Procrustes superimposition.

    Parameters
    ----------
    shapes : array_like
        ``(n_shapes, n_landmarks*d)`` flat coords, ``(n_shapes, n_landmarks, d)``
        already reshaped, or a single shape.
    d : int
        Landmark dimension (2 or 3).
    idxs : sequence of int or None
        If ``None``, full GPA on every landmark. Otherwise, anchor the fit on
        these landmark indices only.
    n_iter : int or None
        Number of refinement iterations. Defaults to 10 (full GPA) or 2
        (anchored), matching the original pipeline.

    Returns
    -------
    aligned : ndarray
        ``(n_shapes, n_landmarks, d)`` aligned shapes.
    reference : ndarray
        ``(n_landmarks, d)`` mean reference shape.
    """
    landmarks_nd = to_nd(shapes, d)
    if idxs is None:
        return _gpa_full(landmarks_nd, n_iter=10 if n_iter is None else n_iter)
    return _gpa_anchored(landmarks_nd, idxs, n_iter=2 if n_iter is None else n_iter)
