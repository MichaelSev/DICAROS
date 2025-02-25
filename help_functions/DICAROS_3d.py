import hyperiax
from jax.random import PRNGKey, split
from jax import numpy as jnp
from hyperiax.execution import LevelwiseTreeExecutor, DependencyTreeExecutor
from hyperiax.models import UpLambda, DownLambda
from hyperiax.models.functional import pass_up
from functools import partial

# Requires Jaxdifferentalgeometry package
from jaxgeometry.manifolds.landmarks import *   
from jaxgeometry.Riemannian import metric
from jaxgeometry.dynamics import Hamiltonian
from jaxgeometry.Riemannian import Log
from jaxgeometry.dynamics import flow_differential


def fuse_lddmm(child_value,child_sigma,child_edge_length, **kwargs):
    d = 3
    def lddmm(childxs1,childxs2,sigma_k,parent_placement):
        # Estimate the avarage distance between each landmark,
        # predefined kernel size 
        sigma_k = jnp.array([sigma_k[0]]*d)
        n_landmarks = jnp.shape(childxs1)[0]//d

        M = landmarks(n_landmarks,k_sigma=sigma_k*jnp.eye(d),m=d) 
        # Riemannian structure

        metric.initialize(M)
        q = M.coords(jnp.array(childxs1))
        v =  (jnp.array(childxs2),[0])
        Hamiltonian.initialize(M)
        # Logarithm map
        Log.initialize(M,f=M.Exp_Hamiltonian)

        # Estimate momentum 
        p = M.Log(q,v)[0]

        # Hamiltonian 
        (_,qps,charts_qp) = M.Hamiltonian_dynamics(q,p,dts(n_steps=100))

        #lift
        flow_differential.initialize(M)
        _,dphis,_ = M.flow_differential(qps,dts())  
        return qps[:,0][parent_placement],p,dphis[parent_placement]


    def true_fn(_):
        parent_placement = jnp.floor(child_edge_length[0] / sum(child_edge_length) * 100).astype(int) - 1
        lddmm_landmarks, p, phi = lddmm(child_value[0], child_value[1], child_sigma[0], parent_placement)
        return parent_placement, lddmm_landmarks, p, phi

    def false_fn(_):
        parent_placement = jnp.floor(child_edge_length[1] / sum(child_edge_length) * 100).astype(int) - 1
        lddmm_landmarks, p, phi = lddmm(child_value[1], child_value[0], child_sigma[1], parent_placement)
        return parent_placement, lddmm_landmarks, p, phi

    parent_placement, lddmm_landmarks, p, phi = jax.lax.cond(
        jnp.less_equal(child_edge_length[0], child_edge_length[1]),
        true_fn,
        false_fn,
        operand=None
    )

    p_out = p / sum(child_edge_length)

    return {'value': lddmm_landmarks, "p_adj": p_out, "phi": phi}
  
  


def fuse_edgelength(child_edge_length,edge_length, **kwargs):

    result = edge_length+(child_edge_length[0]*child_edge_length[1])/(child_edge_length[0]+child_edge_length[1])
    return {'edge_length':result}



def find_sigma3d(landmarks):
    # Reshape the flattened array into a 3D array
    landmarks = landmarks.reshape(-1, 3)
    
    # Calculate the pairwise Euclidean distances
    distances = np.linalg.norm(landmarks[:, np.newaxis] - landmarks, axis=2)
    np.fill_diagonal(distances, np.inf)

    # Find the minimum distance for each row
    min_distances = np.min(distances, axis=0)    
    return np.mean(min_distances)



# Excecution 


#     up_momentum= pass_up('value','sigma','edge_length')
#     upmodel_momentum = UpLambda(up_momentum, fuse_lddmm)
#     root_exe_momentum = DependencyTreeExecutor(upmodel_momentum, batch_size=1)
#     
#     
#     
#      up_correct_edge= pass_up('edge_length')
#     upmodel_edge = UpLambda(up_correct_edge, fuse_edgelength)
#     root_exe_edgelength = DependencyTreeExecutor(upmodel_edge, batch_size=1)   