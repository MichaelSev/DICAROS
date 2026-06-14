

from hyperiax.execution import OrderedExecutor
from hyperiax.models import UpLambdaReducer, DownLambda, UpLambda
from hyperiax.models.functional import pass_up
from hyperiax.plotting import plot_tree_text, plot_tree_2d_scatter


from functools import partial

# Requires Jaxdifferentalgeometry package
from jaxgeometry.manifolds.landmarks import *   
from jaxgeometry.Riemannian import metric
from jaxgeometry.dynamics import Hamiltonian
from jaxgeometry.Riemannian import Log
from jaxgeometry.dynamics import flow_differential




def fuse_DICAROS(value,child_value, child_sigma, child_edge_length,**kwargs):
    def _DICAROS_with_dim(child_coords1, child_coords2, kernel_sigma, parent_index, shape_dim_static):
      
        # Define landmark space 
        M = landmarks(
            jnp.shape(child_coords1)[0]//shape_dim_static,
            k_sigma= kernel_sigma * jnp.eye(shape_dim_static),
            m=shape_dim_static,
        )
        # Riemannian structure

        metric.initialize(M)
        q = M.coords(jnp.array(child_coords1))
        v = (jnp.array(child_coords2), [0])
 
        Hamiltonian.initialize(M)
        # Logarithm map
        Log.initialize(M, f=M.Exp_Hamiltonian)


        # Estimate momentum
        p = M.Log(q, v)[0]
   
        # Hamiltonian
        (_, qps, charts_qp) = M.Hamiltonian_dynamics(q, p, dts(n_steps=100))

        #lift
        flow_differential.initialize(M)
        _,dphis,_ = M.flow_differential(qps,dts())  

        # Return landmarks, contrast and phi
        return qps[:,0][parent_index],p,dphis[parent_index]

    def DICAROS(child_coords1, child_coords2, kernel_sigma, parent_index):
        # Use static shape metadata (2 or 3) from kernel_sigma
        shape_dim_static = kernel_sigma.shape[0]
        return _DICAROS_with_dim(child_coords1, child_coords2, kernel_sigma, parent_index, shape_dim_static)

    # Map function, to determine the shortest branch from child to parent
    def map_function(child_coords_pair, child_edge_length_pair, kernel_sigma_pair,):
        edge_length_sum = child_edge_length_pair.sum()

        def true_fun(_):
            parent_index = jnp.floor(child_edge_length_pair[0] / edge_length_sum * 100 - 1).astype(int)[0]
            return DICAROS(child_coords_pair[0, :], child_coords_pair[1, :], kernel_sigma_pair[0,:], parent_index)

        def false_fun(_):
            parent_index = jnp.floor(child_edge_length_pair[1] / edge_length_sum * 100 - 1).astype(int)[0]
            return DICAROS(child_coords_pair[1, :], child_coords_pair[0, :], kernel_sigma_pair[0,:], parent_index)

        # Run DICAROS; from either left or right

        return jax.lax.cond((child_edge_length_pair[0] < child_edge_length_pair[1])[0], true_fun, false_fun, operand=None)

    # vmap to iterate over set of children
    qps,p,dphis = jax.vmap(map_function, in_axes=(0, 0, 0))(child_value, child_edge_length, child_sigma)

    # P should be divided with the sum of p

    p_adj= p/jnp.sum(child_edge_length,axis=1)

    return {'value': qps,'p_adj':p_adj,'phi':dphis}


def fuse_edgelength(edge_length, child_edge_length, **kwargs):

    prod = jnp.prod(child_edge_length, axis=1)
    s = jnp.sum(child_edge_length, axis=1)
    result = edge_length + prod / s
    return {'edge_length': result}


def find_sigma(landmarks, d):
    arr = np.array(landmarks, dtype=float).reshape(-1, d)
    distances = np.linalg.norm(arr[:, np.newaxis] - arr, axis=2)
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    return np.mean(min_distances)



def tree_initialization(tree,out_df,d):
    mean_species = out_df.iloc[:, 0]   # first column: species names
    mean_counts = out_df.iloc[:, 1]    # second column: counts (if needed)
    mean_shapes = out_df.iloc[:, 2:]   # shape data

    # estimate sigma
    sigma = np.mean([find_sigma(row,d) for _, row in mean_shapes.iterrows()])


    ###
    # Add properties to tree for shape reconstruction
    ##
    # The name of the landmarks are placed in the tree, alongside with the edgelength 
    #landmark = landmark.iloc[:,0:10] # reduce landmark numbers 
    l_dim = int((jnp.shape(mean_shapes)[1]))
    tree.add_property('value', shape=(l_dim,))
    # add landmarks to tree
    tree.add_property('p_adj', shape=(l_dim,))
    tree.add_property('phi', shape=(int(l_dim/d),d,d))

    # How many compartments in the output pca
    tree.add_property('pc', shape=(5,))

    # Saved parameters within the tree 
    tree.add_property('sigma', shape=(d,))  # 1d array, shape=() gives single float value per node
    tree.data['sigma'] = tree.data['sigma'].at[:].set(jnp.array([sigma]*d))


    ###
    # Reorder mean_shapes to match the order from my_nodenames,  by matching my_nodenames to mean_species
    ##

    my_nodenames = []
    for node in tree.iter_topology_bfs():
        if len(node.children)<1:
            my_nodenames.append(node.name)

    mean_species_array = mean_species.to_numpy()
    # Build index array: for each name in my_nodenames, find index in mean_species
    matching_indices = np.array([np.where(mean_species_array == name)[0][0] for name in my_nodenames])
    reordered_mean_shapes = mean_shapes.iloc[matching_indices, :]

    # Replace value
    tree.data['value'] = tree.data['value'].at[tree.is_leaf].set(jnp.array(reordered_mean_shapes, dtype=float))

    return tree