import jax
import jax.numpy as jnp
import numpy as np


def lift_p(tree, n):
    # Lift P 
    p_all = [tree.root["p_adj"]]

    for level in tree.iter_levels():
        # Skip root
        if len(level) == 1:
            continue

        for node in level:
            org_node = node
            if len(org_node.children) == 0:
                continue

            node = org_node.parent
        
            p_lifted = jax.vmap(lambda A,v: jnp.dot(A.T,v))(org_node.data["phi"],org_node.data["p_adj"].reshape((n,2)))
            while node.parent is not None: 
                p_lifted = jax.vmap(lambda A,v: jnp.dot(A.T,v))(node.parent.data["phi"],p_lifted)
                node = node.parent
            org_node["p_lifted"] = p_lifted

            p_all.append(p_lifted.flatten())
    
    return p_all

   
def compute_pca_and_update_tree(tree, p_all, n, n_components=5):
    # Initialize Riemannian metrics for estimating the covariance matrix
    from jaxgeometry.manifolds.landmarks import landmarks   
    M = landmarks(n, k_sigma=1*jnp.eye(2))
    x = M.coords(tree.root["value"])
    
    # Riemannian structure
    from jaxgeometry.Riemannian import metric
    metric.initialize(M)

    # Compute covariance matrix
    cov_uplddmm = jnp.zeros((n*2,n*2))
    for i in range(len(p_all)):
        cov_uplddmm += np.dot(np.dot(p_all[i].reshape(n*2,-1), 
                                    p_all[i].reshape(-1,n*2)), M.g(x))
    cov_uplddmm = cov_uplddmm/(len(p_all)/2)

    # Compute PCA
    eigenvalues, eigenvectors = np.linalg.eig(cov_uplddmm)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components and get landmarks
    components = eigenvectors[:, :n_components]
    landmarks = [leaf.data["value"] for leaf in tree.iter_bfs()]
    
    # Project landmarks onto components
    X_projected = np.dot(landmarks, np.real(components))
    
    # Calculate variance explained
    varexp = eigenvalues[:n_components]/sum(eigenvalues)
    
    # Update tree with projected values
    for i, leaf in enumerate(tree.iter_bfs()):
        leaf["PCs"] = X_projected[i,:]
        
    return varexp,X_projected