# Enable Float64 for more stable matrix inversions.
from jax import config

config.update("jax_enable_x64", True)

from dataclasses import dataclass
from typing import Dict

from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
import matplotlib.pyplot as plt
import numpy as np
from simple_pytree import static_field
import tensorflow_probability.substrates.jax as tfp

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.base.param import param_field
    from beartype.typing import Union
    from gpjax.typing import (
    Array,
    ScalarFloat,
    )

tfb = tfp.bijectors
key = jr.key(123)
# plt.style.use(
#     "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
# )
# cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]


from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)

import time
import optax as ox
import random
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi


from .. import BaseModel
from ...sensors.point_sensor import  PointSensor

#------------ END IMPORT STATEMENT --------------

@dataclass
class NN_Kernel(gpx.kernels.AbstractKernel):
    r"""Neural Network Kernel (Neal, 1996; Williams,1998a, 1998b)"""
    
    lx      : Union[ScalarFloat, Float[Array, " D"]] = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    ly      : Union[ScalarFloat, Float[Array, " D"]] = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    sigma_f : float = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    B       : float = param_field(jnp.array(1.0))
    sigma_n : float = param_field(jnp.array(1.0), bijector=tfb.Softplus())

    def __call__(self, X: Float[Array, " D"], Xp: Float[Array, " D"]) -> ScalarFloat:
        varMat = jnp.array([[self.lx, 0],[0,self.ly]]) 
        Sigma = jnp.linalg.inv(jnp.linalg.matrix_power(varMat, 2))
        x = self.slice_input(X)
        xp = self.slice_input(Xp)
        
        num = self.B + 2*jnp.dot(x.T, jnp.dot(Sigma, xp))
        den_xx = 1 + self.B + 2*jnp.dot(x.T, jnp.dot(Sigma, x))
        den_xpxp = 1 + self.B + 2*jnp.dot(xp.T, jnp.dot(Sigma, xp))
        
        K = self.sigma_f**2 * jnp.arcsin( num / jnp.sqrt(den_xx*den_xpxp))
        return K.squeeze()
    

class GPRModel_NNKernel(BaseModel):
    
    def __init__(self, cfg, x_train, y_train) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.neighborhood_size = 10#10
        self.n_clusters = 30
        
        #evaluation points in the environment   
        x_min, x_max, y_min, y_max = self.cfg.task_extent
        num_x, num_y = self.cfg.eval_grid
        x_grid = np.linspace(x_min, x_max, num_x)
        y_grid = np.linspace(y_min, y_max, num_y)
        self.xx, self.yy = np.meshgrid(x_grid, y_grid)
        self.eval_inputs = np.column_stack((self.xx.flatten(), self.yy.flatten()))
        
        #All possible points in the environment
        # num_x = int((self.cfg.env_extent[1] - self.cfg.env_extent[0]) * 5)
        # num_y = int((self.cfg.env_extent[3] - self.cfg.env_extent[2]) * 5)
        # x_grid = np.linspace(x_min, x_max, num_x)
        # y_grid = np.linspace(y_min, y_max, num_y)
        # xx, yy = np.meshgrid(x_grid, y_grid)
        # self.all_env_points = np.column_stack((xx.flatten(), yy.flatten()))
        
        #initialize the estimated terrain
        self.env_predict = np.zeros(self.cfg.eval_grid)
        self.env_std = np.zeros(self.cfg.eval_grid)
        
        #Object to quickly find row,col of the environment
        self.getMatEntries = PointSensor(
                rate=1.0,
                matrix=self.env_predict,
                env_extent=self.cfg.task_extent,
                noise_scale=0,
                rng=np.random.RandomState(42)
                )
        
        # Find the remaining unknown point in the environment
        '''
        Set Methods
        bad -> it randomizes the set
        '''
        # all_env_points = set(map(tuple, np.round(self.eval_inputs, 1)))
        # train_env_points = set(map(tuple, np.round(x_train,1)))
        # test_env_points = all_env_points - train_env_points
        '''
        List Method (better)
        '''
        all_env_list = list(map(tuple, np.round(self.eval_inputs, 1)))
        train_list = list(map(tuple, np.round(x_train,1)))
        test_list = list(filter(lambda x: x not in train_list, all_env_list))
        
        self.test_env_points = np.array(test_list)
        print("initial unknown testing points: ", len(self.test_env_points) )

        #Define training data
        x_train = jnp.asarray(x_train)
        y_train = jnp.asarray(y_train)
        self.D = gpx.Dataset(X=x_train, y=y_train)
                
        #Append to KD-Tree
        dim = 3 #(x,y,z)
        self.kd_tree = KDTree(np.hstack([x_train, y_train]).tolist(), dim)
        
        # Define Prior
        self.zero_mean = gpx.mean_functions.Zero()
        self.kernel = NN_Kernel(active_dims=[0, 1])
        
        #Optimize Posterior
        self.opt_NN_posterior = self.initialise_gp(sigma_n = self.kernel.sigma_n)
        self.optimize(num_steps=450) #450
        
        #Update estimated environment matrix
        latent_dist = self.opt_NN_posterior(x_train, train_data=self.D)
        predict_dist = self.opt_NN_posterior.likelihood(latent_dist)
        
        # print(predict_dist.mean())
        # print(predict_dist.mean().shape)
        self.env_predict = self.enter_mat_entries(x_train, self.env_predict, value=y_train.flatten()) #predict_dist.mean())
        self.env_std = self.enter_mat_entries(x_train, self.env_std, value=predict_dist.stddev())
        
        
        #Kmean- clustering test points into clusters 
        # np.random.shuffle(self.eval_inputs)
        # test_points = self.eval_inputs[:800]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(self.test_env_points)
        self.cluster_centers = kmeans.cluster_centers_
        
        
    def add_data(self, x_new: np.ndarray, y_new: np.ndarray):
        
        #Append training data
        x_new = jnp.asarray(x_new)
        y_new = jnp.asarray(y_new)
        self.D = self.D + gpx.Dataset(X=x_new, y=y_new)
        
        #Add data to KD-Tree
        for point in np.hstack([x_new, y_new]).tolist():
            self.kd_tree.add_point(point) 
        
            
        x_train = np.array(self.D.X)    
        
        '''
        Set Methods
        bad -> it randomizes the list
        '''
        # all_env_points = set(map(tuple, np.round(self.eval_inputs, 1)))
        # train_env_points = set(map(tuple, np.round(x_train,1)))
        # test_env_points = all_env_points - train_env_points
        '''
        List Method (better)
        '''
        print("initial unknown testing points: ", len(self.test_env_points) )
        
        all_env_list = list(map(tuple, np.round(self.eval_inputs, 1)))
        train_list = list(map(tuple, np.round(x_train,1)))
        test_list = list(filter(lambda x: x not in train_list, all_env_list))
        
        self.test_env_points = np.array(test_list)
            
        print("New data appended.")
        print("unknown testing points: ", len(self.test_env_points) )
        
        latent_dist = self.opt_NN_posterior(x_new, train_data=self.D)
        predict_dist = self.opt_NN_posterior.likelihood(latent_dist)
        self.env_predict = self.enter_mat_entries(x_new, self.env_predict, value=y_new.flatten())#predict_dist.mean())
        self.env_std = self.enter_mat_entries(x_new, self.env_std, value=predict_dist.stddev())                
    

    def predict(self, x: np.ndarray):
        
        all_mean = None
        all_std = None
        
        x = np.atleast_2d(x)
        
        if len(x) == (self.cfg.eval_grid[0] * self.cfg.eval_grid[1]):
            print("Reestimate the Environment.")
            
            # np.random.shuffle(self.eval_inputs)
            # test_points = self.eval_inputs[:800] #randomly sample points for evaluation
            
            #Kmean- clustering test points into clusters
            # kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            # kmeans.fit(self.test_env_points)
            # self.cluster_centers = kmeans.cluster_centers_
            # labels = kmeans.labels_
            
            #-------------------------------
            
            '''
            Estimate the environment by cluster
            '''
            
            assignment = self.assign_points_to_centers(self.cluster_centers, self.test_env_points)
            clusters = self.cluster_points(assignments=assignment, arbitrary_points=self.test_env_points)
            
            for cluster_idx, points in clusters.items():
            
            #------------------------------
            # '''
            # Estimate the environment by grid wise
            # '''
            
            # ws = 5
            # c_i = int(ws/2) * ws + int(ws/2)
            # iters  = int(np.sqrt(self.eval_inputs.shape[0] / ws**2))

            # for i in range(iters):
            #     for j in range(iters): 
            
            #------------------------------
            
            # '''
            # Estimate the environment by each points
            # '''
            # for points in self.eval_inputs.tolist():
            
            #------------------------------
            
                
                # print("--------------------------")
                
                print("cluster: ", cluster_idx)
                
                t0 = time.time()
                
                '''
                Change points and data_train_best for each methods
                '''
                
                points = jnp.array(points)
                # points = np.column_stack((self.xx[i*ws: (i+1)*ws, i*ws: (i+1)*ws].flatten(), self.yy[j*ws: (j+1)*ws, j*ws: (j+1)*ws].flatten()))
                # points = jnp.atleast_2d(points)

                data_train_best = jnp.array(self.kd_tree.get_knn(np.append(self.cluster_centers[cluster_idx], 0).tolist(), self.neighborhood_size, return_dist_sq=False))
                # data_train_best = jnp.array(self.kd_tree.get_knn(np.append(points[c_i], 0).tolist(), self.neighborhood_size, return_dist_sq=False))
                # data_train_best = jnp.array(self.kd_tree.get_knn(np.append(points, 0).tolist(), self.neighborhood_size, return_dist_sq=False))


                # t1 = time.time()
                # print("get nearest neighbor: ", t1-t0) 
                
                #--------------------------
                
                # t0 = time.time()
                # # y_train_best = self.getMatEntries.sense(x_train_best)
                # y_train_best = []
                # for i in range(len(x_train_best)):
                #     index = np.argwhere(np.all(self.D.X == x_train_best[i], axis=1))[0][0]
                #     y_train_best.append(self.D.y[index])
                # y_train_best = jnp.array(y_train_best)
                # t1 = time.time()
                # print("get y_train: ", t1-t0) 
                
                #--------------------------

                # t0 = time.time()
                #Closest Neighbor Dataset
                D_train_best = gpx.Dataset(X=data_train_best[:, :2], y=data_train_best[:,2].reshape(-1,1))
                
                # posterior prediction
                latent_dist = self.opt_NN_posterior(points, train_data=D_train_best)
                predict_dist = self.opt_NN_posterior.likelihood(latent_dist)
                
                # t1 = time.time()
                # print("GP Prediction: ", t1-t0) 
                
                #--------------------------
                    
                # t0 = time.time()
                self.env_predict = self.enter_mat_entries(points, self.env_predict, value=predict_dist.mean())
                self.env_std = self.enter_mat_entries(points, self.env_std, value=predict_dist.stddev())
                
                
                # if all_mean is None and all_std is None:
                #     all_mean = np.array(predict_dist.mean()).reshape(-1,1)
                #     all_std = np.array(predict_dist.stddev()).reshape(-1,1)
                # else:
                #     all_mean = np.vstack([all_mean, predict_dist.mean().reshape(-1,1)])
                #     all_std = np.vstack([all_std, predict_dist.stddev().reshape(-1,1)])
                t1 = time.time()
                
                print("Prediction time: ", t1-t0) 
                
                
            
            '''
            Additional estimate the border points of the cluster
            '''
            
            print("re-estimate the border points")
            t0 = time.time()
            np.random.seed(1234)
            
            # compute Voronoi tesselation
            vor = Voronoi(self.cluster_centers) 

            # find border points
            regions, vertices = voronoi_finite_polygons_2d(vor)
            for region in regions:  
                
                subset = vertices[region]
                
                #reestimate the GP
                for point in subset.tolist():
                    
                    point = jnp.atleast_2d(point)
                    
                    data_train_best = jnp.array(self.kd_tree.get_knn(np.append(point, 0).tolist(), self.neighborhood_size, return_dist_sq=False))
                    D_train_best = gpx.Dataset(X=data_train_best[:, :2], y=data_train_best[:,2].reshape(-1,1))
                
                    # posterior prediction
                    latent_dist = self.opt_NN_posterior(point, train_data=D_train_best)
                    predict_dist = self.opt_NN_posterior.likelihood(latent_dist)
                    
                    self.env_predict = self.enter_mat_entries(point, self.env_predict, value=predict_dist.mean())
                    self.env_std = self.enter_mat_entries(point, self.env_std, value=predict_dist.stddev())
                    
            t1 = time.time()
            print("Border points time: ", t1-t0)      
                    
            # return all_mean, all_std
            return self.env_predict.reshape(-1,1), self.env_std.reshape(-1,1)
            
        else:
            
            assignment = self.assign_points_to_centers(self.cluster_centers, x)
            clusters = self.cluster_points(assignments=assignment, arbitrary_points=x)
            
            for cluster_idx, points in clusters.items():
                print("evaluate cluster")
                points = jnp.array(points)

                data_train_best = jnp.array(self.kd_tree.get_knn(np.append(self.cluster_centers[cluster_idx], 0).tolist(), self.neighborhood_size, return_dist_sq=False))
                
                # y_train_best = self.getMatEntries.sense(x_train_best)
                # y_train_best = []
                # for i in range(len(x_train_best)):
                #     index = np.argwhere(np.all(self.D.X == x_train_best[i], axis=1))[0][0]
                #     y_train_best.append(self.D.y[index])
                    
                # y_train_best = jnp.array(y_train_best)

                #Closest Neighbor Dataset
                D_train_best = gpx.Dataset(X=data_train_best[:, :2], y=data_train_best[:,2].reshape(-1,1))
                
                # posterior prediction
                latent_dist = self.opt_NN_posterior(points, train_data=D_train_best)
                predict_dist = self.opt_NN_posterior.likelihood(latent_dist)
                
                if all_mean is None and all_std is None:
                    all_mean = np.array(predict_dist.mean()).reshape(-1,1)
                    all_std = np.array(predict_dist.stddev()).reshape(-1,1)
                else:
                    all_mean = np.vstack([all_mean, predict_dist.mean().reshape(-1,1)])
                    all_std = np.vstack([all_std, predict_dist.stddev().reshape(-1,1)])
                    
            return all_mean, all_std
    
    
    def initialise_gp(self, sigma_n):
        prior = gpx.gps.Prior(mean_function=self.zero_mean, kernel=self.kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.D.n, obs_stddev=sigma_n)
        posterior = prior * likelihood
        return posterior
    

    def optimize(self, num_steps=25, key=key):
        
        # define the MLL
        objective = gpx.objectives.ConjugateMLL(negative=True)
        key = jr.key(123)
        key, subkey = jr.split(key)

        self.opt_NN_posterior, history = gpx.fit(
            model=self.opt_NN_posterior,
            objective=objective,
            train_data=self.D,
            optim = ox.adamw(learning_rate=1e-2),
            num_iters = num_steps,
            key=key
        )
        
        return [0]
    
    def assign_points_to_centers(self, centers, arbitrary_points):
        # Calculate the distance between each arbitrary point and each center point
        distances = np.sqrt(np.sum((arbitrary_points[:, np.newaxis] - centers) ** 2, axis=2))
        
        # Find the index of the closest center point for each arbitrary point
        closest_center_indices = np.argmin(distances, axis=1)
        
        return closest_center_indices
    
    def cluster_points(self, arbitrary_points, assignments):
        clusters = defaultdict(list)
        for idx, assignment in enumerate(assignments):
            clusters[assignment].append(arbitrary_points[idx])
        return clusters
    
    def enter_mat_entries(self, points, matrix_est, matrix_true = None, value= None):
        points = np.asarray(points)
        rows = self.getMatEntries.ys_to_rows(points[:,1])
        cols = self.getMatEntries.xs_to_cols(points[:,0])
            
        if (matrix_true is not None) and (value is not None):
            raise("Must provide either only the ground truth value reference (i.e., matrix_true = ...), or estimated values (i.e., value = ...)")   
        #if value is known from the ground truth environment
        elif matrix_true is not None:
            for row, col in zip(rows, cols):
                matrix_est[row, col] = matrix_true[row, col]
        #if specific value is known
        elif value is not None:
            for i in range(len(points)):
                matrix_est[rows[i], cols[i]] = value[i]
        else:
            raise("Must atleast provide either the ground truth value reference (i.e., matrix_true = ...), or estimated values (i.e., value = ...)")
        
        return matrix_est
    
    
    
#KD Tree class for Searhing Nearest Neighbor 
class KDTree(object):
    
    """
    This implementation only supports Euclidean distance. 

    The points can be any array-like type, e.g: 
        lists, tuples, numpy arrays.

    Usage:
    1. Make the KD-Tree:
        `kd_tree = KDTree(points, dim)`
    2. You can then use `get_knn` for k nearest neighbors or 
       `get_nearest` for the nearest neighbor

    points are be a list of points: [[0, 1, 2], [12.3, 4.5, 2.3], ...]
    """
    def __init__(self, points, dim, dist_sq_func=None):
        """Makes the KD-Tree for fast lookup.

        Parameters
        ----------
        points : list<point>
            A list of points.
        dim : int 
            The dimension of the points. 
        dist_sq_func : function(point, point), optional
            A function that returns the squared Euclidean distance
            between the two points. 
            If omitted, it uses the default implementation.
        """

        if dist_sq_func is None:
            dist_sq_func = lambda a, b: sum((x - b[i]) ** 2 for i, x in enumerate(a[:2]))
                
        def make(points, i=0):
            if len(points) > 1:
                points.sort(key=lambda x: x[i])
                i = (i + 1) % dim
                m = len(points) >> 1
                return [make(points[:m], i), make(points[m + 1:], i), 
                    points[m]]
            if len(points) == 1:
                return [None, None, points[0]]
        
        def add_point(node, point, i=0):
            if node is not None:
                dx = node[2][i] - point[i]
                for j, c in ((0, dx >= 0), (1, dx < 0)):
                    if c and node[j] is None:
                        node[j] = [None, None, point]
                    elif c:
                        add_point(node[j], point, (i + 1) % dim)

        import heapq
        def get_knn(node, point, k, return_dist_sq, heap, i=0, tiebreaker=1):
            if node is not None:
                dist_sq = dist_sq_func(point, node[2])
                dx = node[2][i] - point[i]
                if len(heap) < k:
                    heapq.heappush(heap, (-dist_sq, tiebreaker, node[2]))
                elif dist_sq < -heap[0][0]:
                    heapq.heappushpop(heap, (-dist_sq, tiebreaker, node[2]))
                i = (i + 1) % dim
                # Goes into the left branch, then the right branch if needed
                for b in (dx < 0, dx >= 0)[:1 + (dx * dx < -heap[0][0])]:
                    get_knn(node[b], point, k, return_dist_sq, 
                        heap, i, (tiebreaker << 1) | b)
            if tiebreaker == 1:
                return [(-h[0], h[2]) if return_dist_sq else h[2] 
                    for h in sorted(heap)][::-1]

        def walk(node):
            if node is not None:
                for j in 0, 1:
                    for x in walk(node[j]):
                        yield x
                yield node[2]

        self._add_point = add_point
        self._get_knn = get_knn 
        self._root = make(points)
        self._walk = walk

    def __iter__(self):
        return self._walk(self._root)
        
    def add_point(self, point):
        """Adds a point to the kd-tree.
        
        Parameters
        ----------
        point : array-like
            The point.
        """
        if self._root is None:
            self._root = [None, None, point]
        else:
            self._add_point(self._root, point)

    def get_knn(self, point, k, return_dist_sq=True):
        """Returns k nearest neighbors.

        Parameters
        ----------
        point : array-like
            The point.
        k: int 
            The number of nearest neighbors.
        return_dist_sq : boolean
            Whether to return the squared Euclidean distances.

        Returns
        -------
        list<array-like>
            The nearest neighbors. 
            If `return_dist_sq` is true, the return will be:
                [(dist_sq, point), ...]
            else:
                [point, ...]
        """
        return self._get_knn(self._root, point, k, return_dist_sq, [])

    def get_nearest(self, point, return_dist_sq=True):
        """Returns the nearest neighbor.

        Parameters
        ----------
        point : array-like
            The point.
        return_dist_sq : boolean
            Whether to return the squared Euclidean distance.

        Returns
        -------
        array-like
            The nearest neighbor. 
            If the tree is empty, returns `None`.
            If `return_dist_sq` is true, the return will be:
                (dist_sq, point)
            else:
                point
        """
        l = self._get_knn(self._root, point, 1, return_dist_sq, [])
        return l[0] if len(l) else None

# Helper Functions
# def dist_sq_func(a, b):
#             return sum((x - b[i]) ** 2 for i, x in enumerate(a))

# def get_knn_naive(points, point, k, return_dist_sq=True):
#     neighbors = []
#     for i, pp in enumerate(points):
#         dist_sq = dist_sq_func(point, pp)
#         neighbors.append((dist_sq, pp))
#     neighbors = sorted(neighbors)[:k]
#     return neighbors if return_dist_sq else [n[1] for n in neighbors]

# def get_nearest_naive(points, point, return_dist_sq=True):
#     nearest = min(points, key=lambda p:dist_sq_func(p, point))
#     if return_dist_sq:
#         return (dist_sq_func(nearest, point), nearest) 
#     return nearest

# def rand_point(dim):
#     return [random.uniform(-1, 1) for d in range(dim)]

# Find border points of the K-mean cluster
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)



    