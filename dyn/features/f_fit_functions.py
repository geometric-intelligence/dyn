"these functions are useful when doing geodesic regression 
"by transforming curves into q space and linearlly fitting them there.

from geomstats.geometry.discrete_curves import ElasticMetric
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import torch
import matplotlib.pyplot as plt


def plot_ft_compare():
    import dyn.dyn.datasets.synthetic as synthetic

    n_geodesics = 1
    n_times = 10
    n_points = 40

    #creating our synthetic dataset
    geods_square_rect = synthetic.geodesics_square_to_rectangle(
        n_geodesics=n_geodesics, n_times=n_times, n_points=n_points)
    
    # this is creating (aka "instantiating") an object elastic_metric of the class ElasticMetric
    elastic_metric = ElasticMetric(a=1, b=1, ambient_manifold=R2)  

    #selecting the first (and only) geodesic
    #geodesic = geods_square_rect[0]

    q_tensor = elastic_metric.f_transform(geods_square_rect[0])

    q=np.array(q_tensor)

    #reshape q into a compressed vector
    q_vector = q.reshape((n_times, -1))

    #create regression object
    regr = linear_model.LinearRegression()

    #Now, i need to create an array that only has the times
    q_times_1d = gs.arange(0, n_times, 1)
    q_times = np.reshape(q_times_1d,(10,1))

    regr.fit(q_times,q_vector)

    #compute estimated q predictions
    q_vector_predict=regr.predict(q_times)

    #first, we will have to de-compress the vector (turn it back into its original shape)
    q_array_predict = np.reshape(q_vector_predict,(n_times,n_points-1,2))

    #now, we will transform the array back into a tensor so that f_transoform_inverse will accept it
    q_tensor_predict= torch.from_numpy(q_array_predict)

    starting_point_array = gs.zeros((n_times, 2))

    predicted_curves=elastic_metric.f_transform_inverse(q_tensor_predict,starting_point_array)

    #first, i will create a new array, where one of the geodesics is the original geodesic and the other geodesic
    #is the predicted geodesic.

    geodesic_array= np.array([geods_square_rect[0],predicted_curves])

    n_geodesics=2
    fig, axes = plt.subplots(
        n_geodesics, n_times, figsize=(20, 10), sharex=True, sharey=True
    )
    fig.suptitle("Comparison between synthetic and 'q-predicted' geodesics", fontsize=20)

    for i_geodesic in range(n_geodesics):
        curve = geodesic_array[i_geodesic]
        for i_time in range(n_times):
            axes[i_geodesic, i_time].plot(
                curve[i_time][:, 0], curve[i_time][:, 1], marker="o", c=f"C{i_geodesic}"
            )
            axes[i_geodesic, i_time].set_aspect("equal")
    plt.tight_layout()

    return 1

def plot_ft_compare(parameter):
    import dyn.dyn.datasets.synthetic as synthetic

    n_geodesics = 1
    n_times = 10
    n_points = 40

    #creating our synthetic dataset
    geods_square_rect = synthetic.geodesics_square_to_rectangle(
        n_geodesics=n_geodesics, n_times=n_times, n_points=n_points)
    
    # this is creating (aka "instantiating") an object elastic_metric of the class ElasticMetric
    elastic_metric = ElasticMetric(a=1, b=1, ambient_manifold=R2)  

    #selecting the first (and only) geodesic
    #geodesic = geods_square_rect[0]

    q_tensor = elastic_metric.f_transform(geods_square_rect[0])

    q=np.array(q_tensor)

    #reshape q into a compressed vector
    q_vector = q.reshape((n_times, -1))

    #create regression object
    regr = linear_model.LinearRegression()

    #Now, i need to create an array that only has the times
    q_times_1d = gs.arange(0, n_times, 1)
    q_times = np.reshape(q_times_1d,(10,1))

    regr.fit(q_times,q_vector)

    #compute estimated q predictions
    q_vector_predict=regr.predict(q_times)

    #first, we will have to de-compress the vector (turn it back into its original shape)
    q_array_predict = np.reshape(q_vector_predict,(n_times,n_points-1,2))

    #now, we will transform the array back into a tensor so that f_transoform_inverse will accept it
    q_tensor_predict= torch.from_numpy(q_array_predict)

    starting_point_array = gs.zeros((n_times, 2))

    predicted_curves=elastic_metric.f_transform_inverse(q_tensor_predict,starting_point_array)

    #first, i will create a new array, where one of the geodesics is the original geodesic and the other geodesic
    #is the predicted geodesic.

    geodesic_array= np.array([geods_square_rect[0],predicted_curves])

    n_geodesics=2
    fig, axes = plt.subplots(
        n_geodesics, n_times, figsize=(20, 10), sharex=True, sharey=True
    )
    fig.suptitle("Comparison between synthetic and 'q-predicted' geodesics", fontsize=20)

    for i_geodesic in range(n_geodesics):
        curve = geodesic_array[i_geodesic]
        for i_time in range(n_times):
            axes[i_geodesic, i_time].plot(
                curve[i_time][:, 0], curve[i_time][:, 1], marker="o", c=f"C{i_geodesic}"
            )
            axes[i_geodesic, i_time].set_aspect("equal")
    plt.tight_layout()

    return parameter