import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import inspect

Standarize_vars = {}
init_vars = {}
comp_resp_vars = {}
M_step_vars = {}
gmm_vars = {}
PCA_vars = {}

def PCA(standardized_data):
    global PCA_vars
    covariance_matrix = np.cov(standardized_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_k_indices = sorted_indices[:2]
    selected_eigenvectors = eigenvectors[:, top_k_indices]
    reduced_data = np.dot(standardized_data, selected_eigenvectors)
    new_df=pd.DataFrame()
    new_df["Principle Component 1"]=reduced_data[:,0]
    new_df["Principle Component 2"]=reduced_data[:,1]
    PCA_vars = inspect.currentframe().f_locals
    return new_df,reduced_data

def standarize(df):
    global Standarize_vars
    df_to_array=df.to_numpy()
    data=df_to_array
    mean_vals = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)
    normalized_data = (data - mean_vals) / std_devs
    Standarize_vars = inspect.currentframe().f_locals
    return normalized_data

def init(data,num_clusters):
    global init_vars
    np.random.seed(42)

    # Means initialization
    mean = np.random.randn(num_clusters, data.shape[1])  # Initialize means randomly

    # Covariance initialization
    covariance = np.array([np.eye(data.shape[1])] * num_clusters)  # Initialize with identity matrices

    # Mixing coefficients initialization
    weights = np.ones(num_clusters) / num_clusters 
    
    init_vars = inspect.currentframe().f_locals
    return mean,covariance,weights

def comp_resp(data,num_clusters,mean,covariance,weights):
    global comp_resp_vars
    n_samples=len(data)
    D=len(data[0])
    responsibilities = np.zeros((n_samples, num_clusters))
    for k in range(num_clusters):
        diff = data - mean[k]
        covariance_inverse = np.linalg.inv(covariance[k])
        exponent = np.sum(diff @ covariance_inverse * diff, axis=1)
        det_covariance = np.sqrt(np.linalg.det(covariance[k]))
        responsibilities[:, k] = weights[k] * np.exp(-0.5 * exponent) / (det_covariance * (2 * np.pi) ** (D / 2))

    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    comp_resp_vars = inspect.currentframe().f_locals
    return responsibilities

def M_step(data,num_clusters,mean,covariance,weights):
    global M_step_vars
    converged = False
    iteration = 0   
    max_iterations=1000
    convergence_threshold=0.01
    
    while not converged and iteration < max_iterations:
        #print("Iteration: "+str(iteration))
        
        old_means = mean.copy()
        old_covariance = covariance.copy()
        old_weights = weights.copy()
        
        responsibilities=comp_resp(data, num_clusters, mean, covariance, weights)
        
        for k in range(num_clusters):
            mean[k] = np.sum(responsibilities[:, k, np.newaxis] * data, axis=0) / np.sum(responsibilities[:, k])
        
        # Update covariances
        for k in range(num_clusters):
            diff = data - mean[k]
            covariance[k] = np.dot((responsibilities[:, k, np.newaxis] * diff).T, diff) / np.sum(responsibilities[:, k])
        
        # Update mixing coefficients (weights)
        weights = np.mean(responsibilities, axis=0)
        weights /= np.sum(weights)  # Normalize weights to sum up to 1
        
        # Check convergence based on parameter change threshold
        mean_change = np.linalg.norm(mean - old_means)
        covariance_change = np.linalg.norm(covariance - old_covariance)
        weights_change = np.linalg.norm(weights - old_weights)
        
        if mean_change < convergence_threshold and covariance_change < convergence_threshold and weights_change < convergence_threshold:
            converged = True
        
        iteration += 1
        
    if converged:
        print("Converged after {} iterations.".format(iteration))
    else:
        print("Did not converge after {} iterations.".format(max_iterations))
    M_step_vars = inspect.currentframe().f_locals
    return responsibilities

def gmm(data,num_clusters,df):
    #start_time = timeit.default_timer()
    global gmm_vars
    normalised_data=data
    
    mean,covariance,weights=init(normalised_data, num_clusters)
    
    responsibilities=M_step(normalised_data, num_clusters, mean, covariance, weights)
    
    cluster_labels = np.argmax(responsibilities, axis=1)
    the_labeled_size=['']*len(cluster_labels)
    for i in range(0,len(cluster_labels)):
        if(num_clusters==3):
            if(cluster_labels[i]==0):
                the_labeled_size[i]='L'
            elif cluster_labels[i]==1 :
                the_labeled_size[i]='M'
            else:
                the_labeled_size[i]='S'
        else:
            if(cluster_labels[i]==0):
                the_labeled_size[i]='XL'
            elif cluster_labels[i]==1 :
                the_labeled_size[i]='L'
            elif cluster_labels[i]==2 :
                the_labeled_size[i]='M'    
            elif cluster_labels[i]==3 :
                the_labeled_size[i]='S'    
            else:
                the_labeled_size[i]='XS'
    new_df=df.copy()          
    new_df["Size"]=the_labeled_size 
    gmm_vars = inspect.currentframe().f_locals
    return new_df,cluster_labels

df=pd.read_csv("D:\GUC\sem7\Analysis and Design of Algorithms\DataSet.csv")
num_clusters=(int)(input("Please Enter the K value: "))

standardized_data=standarize(df)

df,reduced_data=PCA(standardized_data)

new_df,cluster_labels=gmm(reduced_data, num_clusters, df)

plt.figure(figsize=(8, 6))
unique_clusters = np.unique(cluster_labels)
for cluster_label in unique_clusters:
    samples_in_cluster = reduced_data[cluster_labels == cluster_label]
    plt.scatter(samples_in_cluster[:, 0], samples_in_cluster[:, 1], label=f'Cluster {cluster_label}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Samples after GMM with Cluster Assignments')
plt.legend()
plt.grid(True)
plt.show()

#percentage of each cluster from the total N samples.
unique, counts = np.unique(cluster_labels, return_counts=True)
for i in range(0,num_clusters):
    prob=counts[i]/len(standardized_data)
    print("Cluster "+str(i)+": Probability = "+str(prob))
   