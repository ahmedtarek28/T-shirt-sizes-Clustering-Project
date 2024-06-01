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
        
    if(flag==False):
        if converged:
            print("Converged after {} iterations.".format(iteration))
        else:
            print("Did not converge after {} iterations.".format(max_iterations))
    M_step_vars = inspect.currentframe().f_locals        
    return responsibilities

def gmm(df,num_clusters):
    global gmm_vars
    #start_time = timeit.default_timer()

    normalised_data=standarize(df)
    
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
    new_df=pd.DataFrame()
    new_df=df.copy()          
    new_df["Size"]=the_labeled_size 
    
    #elapsed = timeit.default_timer() - start_time
    
    gmm_vars = inspect.currentframe().f_locals
    if flag==False:
        gmm_vars = inspect.currentframe().f_locals
        return new_df
  

df=pd.read_csv("data set path")
num_clusters=(int)(input("Please Enter the K value: "))
f=(input("To plot the runtime curve, type y, else type n: ")).lstrip()
flag=False
if f=="y":
    flag=True
if flag==True:
    runtimes = []
    sample_sizes = np.arange(10000,100001,10000)    
    for size in sample_sizes:
        start_time = timeit.default_timer()
        gmm(df[0:size],num_clusters)
        print(str(size) + " Finished")    
        runtime=timeit.default_timer() - start_time
        runtimes.append(runtime) 
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, runtimes, marker='o')
    plt.title('Runtime Curve for GMM with Varying Sample Sizes for K= '+str(num_clusters))
    plt.xlabel('Number of Samples')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)
    plt.show() 
    
else:
    new_df=gmm(df,num_clusters)
           
    
        
    
    




    