
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import networkx as nx

def generate_initial_positions(n, d, seed=None):
    mu = np.zeros(d)
    cov = np.identity(d) 

    if seed is not None:
        np.random.seed(seed)
    
    x = np.random.multivariate_normal(mu,cov,n)

    return x

def generate_noisy_versions(x, n, d, sigma2, seed=None, noise_dist = "Gaussian", df = None):
    mu = np.zeros(d)
    cov = np.identity(d)

    if seed is not None:
        np.random.seed(seed)

    match noise_dist:
        case "Gaussian":
            z = np.random.multivariate_normal(mu,sigma2*cov,n)
        case "Cauchy":
            z = np.sqrt(sigma2)*np.random.standard_cauchy(size=(n,d))
        case "t":
            z = np.sqrt(sigma2)*np.random.standard_t(df=df, size=(n,d))
            if df > 2:
                z *= np.sqrt((df-2)/df)
        case "Rademacher":
            z = np.sqrt(sigma2)*(2*np.random.binomial(1,1/2, size=(n,d))-1)
        case "Uniform":
            z = np.sqrt(sigma2)*(np.sqrt(3)*np.random.uniform(-1,1, size=(n,d)))
        case "Spherical":
            z = np.random.multivariate_normal(mu,cov,n)
            z /= np.linalg.norm(z,axis=1)[:,np.newaxis]
            z = np.sqrt(sigma2)*np.sqrt(d)*z
        case "Laplace":
            z = np.sqrt(sigma2)*np.random.laplace(size=(n,d),scale=1/np.sqrt(2))

    y = x + z

    return y

def generate_dataset(n, d, sigma2, seed=None, noise_dist = "Gaussian", df = None):

    if seed is not None:
        np.random.seed(seed)
    
    x = generate_initial_positions(n, d)
    y = generate_noisy_versions(x=x, n=n, d=d, sigma2=sigma2, noise_dist=noise_dist, df=df)
    
    return x,y

def EMV(x,y):
    W = x @ y.T
    row_ind, col_ind = linear_sum_assignment(-W)
    return col_ind

def EMV_mistakes(pi):
    n = len(pi)
    mistakes = []
    for i in range(0, n):
        if pi[i] != i:
            mistakes.append(i)
    
    return mistakes, len(mistakes)

def compute_dist_matrix(x):
    n = len(x)
    inner_prod = x @ x.T
    quads = np.zeros((n,n))
    for i in range(0,n):
        quads[i,:] += inner_prod[i,i]
        quads[:,i] += inner_prod[i,i]

    return np.sqrt(quads - 2*inner_prod)


def geometric_graph(x, r, dist_matrix=None, use_dist_matrix=False):
    
    n = len(x)

    if not use_dist_matrix:
        G = nx.Graph()
        G.add_nodes_from([(i,{'pos':x[i]}) for i in range(0,n)])
        for i in range(0,n):
            for j in range(i+1,n):
                if np.linalg.norm(x[i]-x[j]) <= r:
                    G.add_edge(i,j)
    else:
        if dist_matrix is None:
            dist_matrix_ = compute_dist_matrix(x)
        else:
            dist_matrix_ = dist_matrix
        
        adj_matrix = np.zeros((n,n), dtype=np.int16)
        adj_matrix[dist_matrix_<=r] = 1
        for i in range(0,n):
            adj_matrix[i,i] = 0

        G = nx.from_numpy_array(adj_matrix)
        nx.set_node_attributes(G,{i:x[i] for i in range(0,n)},name="pos")


    return G

def plot_geometric_graph(G,x,matching=None,axs=None,lw=0.1):
    for e in G.edges:
        i,j = e
        if axs:
            axs.arrow(x[i,0],x[i,1],x[j,0]-x[i,0],x[j,1]-x[i,1],head_length=0.0,head_width=0.0, lw=lw, color='blue', linestyle="--")
        else:
            plt.arrow(x[i,0],x[i,1],x[j,0]-x[i,0],x[j,1]-x[i,1],head_length=0.0,head_width=0.0, lw=lw, color='blue', linestyle="--")

    if matching:
        for e in matching:
            i,j = e
            if axs:
                axs.arrow(x[i,0],x[i,1],x[j,0]-x[i,0],x[j,1]-x[i,1],head_length=0.0,head_width=0.0, lw=lw, color='orange')
            else:
                plt.arrow(x[i,0],x[i,1],x[j,0]-x[i,0],x[j,1]-x[i,1],head_length=0.0,head_width=0.0, lw=lw, color='orange')

    if axs is None:
        plt.scatter(x[:,0],x[:,1])

        plt.xticks([])
        plt.yticks([])

        plt.savefig("geometric_graph.pdf")

def get_largest_matching(G):
    matching = nx.max_weight_matching(G)
    return matching, len(matching)

                


    


