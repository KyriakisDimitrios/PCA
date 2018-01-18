# -*- coding: utf-8 -*-
"""
Created on Mar 2017
@author: Kyriakis Dimitrios
PCA:
    1. PCA
    2. PPCA
    3.Kernel PCA
"""
import sys
import numpy as np




###############################################################################
"""
    # 1.PCA
        Principal Component Analysis (PCA) is a multivariate statistical technique
        using sophisticated underlying mathematical principles to transform
        a number of possibly correlated variables into a smaller number of variables
        called principal components
"""
###############################################################################

def PCA(X,M):
    #1. Χ-Χmean
    N = len(X[0])
    mean = np.mean(X,axis=1)
    mean_matrix = np.array([mean]*N)      
    X = (X-mean_matrix.T)   ## Subtract the mean of each entry
    Method = input("Choose between:\n\t1.Singular Value Decomposition (SVD)\n\t2.Eigen Decomposition\n Method (1/2)=")
    if Method =="2":
        S = np.dot(X,X.T)
        S_COV = S/N
        print("The Shape of The Coovariance Matrix is : ".format(np.shape(S_COV)))
        ## EIGEN VECTORS
        eigen_Vals,eigen_Vec = np.linalg.eig(S_COV)
        idx = eigen_Vals.argsort()[::-1]   
        U = eigen_Vec[:,idx[:M]]
    elif Method == "1":
        U,S,V = np.linalg.svd(X, full_matrices=False)
        U = U[:,:M]
    Y = np.dot(U.T,X)
    return Y

###############################################################################
"""
    # 2. PPCA
       Advantasges of Probabilistic PCA:
            • Constrained form of the Gaussian distribution
            • Use of EM algorithm -> computationally efficient
            • Deal with missing data
            • Comparison with other probabilistic models
            • Applied to classification problems
            • Use mixtures of PPCA models
"""
###############################################################################

def PPCA(X,M):
    # Set threshold for Difference between W_old-W_new and S_square_old-S_square_new
    e_raw = input("Choose e (Default=1e-10):\t")
    if e_raw =="":       
        e = 1e-10
    else:
        e = float(e_raw)        
    #1. Χ-Χmean
    D = X.shape[0]
    N = len(X[0])
    mean = np.mean(X,axis=1)
    mean_matrix = np.array([mean]*N)      
    X = (X-mean_matrix.T)   ## Subtract the mean of each entry
    
    # Take Random W and S
    W_choice = input("Load W (Default Random):\n\t (.npy file)/Random(""):\t")
    if W_choice == "" :
        W_old = np.random.rand(D,M) # Take a random matrix
    else: 
        W_old = np.load(W_choice)     # LOAD THE W_5.npy with M=9
    S_Sqaure = input("Choose a S_square (Default random):\t S = ")
    if S_Sqaure == "":
        s_square_old = np.random.rand()    #0.447624035010863   #W_4 S= 0.847
    else:
        s_square_old = float(S_Sqaure)
    ## Set the Diff(W) and Diff_s > e
    Diff =1
    Diff_s = 1 
    count =0
    if s_square_old == 0: 		
        X_tilda = X.T
        # 3. E- step    E -> Ω
        #    Expectation p(x,z|W) = Ω = (W.T*W)^-1*W.Tx.T
        Product1 = np.dot(W_old.T,W_old)
        Product2 = np.dot(W_old.T,X_tilda.T)
        Ω = np.dot(np.linalg.inv(Product1),Product2)
        # 4. M - step     Wnew = Χ*Ω.Τ(Ω*Ω.Τ)^-1
        #	 Σ |Wnew - W|**2 < e	
        W_new = np.dot(np.dot(X_tilda.T,Ω.T),np.linalg.inv(np.dot(Ω,Ω.T)))
        while Diff > e:
            Product1 = np.dot(W_new.T,W_new)
            Product2 = np.dot(W_new.T,X_tilda.T)
            Ω = np.dot(np.linalg.inv(Product1),Product2)
            W_new = np.dot(np.dot(X_tilda.T,Ω.T),np.linalg.inv(np.dot(Ω,Ω.T)))
            Diff = (np.sum(W_new) - np.sum(W_old))**2
            W_old = W_new
            sys.stdout.write("\rDIFF_W: {}".format(Diff))
            sys.stdout.flush()
        
    else:
        iterations = input("Choose num of iterations = ")
        iterat = int(iterations)
        W = np.zeros((D,M))
        for K in range(iterat):
            Diff =1
            Diff_s = 1 
            W_old = np.random.rand(D,M)
            while Diff > e or Diff_s > e :  # Σ |Wnew - W|**2 < e
                # E -> Ω: E- step
                #Expectation p(x,z|W) = Ω = (W.T*W)^-1*W.Tx.T    
                M_matrix = W_old.T.dot(W_old)+s_square_old*np.eye(M,M)   # MxM
                E_z = np.linalg.inv(M_matrix).dot(W_old.T).dot(X)        # MxN
                
                # Create an empty EzzT matrix
                # OR I can compute it outside the for loop:  E_ZZT = s_square_old*(np.linalg.inv(M_matrix)) + E_z.dot(E_z.T)
                E_ZZT = np.zeros((M,M))
                
                s_square = 0 
                for  i in range(len(X[0])):
                    d1 = np.linalg.norm(X[:,i].reshape(1,D))**2
                    d2 = 2*(E_z[:,i].reshape(1,M).dot(W_old.T).dot(X[:,i].reshape(D,1)))
                    E_zzT = s_square_old*(np.linalg.inv(M_matrix)) + (np.dot(E_z[:,i].reshape(M,1),E_z[:,i].reshape(1,M)))
                    d4 = W_old.T.dot(W_old)
                    d5 = np.trace(E_zzT.dot(d4))
                    s_square += np.sum(d1 - d2 + d5)
                    E_ZZT += E_zzT
                s_square_new = s_square/(N*D)
                if s_square_new <0:
                    W_old = np.random.rand(D,M)
                    count +=1
                #M - step
                # Wnew = Χ*Ω.Τ(Ω*Ω.Τ)^-1
                W_new = X.dot(E_z.T).dot(np.linalg.inv(E_ZZT))
                # Caculate Diff
                Diff = abs((np.sum(W_new) - np.sum(W_old)))**2
                Diff_s = abs(s_square_new - s_square_old)
                # Update W and S
                W_old = W_new
                s_square_old = s_square_new
                sys.stdout.write("\rDIFF_W: {}\tDiff_S: {}\t Iteration: {}".format(Diff,Diff_s,K+1))
                sys.stdout.flush()
            W += W_new
        W_new = W/iterat
    # np.save("W_6",W_new) SAVE THE W 
    # SVD: TAKE THE EIGENVECTORS     
    U,S,V = np.linalg.svd(W_new, full_matrices=False)
    # Project our data on M dimensions
    Y = U.T.dot(X)
    return(Y)

###############################################################################
""" 
    # 3.  Kernel PCA
       VC (Vapnik-Chervonenkis) theory tells us that often mappings 
       which take us into a higher dimensional space than the dimension of the 
       input space provide us with greater classification power
"""
###############################################################################

def Kernel(X,M):
    #### PRE-PROCESS
    D = X.shape[0]
    N = len(X[0])
    mean = X.mean(axis=1)
    mean_matrix = np.array([mean]*N)
    X = (X-mean_matrix.T)
    # KERNELS
    Kernel_choice = input("Choose kernel:\n\t1.Gaussian = exp(-γ||xi-xj||**2)\n\t2.Polynomial  = (1+(xixj))**p\n\t3.Hyperbolic tangent = tanh(xixj+δ)\nKernel (1/2/3)= ")
    Kernel_dic = {"1":"γ","2":"p","3":"δ"}
    Kernel_parameter = input("Choose value for {}:\t".format(Kernel_dic[Kernel_choice]))
    b=p=d = float(Kernel_parameter)
    Gaussian = lambda xi,xj : np.exp((-b*(np.linalg.norm(xi-xj)**2)))		
    Polynomial = lambda xi,xj : (1+(np.inner(xi,xj)))**p
    Hyperbolic_tangent = lambda xi,xj : np.tanh(np.inner(xi,xj)+d)
    count =0
    lista=[]
    for i in range(len(X[0])):
        if len(lista) !=0:
            if count == 1:
                l = np.array(lista)
                K = np.array([l])
            else:
                l = np.array([lista])
                K = np.concatenate((K,l),axis=0)
        lista=[]
        count +=1
        for j in range(len(X[0])):
            if Kernel_choice == "1": 
                z = Gaussian(X[:,i],X[:,j])
            elif Kernel_choice == "2":
                z = Polynomial(X[:,i],X[:,j])
            elif Kernel_choice == "3":
                z = Hyperbolic_tangent(X[:,i],X[:,j])
            lista.append(z)
    l = np.array([lista])
    K = np.concatenate((K,l),axis=0)
    print("K.shape = {}".format(K.shape))        
    N_1 = np.ones((N,N))*(1/N)
    Kbar = K - N_1.dot(K) - K.dot(N_1) +N_1.dot(K).dot(N_1)
    eigen_Vals,eigen_Vecs = np.linalg.eig(Kbar)
    idx = eigen_Vals.argsort()[::-1]   
    U = eigen_Vecs[:,idx[:M]]
    return U.T


