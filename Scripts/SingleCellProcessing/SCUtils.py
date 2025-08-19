def Protein_normalization(x):
    '''
    CLR normalize the input data using log1p and mean centering.

    Parameters:
    x (array-like): The input data to be normalized. It can be a numpy array, a pandas DataFrame, or a sparse matrix.

    Returns:
    numpy.ndarray: The normalized data.

    '''
    # Import the required dependencies
    try:
        from scipy.sparse import csr_matrix, isspmatrix
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"Missing dependency: {e.name}. Please install it and try again.")

    if isinstance(x, (pd.DataFrame, np.ndarray)) or isspmatrix(x):
        # Convert to CSR matrix
        x = csr_matrix(x)
    else:
        raise ValueError("Input x must be a numpy array, a pandas DataFrame, or a sparse matrix")
    
    # Apply log1p transformation to make the distribution more Gaussian-like
    normalised_counts = np.log1p(x.astype(float))
    
    # Subtract the mean of each row
    normalised_counts = normalised_counts - normalised_counts.mean(axis=1)[:,]
    
    # Convert the result to a numpy array
    normalised_counts = np.asarray(normalised_counts)
    
    return normalised_counts


def Filter_duplicate_vars(dataset):
    '''
    Filter out duplicate variables from the dataset based on their sum and variance.

    Parameters:
    dataset (anndata.AnnData): The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.

    Returns:
    pandas.Index: The names of the variables that were dropped from the dataset.

    '''
    # Import the required dependencies
    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        import anndata
    except ImportError as e:
        print(f"Missing dependency: {e.name}. Please install it and try again.")

    # Check if dataset is an anndata.AnnData object
    if not isinstance(dataset, anndata.AnnData):
        raise ValueError("Input dataset must be an anndata.AnnData object")

    # Get the variable names
    var_names = dataset.var_names

    # Find the unique root names
    root_names = np.unique([name.rsplit('-', 1)[0] for name in var_names if '-' in name])

    # Initialize a dictionary to hold the old and new names
    rename_dict = {}

    # Initialize a list to hold the variables to drop
    drop_vars = []

    # For each root name, find the variable with the highest sum and variance
    for root_name in root_names:
        matching_vars = [name for name in var_names if name.startswith(root_name + '-')]
        if len(matching_vars) > 1:
            # Calculate the sum and variance for each variable
            sums = np.array([dataset[:, var_names == name].X.toarray().sum() for name in matching_vars])
            variances = np.array([dataset[:, var_names == name].X.toarray().var() for name in matching_vars])
            
            # Normalize the sums and variances to the range [0, 1]
            scaler = MinMaxScaler()
            sums = scaler.fit_transform(sums.reshape(-1, 1)).flatten()
            variances = scaler.fit_transform(variances.reshape(-1, 1)).flatten()
            
            # Calculate the scores by adding the normalized sums and variances
            scores = sums + variances
            
            # Find the variable with the highest score
            highest_score_var = matching_vars[np.argmax(scores)]
            
            # Update the rename_dict to keep only the variable with the highest score
            rename_dict[highest_score_var] = root_name
            
            # Add the other variables to the drop_vars list
            drop_vars.extend([var for var in matching_vars if var != highest_score_var])

    # Now, you can use the rename_dict to rename the variables in your dataset
    dataset.var_names_make_unique()
    dataset.var_names = [str(rename_dict.get(name, name)) for name in dataset.var_names]

    # Drop the variables in the drop_vars list from your dataset
    dataset = anndata.AnnData(X=dataset[:, ~dataset.var_names.isin(drop_vars)].X, 
                              var=dataset[:, ~dataset.var_names.isin(drop_vars)].var,
                              obs=dataset.obs,
                              uns=dataset.uns,
                              obsm=dataset.obsm)

    print(rename_dict)

    return pd.Index(drop_vars)


def Dataset_subsampling(dataset, obs_key, N):
    '''
    Subsample each class to same cell numbers (N). Classes are given by obs_key pointing to categorical in dataset.obs.

    Parameters:
    dataset (anndata.AnnData): The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    obs_key (str): The key in dataset.obs that categorizes the data into different classes.
    N (int): The number of samples to draw from each class.

    Returns:
    anndata.AnnData: A new AnnData object with the subsampled data.

    '''
    # Import the required dependencies
    try:
        import numpy as np
        import anndata
    except ImportError as e:
        print(f"Missing dependency: {e.name}. Please install it and try again.")

    # Check if dataset is an anndata.AnnData object
    if not isinstance(dataset, anndata.AnnData):
        raise ValueError("Input dataset must be an anndata.AnnData object")

    # Count the number of occurrences of each class
    counts = dataset.obs[obs_key].value_counts()

    # Initialize an empty list to hold the indices of the selected samples
    indices = []

    # For each class...
    for group in counts.index:
        # Get the indices of the samples in this class
        group_indices = dataset.obs_names[dataset.obs[obs_key] == group]

        # Get the number of samples in this class
        group_size = len(group_indices)

        # If the class size is less than N...
        if group_size < N:
            # Add all the samples from this class to the indices list
            indices.append(group_indices)
        else:
            # Set the random seed for reproducibility
            np.random.seed(42)

            # Randomly select N samples from this class and add their indices to the indices list
            indices.append(np.random.choice(group_indices, size=N, replace=False))

    # Convert the list of arrays into a single array
    selection = np.concatenate(indices)

    # Return a new AnnData object with the subsampled data
    return dataset[selection].copy()


def Intersect_lists(*args):
    '''
    Finds the intersection of any number of pandas Index objects.

    Parameters:
    *args (pandas.Index): Any number of pandas Index objects.

    Returns:
    list: The intersection of all pandas Index objects.
    '''
    # Import the required dependencies
    try:
        import pandas
    except ImportError as e:
        print(f"Missing dependency: {e.name}. Please install it and try again.")

    # Check if all inputs are pandas Index objects
    if not all(isinstance(arr, pandas.Index) for arr in args):
        raise ValueError("All inputs must be pandas Index objects")

    # Convert the first Index to a set
    result = set(args[0])

    # Iterate over each additional Index
    for arr in args[1:]:
        # Update the result to its intersection with the current Index
        result = result.intersection(arr)

    # Convert the result back to a list
    result = list(result)

    # Print the result
    print(result)

    # Return the result
    return result

def reassign_clusters(data, labels):
    '''
    Reassigns the cluster identity for the lower 25% of cells based on silhouette score.

    Parameters:
    data (numpy.ndarray): The data for each cell.
    labels (list): The cluster identity for each cell.

    Returns:
    list: The new cluster identities.
    '''
    # Import the required dependencies
    try:
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import silhouette_samples
    except ImportError as e:
        print(f"Missing dependency: {e.name}. Please install it and try again.")

    # Calculate the silhouette score for each cell
    silhouette_scores = silhouette_samples(data, labels, metric=data.uns['neighbors']['distances'])

    # Get the indices of the lower 25% of cells
    lower_25_indices = np.argsort(silhouette_scores)[:len(silhouette_scores) // 4]

    # Fit the nearest neighbors model to the data
    nbrs = NearestNeighbors(n_neighbors=2).fit(data)

    # Get the indices of the nearest neighbors
    _, indices = nbrs.kneighbors(data)

    # For each cell in the lower 25%, reassign its cluster identity
    for index in lower_25_indices:
        # Get the index of the nearest neighbor with a different cluster identity
        nn_index = next(i for i in indices[index] if labels[i] != labels[index])

        # Reassign the cluster identity
        labels[index] = labels[nn_index]

    return labels


def plot_silhouette_scores(data, labels, new_labels):
    '''
    Plots the silhouette scores before and after reassignment.

    Parameters:
    data (numpy.ndarray): The data for each cell.
    labels (list): The original cluster identities.
    new_labels (list): The new cluster identities.
    '''
    # Import the required dependencies
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import silhouette_score
    except ImportError as e:
        print(f"Missing dependency: {e.name}. Please install it and try again.")

    # Calculate the silhouette scores before and after reassignment
    old_score = silhouette_score(data, labels)
    new_score = silhouette_score(data, new_labels)

    # Create a bar plot of the silhouette scores
    plt.bar(['Before', 'After'], [old_score, new_score])
    plt.ylabel('Silhouette Score')
    plt.show()