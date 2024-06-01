# T-shirt-sizes-Clustering-Project
This repository contains code for clustering a large dataset of t-shirts into size categories using Gaussian Mixture Models (GMM) and Principal Component Analysis (PCA). The dataset includes five features for each t-shirt: height, weight, body mass index (BMI), shoulder width, and arm span. The primary goal is to categorize the t-shirts into either 3 sizes (S, M, L) or 5 sizes (XS, S, M, L, XL).

Key Features:
Clustering with GMM: Implements the GMM clustering algorithm to classify t-shirts into predefined size categories.
Dimensionality Reduction with PCA: Applies PCA to reduce the dataset's dimensionality, retaining the most significant features for improved clustering performance.
Custom Implementations: All algorithms are implemented from scratch, without using any built-in machine learning libraries, providing a deep understanding of the underlying mechanics.
Methodology:
Data Collection: The dataset comprises measurements of various t-shirts, characterized by five features: height, weight, BMI, shoulder width, and arm span.
Initial Clustering: The GMM algorithm is used to cluster the t-shirts into 3 or 5 size categories based on the provided features.
Dimensionality Reduction: To refine the clustering process, PCA is applied to reduce the feature set while preserving essential information.
Re-Clustering: The reduced dataset is then re-clustered using GMM to observe any improvements in clustering accuracy and efficiency.
This project is ideal for those interested in understanding clustering algorithms and dimensionality reduction techniques from a foundational perspective, as it provides hands-on experience with implementing these methods from the ground up.

