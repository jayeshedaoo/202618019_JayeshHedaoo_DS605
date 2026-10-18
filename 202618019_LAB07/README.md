# DS605 Lab 7: Image Representation, PCA, Eigenfaces and Face Recognition

## About the Assignment
This project explores how computers can represent and recognize human faces using image processing and machine learning techniques. As part of the DS605: Fundamentals of Machine Learning course, I worked with the AT&T Database of Faces (ORL) to understand how high-dimensional image data can be transformed into a smaller representation while retaining important information.

The main focus of this assignment is Principal Component Analysis (PCA), also known as the basis for the eigenfaces approach. I implemented PCA using NumPy's linear algebra operations and compared face recognition using raw pixel values, PCA features, and traditional machine learning classifiers.

## Objectives

- Represent grayscale face images as numerical feature vectors.
- Apply mean centering and implement PCA using NumPy.
- Visualize eigenfaces and cumulative explained variance.
- Reduce image dimensionality while preserving at least 90% of the variance.
- Reconstruct face images from their reduced representations.
- Perform face recognition using Euclidean distance.
- Compare recognition accuracy using raw pixels, PCA, KNN, Logistic Regression, and SVM.

## Dataset

This project uses the AT&T Database of Faces, also known as the ORL Face Database.

The dataset contains images of different subjects, with multiple images per subject captured under variations in facial expressions, lighting, and appearance.

Each image is converted to grayscale and represented as a flattened pixel vector. The subject folder is used to assign the corresponding label.

The standard dataset contains 40 subjects and 400 images. The images used in the experiment are checked during loading.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Pillow (PIL)
- Pandas
- Scikit-learn

NumPy is used for the PCA calculations, including eigendecomposition and Singular Value Decomposition (SVD). Scikit-learn is used for the train-test split and traditional machine learning classifiers. Scikit-learn's PCA implementation is not used.

## Implementation

### Part A: Image Representation

The face images are loaded and converted into grayscale pixel matrices. Each image is then flattened into a one-dimensional feature vector, and its subject label is recorded.

The dataset is divided into training and testing sets using a stratified split with a fixed random seed. This ensures that images from every subject are represented in both sets.

### Part B: PCA, Eigenfaces and Compression

First, the mean face is calculated from the training data and subtracted from each training image to center the data.

PCA is implemented using NumPy linear algebra. Eigendecomposition is performed on the smaller Gram matrix to obtain the principal directions efficiently. SVD is also applied to compare the resulting principal components.

The cumulative explained variance is plotted, and the minimum number of components needed to preserve at least 90% of the variance is selected.

The selected components are used to project images into a reduced feature space. Sample faces are then reconstructed from their reduced representations to examine the information retained and lost during dimensionality reduction.

### Part C: Face Recognition

Face recognition is performed using a nearest-neighbor approach. For each test image, Euclidean distances are calculated between its representation and the stored training representations. The subject label of the closest training image is used as the prediction.

The recognition accuracy using raw pixel features is compared with the accuracy obtained using PCA features. Sample predictions are also inspected to identify correct and incorrect matches.

### Part D: Traditional Machine Learning

Three traditional classifiers are trained using the PCA feature representations:

- **K-Nearest Neighbors (KNN):** Classifies a face based on nearby training samples.
- **Logistic Regression:** Predicts the subject using a multiclass classification model.
- **Support Vector Machine (SVM):** Uses an RBF kernel to learn decision boundaries between subjects.

Their accuracies are compared with the nearest-neighbor recognition methods. Different numbers of PCA components are also tested to understand the relationship between dimensionality and recognition performance.

## Results

The experiment generates the following outputs:

| Output | Description |
|---|---|
| `explained_variance.png` | Shows cumulative explained variance and the 90% threshold. |
| `eigenfaces.png` | Visualizes selected principal components as eigenfaces. |
| `reconstruction.png` | Compares original faces with reconstructed faces. |
| `recognition_results.png` | Displays actual and predicted subject labels. |
| `accuracy_vs_components.png` | Shows how the number of PCA components affects KNN accuracy. |
| `comparison_results.csv` | Stores recognition accuracy for the evaluated methods. |

The main numerical results from the experiment are:

- Number of PCA components required for at least 90% variance: **[Enter your value]**
- Nearest-neighbor accuracy using raw pixels: **[Enter your value]%**
- Nearest-neighbor accuracy using PCA features: **[Enter your value]%**
- KNN accuracy using PCA features: **[Enter your value]%**
- Logistic Regression accuracy using PCA features: **[Enter your value]%**
- SVM accuracy using PCA features: **[Enter your value]%**

These values should be filled in using the actual outputs generated by the notebook.

## Observations and Conclusion

This assignment demonstrates how PCA can reduce the dimensionality of face images without requiring every original pixel to be stored in the reduced representation. Eigenfaces provide a visual interpretation of the dominant directions of variation in the training images.

The reconstruction examples help show how much visual information is retained after dimensionality reduction. The recognition experiments also demonstrate that reducing dimensionality can affect classification accuracy, so the most compact representation is not necessarily the most accurate one for face recognition.

Comparing eigendecomposition with SVD provides a practical understanding of two approaches to obtaining principal directions. Finally, comparing nearest-neighbor matching with KNN, Logistic Regression, and SVM illustrates how different classification methods perform on the same PCA features.

Overall, this project connects image representation, linear algebra, dimensionality reduction, and supervised machine learning in a practical face recognition task.


**Jayesh Hedaoo**  
202618019
DS605: Fundamentals of Machine Learning  

