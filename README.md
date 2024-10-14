# Music-Spotify-Classification-Project

This project involves using machine learning techniques to classify music tracks based on various features using a logistic regression model. The dataset contains features such as acousticness, danceability, energy, instrumentalness, and more. The target variable classifies whether a track is liked or not.

## Project Structure

- **music_spotify.csv**: The dataset containing music features and their target labels.
- **notebooks/**: Contains Jupyter notebooks with analysis and model training steps.
- **scripts/**: Contains Python scripts for data preprocessing, model training, and evaluation.

## Dependencies

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the required libraries using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn


Data Description

The dataset includes the following features:

    acousticness: A measure of the acoustic quality of the track.
    danceability: How suitable the track is for dancing.
    duration_ms: Duration of the track in milliseconds.
    energy: Energy level of the track.
    instrumentalness: Predicts whether the track contains instrumental music.
    key: The key the track is in.
    liveness: Presence of audience in the recording.
    loudness: The overall loudness of the track.
    speechiness: Presence of spoken words in the track.
    tempo: Tempo of the track.
    valence: A measure of the musical positiveness of the track.
    target: The target variable indicating if the track is liked or not.

Data Exploration and Visualization

The project begins with visualizing the distribution of various features in the dataset and exploring the relationships between different music characteristics. We use Seaborn to create KDE plots for visual exploration of features like acousticness, danceability, duration_ms, etc., categorized by the target label.
Model Training

The project uses a logistic regression model to classify the music tracks. Here's the general workflow:

    Preprocessing: The target variable is converted to a categorical type.
    Train-Test Split: The data is split into training and test sets (80% training, 20% testing).
    Model: A logistic regression model is trained using the sklearn library.
    Evaluation: The model is evaluated using accuracy scores and confusion matrices.

Hyperparameter Tuning

We evaluate the performance of the model at different probability thresholds (e.g., 0.3, 0.4, 0.5) to adjust the decision boundary.
Feature Engineering

We perform feature transformations such as:

    Log transformation on energy.
    Square root transformation on acousticness.
    Square transformation on danceability.

These transformations help enhance the model's ability to classify music based on transformed features.
Results

After training the models, we display the confusion matrix and accuracy for both the original and transformed models. The results indicate how well the model performs with different feature transformations and thresholds.
Running the Project

To run the project, ensure that you have the required libraries installed. Then, execute the scripts in the following order:

    Preprocess the data: Clean and explore the dataset.
    Train the model: Fit the logistic regression model.
    Evaluate the model: Evaluate accuracy and confusion matrix.

You can also experiment with feature transformations and thresholds to further tune the model.
Conclusion

This project demonstrates how to use logistic regression for binary classification tasks, with a focus on music data. By exploring different feature transformations and evaluating the model at various thresholds, we achieve better performance in classifying the music tracks.
