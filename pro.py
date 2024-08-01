import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.feature_stats = None
        self.class_priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # Calculate class priors
        self.class_priors = {c: np.sum(y == c) / n_samples for c in self.classes}
        
        # Calculate feature statistics
        self.feature_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        for c in self.classes:
            X_c = X[y == c]
            for feature in range(n_features):
                feature_values = X_c[:, feature]
                mean = np.mean(feature_values)
                std = np.std(feature_values)
                self.feature_stats[c][feature]['mean'] = mean
                self.feature_stats[c][feature]['std'] = std

    def _calculate_likelihood(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2 / (2 * (std ** 2))))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = {}
            for c in self.classes:
                class_scores[c] = np.log(self.class_priors[c])
                for feature, value in enumerate(x):
                    mean = self.feature_stats[c][feature]['mean']
                    std = self.feature_stats[c][feature]['std']
                    likelihood = self._calculate_likelihood(value, mean, std)
                    class_scores[c] += np.log(likelihood)
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)

# Example usage
if __name__ == "__main__":
    # Sample weather data: [temperature, humidity]
    X_train = np.array([
        [25, 70], [30, 80], [20, 60], [15, 75], [28, 65],
        [22, 82], [18, 58], [26, 72], [32, 85], [24, 68]
    ])
    y_train = np.array(['No rain', 'Rain', 'No rain', 'Rain', 'No rain',
                        'Rain', 'No rain', 'No rain', 'Rain', 'No rain'])

    # Create and train the Naive Bayes classifier
    nb_classifier = NaiveBayes()
    nb_classifier.fit(X_train, y_train)

    # Make predictions
    X_test = np.array([[23, 75], [31, 83], [19, 62]])
    predictions = nb_classifier.predict(X_test)

    print("Predictions:")
    for i, pred in enumerate(predictions):
        print(f"Weather condition {i+1}: {pred}")