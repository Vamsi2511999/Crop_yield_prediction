import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, explained_variance_score, mean_squared_log_error
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


# Define paths to image dataset
train_dir = '../dataset/Maize/train_set'
validation_dir = '../dataset/Maize/test_set'

# Image data generator for preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='softmax')
])

# Compile CNN model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('pest_prediction_model.h5')

# Define function to predict pest infestation using CNN model
def predict_pest_infestation(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)  # Get the class index with highest probability
    return class_index

class_labels = ['fall armyworm', 'grasshopper', 'healthy', 'leaf beetle', 'leaf blight', 'leaf spot', 'streak virus']


# Load dataset
data = pd.read_csv('../dataset/crop_yield.csv')
#Adding images path of the data as column
image_folder = '../dataset/maize_leaf_images'
image_files = os.listdir(image_folder)
selected_images = random.sample(image_files, len(data))
image_paths = [os.path.join(image_folder, img) for img in selected_images]
data['Image_Path'] = image_paths

# Add pest infestation predictions to dataset
data['pest_class_index'] = data['image_path'].apply(predict_pest_infestation)
data['Pest_Type'] = data['pest_class_index'].apply(lambda x: class_labels[x])

severity_mapping = {
    'Healthy': 0.0,
    'fall armyworm': 0.2,
    'grasshopper': 0.3,
    'leaf beetle': 0.4,
    'leaf blight': 0.5,
    'leaf spot': 0.6,
    'streak virus': 0.7
}

data['Pest_Severity'] = data['Pest_Type'].map(severity_mapping)
data['Adjusted_Yield'] = data['Yield'] * (1 - data['Pest_Severity'])

# Prepare features and target variables
X = data.drop(['Yield', 'pest_class_index', 'Pest_Type', 'Pest_Severity', 'Adjusted_Yield', 'image_path'], axis=1)
y_adjusted_yield = data['Adjusted_Yield']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_adjusted_yield, test_size=0.2, random_state=42)

# Define parameter grid for RandomForestRegressor
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search for best parameters
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train RandomForestRegressor with best parameters
best_yield_model = RandomForestRegressor(**best_params, random_state=42)
best_yield_model.fit(X_train, y_train)

# Cross-validation for yield model
yield_cv_scores = cross_val_score(best_yield_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Make predictions
best_yield_predictions = best_yield_model.predict(X_test)

# Evaluate models
best_yield_mse = mean_squared_error(y_test, best_yield_predictions)
mae = np.mean(np.abs(best_yield_predictions - y_test))
rmse = np.sqrt(best_yield_mse)

# Print evaluation metrics
print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Score: {best_score}')
print(f'Adjusted Yield Model Cross-Validation MSE: {-yield_cv_scores.mean()}')
print(f'Mean Squared Error (MSE): {best_yield_mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Additional metrics for evaluation
yield_r2 = best_yield_model.score(X_test, y_test)
yield_feature_importances = best_yield_model.feature_importances_
yield_mape = mean_absolute_percentage_error(y_test, best_yield_predictions)
yield_explained_variance = explained_variance_score(y_test, best_yield_predictions)
yield_msle = mean_squared_log_error(y_test, best_yield_predictions)

# Print additional metrics
print(f'Yield Model R-squared: {yield_r2}')
print(f'Yield Model MAE: {mae}')
print(f'Yield Model RMSE: {rmse}')
print(f'Yield Model Feature Importances: {yield_feature_importances}')
print(f'Yield Model MAPE: {yield_mape}')
print(f'Yield Model Explained Variance: {yield_explained_variance}')
print(f'Yield Model MSLE: {yield_msle}')

# Plot feature importance for yield model
plt.figure(figsize=(10, 6))
plt.barh(X.columns, best_yield_model.feature_importances_)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Yield Model')
plt.show()

# Plot prediction vs actual for yield model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_yield_predictions)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Prediction vs Actual for Yield Model')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

# Plot residuals for yield model
yield_residuals = y_test - best_yield_predictions
plt.figure(figsize=(10, 6))
plt.scatter(best_yield_predictions, yield_residuals)
plt.xlabel('Predicted Yield')
plt.ylabel('Residuals')
plt.title('Residual Plot for Yield Model')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# # Plot residuals for pest model
# pest_residuals = y_pest_test - pest_predictions
# plt.figure(figsize=(10, 6))
# plt.scatter(pest_predictions, pest_residuals)
# plt.xlabel('Predicted Pest Infestation')
# plt.ylabel('Residuals')
# plt.title('Residual Plot for Pest Detection Model')
# plt.axhline(y=0, color='red', linestyle='--')
# plt.show()
