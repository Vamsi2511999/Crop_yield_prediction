import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, explained_variance_score, mean_squared_log_error
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Load dataset
data = pd.read_csv('wheat_crop_data.csv')

# Define function to predict pest infestation using CNN model
def predict_pest_infestation(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction[0][0]

# Add pest infestation predictions to dataset
data['pest_infestation_prediction'] = data['image_path'].apply(predict_pest_infestation)

# Prepare features and target variables
X = data.drop(['yield', 'pest_infestation', 'image_path'], axis=1)
y_yield = data['yield']
y_pest = data['pest_infestation_prediction']

# Split data into training and testing sets
X_train, X_test, y_yield_train, y_yield_test, y_pest_train, y_pest_test = train_test_split(X, y_yield, y_pest, test_size=0.2, random_state=42)

# Define parameter grid for RandomForestRegressor
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search for best parameters
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_yield_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train RandomForestRegressor with best parameters
best_yield_model = RandomForestRegressor(**best_params, random_state=42)
best_yield_model.fit(X_train, y_yield_train)

# Cross-validation for yield model
yield_cv_scores = cross_val_score(best_yield_model, X_train, y_yield_train, cv=5, scoring='neg_mean_squared_error')

# Train RandomForestRegressor for pest detection
pest_model = RandomForestRegressor(n_estimators=100, random_state=42)
pest_model.fit(X_train, y_pest_train)

# Cross-validation for pest model
pest_cv_scores = cross_val_score(pest_model, X_train, y_pest_train, cv=5, scoring='neg_mean_squared_error')

# Make predictions
best_yield_predictions = best_yield_model.predict(X_test)
pest_predictions = pest_model.predict(X_test)

# Combine results to predict optimum crop output
optimum_output = best_yield_predictions * (1 - pest_predictions)

# Evaluate models
best_yield_mse = mean_squared_error(y_yield_test, best_yield_predictions)
pest_mse = mean_squared_error(y_pest_test, pest_predictions)

# Adjust yield predictions based on pest predictions
yield_weight = 0.7
pest_weight = 0.3
adjusted_yield_predictions = best_yield_predictions * (yield_weight - pest_weight * pest_predictions)
adjusted_yield_mse = mean_squared_error(y_yield_test, adjusted_yield_predictions)

# Print evaluation metrics
print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Score: {best_score}')
print(f'Yield Model Cross-Validation MSE: {-yield_cv_scores.mean()}')
print(f'Pest Detection Model Cross-Validation MSE: {-pest_cv_scores.mean()}')
print(f'Best Yield Model MSE: {best_yield_mse}')
print(f'Pest Detection Model MSE: {pest_mse}')
print(f'Adjusted Yield Model MSE: {adjusted_yield_mse}')
print(f'Optimum Crop Output Predictions: {optimum_output}')

# Additional metrics for evaluation
yield_r2 = best_yield_model.score(X_test, y_yield_test)
pest_r2 = pest_model.score(X_test, y_pest_test)
yield_mae = np.mean(np.abs(best_yield_predictions - y_yield_test))
pest_mae = np.mean(np.abs(pest_predictions - y_pest_test))
yield_rmse = np.sqrt(best_yield_mse)
pest_rmse = np.sqrt(pest_mse)
yield_feature_importances = best_yield_model.feature_importances_
pest_feature_importances = pest_model.feature_importances_
yield_mape = mean_absolute_percentage_error(y_yield_test, best_yield_predictions)
pest_mape = mean_absolute_percentage_error(y_pest_test, pest_predictions)
yield_explained_variance = explained_variance_score(y_yield_test, best_yield_predictions)
pest_explained_variance = explained_variance_score(y_pest_test, pest_predictions)
yield_msle = mean_squared_log_error(y_yield_test, best_yield_predictions)
pest_msle = mean_squared_log_error(y_pest_test, pest_predictions)

# Print additional metrics
print(f'Yield Model R-squared: {yield_r2}')
print(f'Pest Detection Model R-squared: {pest_r2}')
print(f'Yield Model MAE: {yield_mae}')
print(f'Pest Detection Model MAE: {pest_mae}')
print(f'Yield Model RMSE: {yield_rmse}')
print(f'Pest Detection Model RMSE: {pest_rmse}')
print(f'Yield Model Feature Importances: {yield_feature_importances}')
print(f'Pest Detection Model Feature Importances: {pest_feature_importances}')
print(f'Yield Model MAPE: {yield_mape}')
print(f'Pest Detection Model MAPE: {pest_mape}')
print(f'Yield Model Explained Variance: {yield_explained_variance}')
print(f'Pest Detection Model Explained Variance: {pest_explained_variance}')
print(f'Yield Model MSLE: {yield_msle}')
print(f'Pest Detection Model MSLE: {pest_msle}')

# Plot feature importance for yield model
plt.figure(figsize=(10, 6))
plt.barh(X.columns, best_yield_model.feature_importances_)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Yield Model')
plt.show()

# Plot feature importance for pest model
plt.figure(figsize=(10, 6))
plt.barh(X.columns, pest_model.feature_importances_)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Pest Detection Model')
plt.show()

# Plot prediction vs actual for yield model
plt.figure(figsize=(10, 6))
plt.scatter(y_yield_test, best_yield_predictions)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Prediction vs Actual for Yield Model')
plt.plot([min(y_yield_test), max(y_yield_test)], [min(y_yield_test), max(y_yield_test)], color='red')
plt.show()

# Plot prediction vs actual for pest model
plt.figure(figsize=(10, 6))
plt.scatter(y_pest_test, pest_predictions)
plt.xlabel('Actual Pest Infestation')
plt.ylabel('Predicted Pest Infestation')
plt.title('Prediction vs Actual for Pest Detection Model')
plt.plot([min(y_pest_test), max(y_pest_test)], [min(y_pest_test), max(y_pest_test)], color='red')
plt.show()

# Plot residuals for yield model
yield_residuals = y_yield_test - best_yield_predictions
plt.figure(figsize=(10, 6))
plt.scatter(best_yield_predictions, yield_residuals)
plt.xlabel('Predicted Yield')
plt.ylabel('Residuals')
plt.title('Residual Plot for Yield Model')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# Plot residuals for pest model
pest_residuals = y_pest_test - pest_predictions
plt.figure(figsize=(10, 6))
plt.scatter(pest_predictions, pest_residuals)
plt.xlabel('Predicted Pest Infestation')
plt.ylabel('Residuals')
plt.title('Residual Plot for Pest Detection Model')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# Define paths to image dataset
train_dir = 'path/to/train'
validation_dir = 'path/to/validation'

# Image data generator for preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
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
    layers.Dense(1, activation='sigmoid')
])

# Compile CNN model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN model
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)
