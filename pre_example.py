import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# 데이터 전처리 함수
def preprocess_data(train_csv_path, test_csv_path, train_image_dir, test_image_dir, batch_size=64):
    # Load train and test dataframes
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Train-test split
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_data,
        directory=train_image_dir,
        x_col="filename",
        y_col="risk",
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="binary",
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_data,
        directory=train_image_dir,
        x_col="filename",
        y_col="risk",
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="binary",
    )

    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        directory=test_image_dir,
        x_col="filename",
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )

    return train_generator, val_generator, test_generator

# 모델 학습 함수
def train_model(model, train_generator, val_generator, num_epochs=10, lr=0.001):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_generator,
                        epochs=num_epochs,
                        validation_data=val_generator)

    return history

# 모델 성능 확인 함수
def evaluate_model(model, val_generator):
    predictions = model.predict(val_generator)
    val_labels = val_generator.labels
    val_preds = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(val_labels, val_preds)
    print(f"Accuracy: {accuracy}")
    return accuracy

# 테스트 데이터셋 예측 후 CSV 파일로 저장하는 함수
def predict_and_save(model, test_generator, output_csv_path):
    predictions = model.predict(test_generator)
    test_preds = (predictions > 0.5).astype(int)

    # Create a DataFrame with predictions
    submission_df = pd.DataFrame({"filename": test_generator.filenames, "prediction": test_preds.flatten()})
    submission_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

# 예시: 사용할 모델 정의 (이 예제에서는 간단한 CNN 모델 사용)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

train_csv_path = "data/train/train.csv"
test_csv_path = "data/test/test.csv"
train_image_dir = "data/train/images"
test_image_dir = "data/test/images"
output_csv_path = "submission.csv"

# 데이터 전처리
train_generator, val_generator, test_generator = preprocess_data(train_csv_path, test_csv_path, train_image_dir, test_image_dir)

# 모델 학습
history = train_model(model, train_generator, val_generator, num_epochs=10, lr=0.001)

# 모델 성능 확인
evaluate_model(model, val_generator)

# 테스트 데이터셋 예측 후 CSV 파일로 저장
predict_and_save(model, test_generator, output_csv_path)
