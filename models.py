import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

class EmailClassifier:
    """Class for training and using the email classification model"""
    
    def __init__(self):
        """Initialize the classifier with default parameters"""
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.classes = None
        self.model_path = 'email_classifier_model.pkl'
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess the dataset for training
        
        Args:
            df: DataFrame containing 'email' and 'type' columns
            
        Returns:
            Processed DataFrame and list of class labels
        """
        # Ensure we have the required columns
        if 'email' not in df.columns or 'type' not in df.columns:
            raise ValueError("DataFrame must contain 'email' and 'type' columns")
        
        # Check for missing values
        missing_emails = df['email'].isna().sum()
        missing_types = df['type'].isna().sum()
        
        if missing_emails > 0 or missing_types > 0:
            print(f"Warning: Found {missing_emails} missing emails and {missing_types} missing types")
            df = df.dropna(subset=['email', 'type'])
        
        # Get list of unique classes
        class_labels = df['type'].unique().tolist()
        
        return df, class_labels
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the classification model
        
        Args:
            df: DataFrame containing 'email' and 'type' columns
            
        Returns:
            Dictionary with training metrics
        """
        # Preprocess data
        df, self.classes = self.preprocess_data(df)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['email'], df['type'], test_size=0.2, random_state=42, stratify=df['type']
        )
        
        # Train the model
        print("Training email classification model...")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Model training complete. Accuracy: {accuracy:.4f}")
        
        # Save the model
        self.save_model()
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'classes': self.classes
        }
    
    def predict(self, email_text: str) -> str:
        """
        Classify an email
        
        Args:
            email_text: The email text to classify
            
        Returns:
            Predicted category
        """
        # Load model if not loaded
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            self.load_model()
        
        # Make prediction
        prediction = self.pipeline.predict([email_text])[0]
        return prediction
    
    def save_model(self) -> None:
        """Save the trained model to disk"""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'classes': self.classes
            }, f)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self) -> None:
        """Load a saved model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.pipeline = model_data['pipeline']
            self.classes = model_data['classes']
        
        print(f"Model loaded from {self.model_path}")

def train_model_from_data(data_path: str) -> EmailClassifier:
    """
    Train the email classifier using the provided dataset
    
    Args:
        data_path: Path to the dataset CSV file
        
    Returns:
        Trained EmailClassifier instance
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize and train classifier
    classifier = EmailClassifier()
    training_metrics = classifier.train(df)
    
    print("Training metrics:", training_metrics)
    return classifier