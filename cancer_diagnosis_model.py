import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CancerDiagnosisNLPModel:
    """
    A comprehensive NLP-based model for predicting cancer probability 
    based on patient conditions, symptoms, and medical history.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.best_model = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('punkt_tab')
            print("NLTK data downloaded successfully.")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess medical text data for NLP analysis.
        
        Args:
            text (str): Raw medical text
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        
        # Remove stopwords but keep medical relevant words
        medical_stopwords = set(stopwords.words('english')) - {
            'pain', 'no', 'not', 'never', 'always', 'very', 'more', 'less',
            'much', 'severe', 'mild', 'chronic', 'acute', 'sudden'
        }
        
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in medical_stopwords and len(token) > 2]
        
        return ' '.join(tokens)
    
    def create_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create synthetic medical data for training the model.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Synthetic medical dataset
        """
        np.random.seed(42)
        
        # Define medical symptoms and conditions
        cancer_symptoms = [
            'unexplained weight loss', 'persistent fatigue', 'fever', 'night sweats',
            'persistent cough', 'difficulty swallowing', 'changes in bowel habits',
            'unusual bleeding', 'lumps or swelling', 'persistent pain',
            'skin changes', 'persistent headaches', 'vision changes',
            'difficulty breathing', 'hoarseness', 'indigestion', 'bloating'
        ]
        
        benign_symptoms = [
            'mild headache', 'common cold', 'seasonal allergies', 'muscle strain',
            'minor cuts', 'temporary fatigue', 'stress', 'insomnia',
            'mild anxiety', 'occasional nausea', 'heartburn', 'joint stiffness',
            'minor back pain', 'eye strain', 'dry skin', 'common flu'
        ]
        
        risk_factors = [
            'smoking history', 'family history of cancer', 'radiation exposure',
            'chemical exposure', 'age over 50', 'obesity', 'alcohol consumption',
            'genetic mutations', 'immunosuppression', 'chronic inflammation'
        ]
        
        data = []
        
        for i in range(n_samples):
            # Generate patient demographics
            age = np.random.randint(20, 85)
            gender = np.random.choice(['male', 'female'])
            smoking = np.random.choice([0, 1], p=[0.7, 0.3])
            family_history = np.random.choice([0, 1], p=[0.8, 0.2])
            
            # Generate cancer probability based on risk factors
            cancer_prob = 0.1  # Base probability
            if age > 50:
                cancer_prob += 0.2
            if smoking:
                cancer_prob += 0.3
            if family_history:
                cancer_prob += 0.2
            
            # Add some randomness
            cancer_prob += np.random.normal(0, 0.1)
            cancer_prob = max(0, min(1, cancer_prob))
            
            # Determine if patient has cancer
            has_cancer = np.random.random() < cancer_prob
            
            # Generate symptoms based on cancer status
            if has_cancer:
                n_symptoms = np.random.randint(2, 5)
                symptoms = np.random.choice(cancer_symptoms + risk_factors, n_symptoms)
                # Add some benign symptoms to make it realistic
                n_benign = np.random.randint(0, 2)
                if n_benign > 0:
                    benign = np.random.choice(benign_symptoms, n_benign)
                    symptoms = np.concatenate([symptoms, benign])
            else:
                n_symptoms = np.random.randint(1, 3)
                symptoms = np.random.choice(benign_symptoms, n_symptoms)
                # Occasionally add a cancer symptom to non-cancer cases
                if np.random.random() < 0.2:
                    cancer_symptom = np.random.choice(cancer_symptoms, 1)
                    symptoms = np.concatenate([symptoms, cancer_symptom])
            
            # Create medical history text
            medical_history = f"Patient reports {', '.join(symptoms)}. "
            if smoking:
                medical_history += "History of smoking. "
            if family_history:
                medical_history += "Family history of cancer. "
            
            data.append({
                'patient_id': f'P{i+1:04d}',
                'age': age,
                'gender': gender,
                'smoking_history': smoking,
                'family_history': family_history,
                'medical_history': medical_history,
                'symptoms': ', '.join(symptoms),
                'cancer_diagnosis': int(has_cancer)
            })
        
        return pd.DataFrame(data)
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from the dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and target variable
        """
        # Preprocess text data
        df['processed_history'] = df['medical_history'].apply(self.preprocess_text)
        df['processed_symptoms'] = df['symptoms'].apply(self.preprocess_text)
        
        # Combine text features
        df['combined_text'] = df['processed_history'] + ' ' + df['processed_symptoms']
        
        # Extract text features using TF-IDF
        text_features = self.vectorizer.fit_transform(df['combined_text'])
        
        # Extract numerical features
        numerical_features = df[['age', 'smoking_history', 'family_history']].values
        
        # Encode gender
        gender_encoded = self.label_encoder.fit_transform(df['gender']).reshape(-1, 1)
        
        # Scale numerical features
        numerical_features = self.scaler.fit_transform(
            np.column_stack([numerical_features, gender_encoded])
        )
        
        # Combine all features
        from scipy.sparse import hstack
        features = hstack([text_features, numerical_features])
        
        target = df['cancer_diagnosis'].values
        
        return features, target
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train multiple models and select the best one.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dict: Training results and metrics
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_proba_val = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            val_accuracy = accuracy_score(y_val, y_pred_val)
            val_auc = roc_auc_score(y_val, y_pred_proba_val)
            
            results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'val_auc': val_auc,
                'predictions': y_pred_val,
                'probabilities': y_pred_proba_val
            }
            
            print(f"{name}: Train Acc: {train_accuracy:.3f}, Val Acc: {val_accuracy:.3f}, AUC: {val_auc:.3f}")
        
        # Select best model based on validation AUC
        best_model_name = max(results.keys(), key=lambda x: results[x]['val_auc'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        
        return results
    
    def predict_cancer_probability(self, patient_data: Dict) -> Dict:
        """
        Predict cancer probability for a new patient.
        
        Args:
            patient_data (Dict): Patient information
            
        Returns:
            Dict: Prediction results
        """
        if self.best_model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Create DataFrame from patient data
        df = pd.DataFrame([patient_data])
        
        # Preprocess
        df['processed_history'] = df['medical_history'].apply(self.preprocess_text)
        df['processed_symptoms'] = df['symptoms'].apply(self.preprocess_text)
        df['combined_text'] = df['processed_history'] + ' ' + df['processed_symptoms']
        
        # Extract features
        text_features = self.vectorizer.transform(df['combined_text'])
        numerical_features = df[['age', 'smoking_history', 'family_history']].values
        gender_encoded = self.label_encoder.transform(df['gender']).reshape(-1, 1)
        numerical_features = self.scaler.transform(
            np.column_stack([numerical_features, gender_encoded])
        )
        
        from scipy.sparse import hstack
        features = hstack([text_features, numerical_features])
        
        # Make prediction
        probability = self.best_model.predict_proba(features)[0, 1]
        prediction = self.best_model.predict(features)[0]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        return {
            'cancer_probability': probability,
            'predicted_diagnosis': bool(prediction),
            'risk_level': risk_level,
            'confidence': max(probability, 1 - probability)
        }
    
    def plot_results(self, results: Dict, y_val: np.ndarray):
        """
        Plot model performance results.
        
        Args:
            results (Dict): Training results
            y_val (np.ndarray): Validation labels
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        models = list(results.keys())
        accuracies = [results[model]['val_accuracy'] for model in models]
        aucs = [results[model]['val_auc'] for model in models]
        
        axes[0, 0].bar(models, accuracies, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Model Validation Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(models, aucs, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Model AUC Score')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Confusion matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['val_auc'])
        y_pred = results[best_model_name]['predictions']
        
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Feature importance (for tree-based models)
        if hasattr(self.best_model, 'feature_importances_'):
            # Get top features
            feature_importance = self.best_model.feature_importances_
            # For TF-IDF features, get top words
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top 10 most important features
            top_indices = np.argsort(feature_importance)[-10:]
            top_features = [feature_names[i] if i < len(feature_names) 
                          else f'numerical_feature_{i-len(feature_names)}' 
                          for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            axes[1, 1].barh(range(len(top_features)), top_importance)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features)
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('cancer_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessors."""
        model_data = {
            'best_model': self.best_model,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and preprocessors."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.vectorizer = model_data['vectorizer']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        
        print(f"Model loaded from {filepath}")

def main():
    """
    Main function to demonstrate the cancer diagnosis model.
    """
    print("Cancer Diagnosis NLP Model")
    print("=" * 50)
    
    # Initialize model
    model = CancerDiagnosisNLPModel()
    
    # Generate synthetic data
    print("Generating synthetic medical data...")
    df = model.create_synthetic_data(n_samples=2000)
    print(f"Generated {len(df)} patient records")
    
    # Extract features
    print("Extracting features...")
    X, y = model.extract_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    print("\nTraining models...")
    results = model.train_models(X_train, y_train, X_val, y_val)
    
    # Plot results
    print("\nGenerating visualizations...")
    model.plot_results(results, y_val)
    
    # Test predictions on test set
    print("\nEvaluating on test set...")
    y_test_pred = model.best_model.predict(X_test)
    y_test_proba = model.best_model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test AUC: {test_auc:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Example predictions
    print("\nExample Predictions:")
    print("-" * 30)
    
    # High-risk patient
    high_risk_patient = {
        'age': 65,
        'gender': 'male',
        'smoking_history': 1,
        'family_history': 1,
        'medical_history': 'Patient reports persistent cough, unexplained weight loss, and night sweats',
        'symptoms': 'persistent cough, weight loss, night sweats, fatigue'
    }
    
    result = model.predict_cancer_probability(high_risk_patient)
    print(f"High-risk patient: {result}")
    
    # Low-risk patient
    low_risk_patient = {
        'age': 30,
        'gender': 'female',
        'smoking_history': 0,
        'family_history': 0,
        'medical_history': 'Patient reports mild headache and seasonal allergies',
        'symptoms': 'headache, allergies'
    }
    
    result = model.predict_cancer_probability(low_risk_patient)
    print(f"Low-risk patient: {result}")
    
    # Save model
    print("\nSaving model...")
    model.save_model('cancer_diagnosis_model.pkl')
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main()