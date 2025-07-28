#!/usr/bin/env python3
"""
Quick test of the Cancer Diagnosis NLP Model
"""

from cancer_diagnosis_model import CancerDiagnosisNLPModel
import numpy as np

def quick_test():
    """Run a quick test of the model."""
    print("🏥 Cancer Diagnosis NLP Model - Quick Test")
    print("=" * 50)
    
    # Initialize model
    print("1. Initializing model...")
    model = CancerDiagnosisNLPModel()
    print("   ✅ Model initialized")
    
    # Generate synthetic data (small dataset for quick test)
    print("\n2. Generating synthetic data...")
    df = model.create_synthetic_data(n_samples=200)
    print(f"   ✅ Generated {len(df)} patient records")
    
    # Show sample data
    print("\n3. Sample patient data:")
    sample = df.iloc[0]
    print(f"   Patient ID: {sample['patient_id']}")
    print(f"   Age: {sample['age']}, Gender: {sample['gender']}")
    print(f"   Medical History: {sample['medical_history'][:80]}...")
    print(f"   Cancer Diagnosis: {sample['cancer_diagnosis']}")
    
    # Extract features
    print("\n4. Extracting features...")
    X, y = model.extract_features(df)
    print(f"   ✅ Feature matrix shape: {X.shape}")
    print(f"   ✅ Target vector shape: {y.shape}")
    print(f"   ✅ Cancer cases: {np.sum(y)}/{len(y)} ({np.mean(y)*100:.1f}%)")
    
    # Train models
    print("\n5. Training models...")
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    results = model.train_models(X_train, y_train, X_val, y_val)
    print(f"   ✅ Best model: {type(model.best_model).__name__}")
    
    # Test predictions
    print("\n6. Testing predictions...")
    
    # High-risk patient
    high_risk_patient = {
        'age': 65,
        'gender': 'male',
        'smoking_history': 1,
        'family_history': 1,
        'medical_history': 'Patient reports persistent cough, unexplained weight loss, and night sweats over 3 months.',
        'symptoms': 'persistent cough, weight loss, night sweats, fatigue'
    }
    
    result_high = model.predict_cancer_probability(high_risk_patient)
    print(f"   High-risk patient:")
    print(f"   - Cancer Probability: {result_high['cancer_probability']:.3f} ({result_high['cancer_probability']*100:.1f}%)")
    print(f"   - Risk Level: {result_high['risk_level']}")
    print(f"   - Predicted: {result_high['predicted_diagnosis']}")
    
    # Low-risk patient
    low_risk_patient = {
        'age': 28,
        'gender': 'female',
        'smoking_history': 0,
        'family_history': 0,
        'medical_history': 'Patient reports mild headaches and seasonal allergies. No significant medical history.',
        'symptoms': 'headaches, allergies'
    }
    
    result_low = model.predict_cancer_probability(low_risk_patient)
    print(f"\n   Low-risk patient:")
    print(f"   - Cancer Probability: {result_low['cancer_probability']:.3f} ({result_low['cancer_probability']*100:.1f}%)")
    print(f"   - Risk Level: {result_low['risk_level']}")
    print(f"   - Predicted: {result_low['predicted_diagnosis']}")
    
    # Model evaluation on test set
    print("\n7. Model performance on test set:")
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    y_pred = model.best_model.predict(X_test)
    y_proba = model.best_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"   - Test Accuracy: {accuracy:.3f}")
    print(f"   - Test AUC: {auc:.3f}")
    
    # Save model
    print("\n8. Saving model...")
    model.save_model('cancer_diagnosis_model.pkl')
    print("   ✅ Model saved successfully")
    
    print("\n" + "=" * 50)
    print("🎉 Quick test completed successfully!")
    print("\nKey Results:")
    print(f"✅ Model trained on {len(df)} patient records")
    print(f"✅ Test Accuracy: {accuracy:.1%}")
    print(f"✅ Test AUC: {auc:.3f}")
    print(f"✅ High-risk patient probability: {result_high['cancer_probability']:.1%}")
    print(f"✅ Low-risk patient probability: {result_low['cancer_probability']:.1%}")
    
    print("\n⚠️  IMPORTANT: This is a demonstration model for educational purposes only.")
    print("   Always consult healthcare professionals for medical advice.")

if __name__ == "__main__":
    quick_test()