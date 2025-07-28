#!/usr/bin/env python3
"""
Demonstration script for Cancer Diagnosis NLP Model
"""

from cancer_diagnosis_model import CancerDiagnosisNLPModel
import pandas as pd

def demonstrate_model():
    """
    Demonstrate the cancer diagnosis model with various patient examples.
    """
    print("🏥 Cancer Diagnosis NLP Model - Quick Demo")
    print("=" * 60)
    
    # Initialize and train model with smaller dataset for demo
    model = CancerDiagnosisNLPModel()
    
    # Generate and train on synthetic data
    print("📊 Generating synthetic medical data...")
    df = model.create_synthetic_data(n_samples=500)  # Smaller for demo
    
    print("🔧 Training model...")
    X, y = model.extract_features(df)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train models (without plotting for demo)
    results = model.train_models(X_train, y_train, X_val, y_val)
    
    print("\n🎯 Testing with Various Patient Scenarios:")
    print("-" * 50)
    
    # Test cases with different risk levels
    test_patients = [
        {
            'name': 'High-Risk Patient (John, 68)',
            'data': {
                'age': 68,
                'gender': 'male',
                'smoking_history': 1,
                'family_history': 1,
                'medical_history': 'Patient reports persistent cough lasting 3 months, unexplained weight loss of 15 pounds, night sweats, and chronic fatigue. History of smoking for 30 years.',
                'symptoms': 'persistent cough, unexplained weight loss, night sweats, chronic fatigue, difficulty breathing'
            }
        },
        {
            'name': 'Moderate-Risk Patient (Sarah, 55)',
            'data': {
                'age': 55,
                'gender': 'female',
                'smoking_history': 0,
                'family_history': 1,
                'medical_history': 'Patient reports occasional fatigue, mild indigestion, and family history of breast cancer. No smoking history.',
                'symptoms': 'fatigue, indigestion, family history cancer'
            }
        },
        {
            'name': 'Low-Risk Patient (Emily, 28)',
            'data': {
                'age': 28,
                'gender': 'female',
                'smoking_history': 0,
                'family_history': 0,
                'medical_history': 'Patient reports seasonal allergies, occasional mild headaches, and stress from work. No significant medical history.',
                'symptoms': 'seasonal allergies, mild headache, stress'
            }
        },
        {
            'name': 'Elderly Low-Risk Patient (Robert, 72)',
            'data': {
                'age': 72,
                'gender': 'male',
                'smoking_history': 0,
                'family_history': 0,
                'medical_history': 'Patient reports joint stiffness, mild back pain, and occasional insomnia. Regular checkups show normal results.',
                'symptoms': 'joint stiffness, mild back pain, insomnia'
            }
        },
        {
            'name': 'Middle-Age with Concerning Symptoms (Lisa, 45)',
            'data': {
                'age': 45,
                'gender': 'female',
                'smoking_history': 1,
                'family_history': 0,
                'medical_history': 'Patient reports finding a small lump in breast, some unusual bleeding, and persistent headaches. History of smoking.',
                'symptoms': 'lump, unusual bleeding, persistent headaches'
            }
        }
    ]
    
    for patient in test_patients:
        print(f"\n👤 {patient['name']}:")
        try:
            result = model.predict_cancer_probability(patient['data'])
            
            print(f"   📋 Symptoms: {patient['data']['symptoms']}")
            print(f"   🎯 Cancer Probability: {result['cancer_probability']:.3f} ({result['cancer_probability']*100:.1f}%)")
            print(f"   ⚠️  Risk Level: {result['risk_level']}")
            print(f"   📊 Confidence: {result['confidence']:.3f}")
            print(f"   🔬 Predicted Diagnosis: {'⚠️ POSITIVE' if result['predicted_diagnosis'] else '✅ NEGATIVE'}")
            
            # Risk interpretation
            if result['risk_level'] == 'High':
                print("   💡 Recommendation: Immediate medical consultation required")
            elif result['risk_level'] == 'Moderate':
                print("   💡 Recommendation: Schedule medical examination within 2 weeks")
            else:
                print("   💡 Recommendation: Continue routine monitoring")
                
        except Exception as e:
            print(f"   ❌ Error processing patient: {e}")
    
    print(f"\n📈 Model Performance Summary:")
    print(f"   - Best Model: {type(model.best_model).__name__}")
    print(f"   - Training completed on {len(df)} patient records")
    print(f"   - Features: Text analysis + Demographics + Risk factors")
    
    print(f"\n⚠️  Important Disclaimer:")
    print("   This is a demonstration model for educational purposes only.")
    print("   It should NOT be used for actual medical diagnosis.")
    print("   Always consult qualified healthcare professionals for medical advice.")

def interactive_prediction():
    """
    Allow user to input custom patient data for prediction.
    """
    print("\n" + "="*60)
    print("🔬 Interactive Cancer Risk Assessment")
    print("="*60)
    
    # Load pre-trained model or train a new one
    try:
        model = CancerDiagnosisNLPModel()
        model.load_model('cancer_diagnosis_model.pkl')
        print("✅ Loaded pre-trained model")
    except:
        print("⚠️  No pre-trained model found. Training new model...")
        model = CancerDiagnosisNLPModel()
        df = model.create_synthetic_data(n_samples=500)
        X, y = model.extract_features(df)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        model.train_models(X_train, y_train, X_val, y_val)
        print("✅ Model trained successfully")
    
    print("\nEnter patient information (press Enter to use default values):")
    
    # Collect patient data
    try:
        age = input("Age (default: 45): ").strip()
        age = int(age) if age else 45
        
        gender = input("Gender (male/female, default: female): ").strip().lower()
        gender = gender if gender in ['male', 'female'] else 'female'
        
        smoking = input("Smoking history (1=yes, 0=no, default: 0): ").strip()
        smoking = int(smoking) if smoking in ['0', '1'] else 0
        
        family_history = input("Family history of cancer (1=yes, 0=no, default: 0): ").strip()
        family_history = int(family_history) if family_history in ['0', '1'] else 0
        
        symptoms = input("Symptoms (comma-separated, default: 'fatigue, headache'): ").strip()
        symptoms = symptoms if symptoms else 'fatigue, headache'
        
        medical_history = input("Medical history description (default: 'Patient reports general symptoms'): ").strip()
        medical_history = medical_history if medical_history else f'Patient reports {symptoms}.'
        
        # Create patient data
        patient_data = {
            'age': age,
            'gender': gender,
            'smoking_history': smoking,
            'family_history': family_history,
            'medical_history': medical_history,
            'symptoms': symptoms
        }
        
        # Make prediction
        result = model.predict_cancer_probability(patient_data)
        
        print(f"\n🎯 Risk Assessment Results:")
        print(f"   Patient: {age}-year-old {gender}")
        print(f"   Cancer Probability: {result['cancer_probability']:.3f} ({result['cancer_probability']*100:.1f}%)")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Predicted Diagnosis: {'POSITIVE' if result['predicted_diagnosis'] else 'NEGATIVE'}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Assessment cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during assessment: {e}")

if __name__ == "__main__":
    # Run demonstration
    demonstrate_model()
    
    # Offer interactive session
    while True:
        user_input = input("\n🤔 Would you like to try interactive prediction? (y/n): ").strip().lower()
        if user_input in ['y', 'yes']:
            interactive_prediction()
            break
        elif user_input in ['n', 'no']:
            print("👋 Thank you for using the Cancer Diagnosis NLP Model!")
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")