#!/usr/bin/env python3
"""
Example of loading and using the saved Cancer Diagnosis NLP Model
"""

from cancer_diagnosis_model import CancerDiagnosisNLPModel

def load_model_example():
    """Demonstrate loading and using a saved model."""
    print("🔬 Loading Saved Cancer Diagnosis Model")
    print("=" * 50)
    
    # Load the saved model
    print("Loading pre-trained model...")
    model = CancerDiagnosisNLPModel()
    
    try:
        model.load_model('cancer_diagnosis_model.pkl')
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ No saved model found. Please run quick_test.py first to train and save a model.")
        return
    
    # Example patients for prediction
    patients = [
        {
            'name': 'Patient A - High Risk',
            'data': {
                'age': 72,
                'gender': 'male',
                'smoking_history': 1,
                'family_history': 1,
                'medical_history': 'Patient has a 40-year history of smoking and reports persistent cough with blood, unexplained weight loss of 20 pounds over 2 months, severe fatigue, and night sweats. Family history includes lung cancer in father.',
                'symptoms': 'persistent cough with blood, unexplained weight loss, severe fatigue, night sweats, difficulty breathing'
            }
        },
        {
            'name': 'Patient B - Moderate Risk',
            'data': {
                'age': 58,
                'gender': 'female',
                'smoking_history': 0,
                'family_history': 1,
                'medical_history': 'Patient reports changes in bowel habits, occasional abdominal pain, and mild fatigue. Mother had breast cancer at age 62. No smoking history but occasional alcohol consumption.',
                'symptoms': 'changes in bowel habits, abdominal pain, mild fatigue'
            }
        },
        {
            'name': 'Patient C - Low Risk',
            'data': {
                'age': 25,
                'gender': 'female',
                'smoking_history': 0,
                'family_history': 0,
                'medical_history': 'Young patient reports occasional stress-related headaches, seasonal allergies in spring, and mild insomnia due to work stress. Regular exercise routine and healthy diet. No significant family medical history.',
                'symptoms': 'stress headaches, seasonal allergies, mild insomnia'
            }
        },
        {
            'name': 'Patient D - Concerning Symptoms',
            'data': {
                'age': 48,
                'gender': 'female',
                'smoking_history': 1,
                'family_history': 0,
                'medical_history': 'Patient found a lump in left breast during self-examination. Reports unusual bleeding and persistent headaches. History of smoking for 15 years, quit 5 years ago.',
                'symptoms': 'breast lump, unusual bleeding, persistent headaches'
            }
        }
    ]
    
    print(f"\n🎯 Analyzing {len(patients)} patient cases:")
    print("-" * 50)
    
    for i, patient in enumerate(patients, 1):
        print(f"\n{i}. {patient['name']}")
        print(f"   Age: {patient['data']['age']}, Gender: {patient['data']['gender']}")
        print(f"   Smoking: {'Yes' if patient['data']['smoking_history'] else 'No'}")
        print(f"   Family History: {'Yes' if patient['data']['family_history'] else 'No'}")
        print(f"   Symptoms: {patient['data']['symptoms']}")
        
        # Make prediction
        try:
            result = model.predict_cancer_probability(patient['data'])
            
            print(f"   🎯 PREDICTION RESULTS:")
            print(f"      Cancer Probability: {result['cancer_probability']:.3f} ({result['cancer_probability']*100:.1f}%)")
            print(f"      Risk Level: {result['risk_level']}")
            print(f"      Predicted Diagnosis: {'⚠️ POSITIVE' if result['predicted_diagnosis'] else '✅ NEGATIVE'}")
            print(f"      Model Confidence: {result['confidence']:.3f}")
            
            # Clinical recommendations
            if result['risk_level'] == 'High':
                print(f"      💡 RECOMMENDATION: 🚨 Immediate medical consultation required")
                print(f"      📞 Suggested action: Schedule oncology referral within 24-48 hours")
            elif result['risk_level'] == 'Moderate':
                print(f"      💡 RECOMMENDATION: ⚠️ Medical examination within 1-2 weeks")
                print(f"      📞 Suggested action: Schedule appointment with primary care physician")
            else:
                print(f"      💡 RECOMMENDATION: ✅ Continue routine monitoring")
                print(f"      📞 Suggested action: Regular health checkups as scheduled")
                
        except Exception as e:
            print(f"   ❌ Error making prediction: {e}")
    
    print(f"\n" + "=" * 50)
    print("📊 SUMMARY STATISTICS")
    print("=" * 50)
    
    # Calculate summary statistics
    high_risk_count = 0
    moderate_risk_count = 0
    low_risk_count = 0
    
    for patient in patients:
        try:
            result = model.predict_cancer_probability(patient['data'])
            if result['risk_level'] == 'High':
                high_risk_count += 1
            elif result['risk_level'] == 'Moderate':
                moderate_risk_count += 1
            else:
                low_risk_count += 1
        except:
            pass
    
    print(f"Total patients analyzed: {len(patients)}")
    print(f"High risk patients: {high_risk_count}")
    print(f"Moderate risk patients: {moderate_risk_count}")
    print(f"Low risk patients: {low_risk_count}")
    
    print(f"\n⚠️  IMPORTANT DISCLAIMERS:")
    print("   • This is a demonstration model for educational purposes only")
    print("   • Results should NOT be used for actual medical diagnosis")
    print("   • Always consult qualified healthcare professionals")
    print("   • Model trained on synthetic data, not real patient records")
    print("   • Clinical validation required before any medical use")
    
    print(f"\n🔍 ABOUT THE MODEL:")
    print(f"   • Model Type: {type(model.best_model).__name__}")
    print(f"   • Features: NLP text analysis + demographics + risk factors")
    print(f"   • Training: Multiple algorithms with automatic best model selection")
    print(f"   • Validation: Cross-validation with holdout test sets")

if __name__ == "__main__":
    load_model_example()