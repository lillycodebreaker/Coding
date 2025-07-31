# models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class RuleType(str, Enum):
    BUSINESS_RULE = "business_rule"
    DATA_VALIDATION = "data_validation"
    PROCESSING_RULE = "processing_rule"
    CONDITIONAL_LOGIC = "conditional_logic"

class BusinessRule(BaseModel):
    id: str = Field(..., description="Unique identifier for the rule")
    type: RuleType = Field(..., description="Type of rule")
    title: str = Field(..., description="Rule title or summary")
    description: str = Field(..., description="Detailed rule description")
    source: str = Field(..., description="Source document (BRD or SP)")
    priority: str = Field(default="medium", description="Rule priority")
    conditions: List[str] = Field(default=[], description="Rule conditions")
    actions: List[str] = Field(default=[], description="Rule actions")
    validation_criteria: Optional[str] = Field(None, description="Validation criteria")
    confidence_score: float = Field(default=0.0, description="Extraction confidence score")

class DataValidation(BaseModel):
    id: str = Field(..., description="Unique identifier for the validation")
    field_name: str = Field(..., description="Field being validated")
    validation_type: str = Field(..., description="Type of validation")
    validation_rule: str = Field(..., description="Validation rule")
    error_message: str = Field(..., description="Error message for validation failure")
    source: str = Field(..., description="Source document")
    confidence_score: float = Field(default=0.0, description="Extraction confidence score")

class ProcessingRule(BaseModel):
    id: str = Field(..., description="Unique identifier for the processing rule")
    step_order: int = Field(..., description="Order of processing step")
    condition: str = Field(..., description="Processing condition")
    action: str = Field(..., description="Processing action")
    input_fields: List[str] = Field(default=[], description="Input fields")
    output_fields: List[str] = Field(default=[], description="Output fields")
    source: str = Field(..., description="Source document")
    confidence_score: float = Field(default=0.0, description="Extraction confidence score")

class ExtractedRules(BaseModel):
    document_id: str = Field(..., description="Unique document processing ID")
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    business_rules: List[BusinessRule] = Field(default=[], description="Extracted business rules")
    data_validations: List[DataValidation] = Field(default=[], description="Data validation rules")
    processing_rules: List[ProcessingRule] = Field(default=[], description="Processing rules")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")
    summary: Dict[str, int] = Field(default={}, description="Extraction summary statistics")

class ProcessingResult(BaseModel):
    id: str = Field(..., description="Unique processing result ID")
    brd_filename: str = Field(..., description="BRD document filename")
    sp_filename: str = Field(..., description="Stored procedure filename")
    extracted_rules: ExtractedRules = Field(..., description="Extracted rules")
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="pending", description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")

# Example rule extraction templates
RULE_EXTRACTION_EXAMPLES = {
    "business_rules": [
        {
            "id": "BR001",
            "type": "business_rule",
            "title": "Customer Age Validation",
            "description": "Customer must be at least 18 years old to open an account",
            "source": "BRD",
            "priority": "high",
            "conditions": ["customer_age >= 18"],
            "actions": ["allow_account_creation"],
            "confidence_score": 0.95
        }
    ],
    "data_validations": [
        {
            "id": "DV001",
            "field_name": "email_address",
            "validation_type": "format",
            "validation_rule": "must contain @ symbol and valid domain",
            "error_message": "Invalid email format",
            "source": "SP",
            "confidence_score": 0.90
        }
    ],
    "processing_rules": [
        {
            "id": "PR001",
            "step_order": 1,
            "condition": "account_type = 'premium'",
            "action": "apply_premium_benefits",
            "input_fields": ["account_type", "customer_id"],
            "output_fields": ["benefit_status", "discount_rate"],
            "source": "SP",
            "confidence_score": 0.88
        }
    ]
}