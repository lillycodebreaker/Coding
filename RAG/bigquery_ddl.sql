-- BigQuery Database Schema for RAG Document Processing System
-- Create dataset
CREATE SCHEMA IF NOT EXISTS `your-project-id.rag_document_processing`
OPTIONS (
  description = "RAG Document Processing System Data",
  location = "US"
);

-- Processing Results Table
CREATE OR REPLACE TABLE `your-project-id.rag_document_processing.processing_results` (
  id STRING NOT NULL,
  brd_filename STRING NOT NULL,
  sp_filename STRING NOT NULL,
  processing_timestamp TIMESTAMP NOT NULL,
  status STRING NOT NULL,
  error_message STRING,
  document_id STRING NOT NULL,
  extraction_timestamp TIMESTAMP NOT NULL,
  total_business_rules INT64,
  total_data_validations INT64,
  total_processing_rules INT64,
  metadata JSON
)
PARTITION BY DATE(processing_timestamp)
CLUSTER BY status, document_id
OPTIONS (
  description = "Main processing results and summary information"
);

-- Business Rules Table
CREATE OR REPLACE TABLE `your-project-id.rag_document_processing.business_rules` (
  id STRING NOT NULL,
  processing_id STRING NOT NULL,
  document_id STRING NOT NULL,
  type STRING NOT NULL,
  title STRING NOT NULL,
  description STRING NOT NULL,
  source STRING NOT NULL,
  priority STRING NOT NULL,
  conditions ARRAY<STRING>,
  actions ARRAY<STRING>,
  validation_criteria STRING,
  confidence_score FLOAT64 NOT NULL,
  created_timestamp TIMESTAMP NOT NULL
)
PARTITION BY DATE(created_timestamp)
CLUSTER BY source, priority, confidence_score
OPTIONS (
  description = "Extracted business rules from BRD and stored procedures"
);

-- Data Validations Table
CREATE OR REPLACE TABLE `your-project-id.rag_document_processing.data_validations` (
  id STRING NOT NULL,
  processing_id STRING NOT NULL,
  document_id STRING NOT NULL,
  field_name STRING NOT NULL,
  validation_type STRING NOT NULL,
  validation_rule STRING NOT NULL,
  error_message STRING NOT NULL,
  source STRING NOT NULL,
  confidence_score FLOAT64 NOT NULL,
  created_timestamp TIMESTAMP NOT NULL
)
PARTITION BY DATE(created_timestamp)
CLUSTER BY validation_type, source, confidence_score
OPTIONS (
  description = "Data validation rules extracted from documents"
);

-- Processing Rules Table
CREATE OR REPLACE TABLE `your-project-id.rag_document_processing.processing_rules` (
  id STRING NOT NULL,
  processing_id STRING NOT NULL,
  document_id STRING NOT NULL,
  step_order INT64 NOT NULL,
  condition STRING NOT NULL,
  action STRING NOT NULL,
  input_fields ARRAY<STRING>,
  output_fields ARRAY<STRING>,
  source STRING NOT NULL,
  confidence_score FLOAT64 NOT NULL,
  created_timestamp TIMESTAMP NOT NULL
)
PARTITION BY DATE(created_timestamp)
CLUSTER BY source, step_order, confidence_score
OPTIONS (
  description = "Processing rules and logic extracted from stored procedures"
);

-- Extraction Metadata Table
CREATE OR REPLACE TABLE `your-project-id.rag_document_processing.extraction_metadata` (
  document_id STRING NOT NULL,
  processing_id STRING NOT NULL,
  extraction_method STRING NOT NULL,
  brd_length INT64,
  sp_length INT64,
  processing_time STRING,
  confidence_threshold FLOAT64,
  additional_metadata JSON,
  created_timestamp TIMESTAMP NOT NULL
)
PARTITION BY DATE(created_timestamp)
CLUSTER BY extraction_method, document_id
OPTIONS (
  description = "Metadata about the extraction process and document characteristics"
);

-- Views for easier querying

-- Rules Summary View
CREATE OR REPLACE VIEW `your-project-id.rag_document_processing.rules_summary` AS
SELECT 
  pr.id as processing_id,
  pr.brd_filename,
  pr.sp_filename,
  pr.processing_timestamp,
  pr.status,
  pr.document_id,
  pr.total_business_rules,
  pr.total_data_validations,
  pr.total_processing_rules,
  COUNT(DISTINCT br.id) as actual_business_rules,
  COUNT(DISTINCT dv.id) as actual_data_validations,
  COUNT(DISTINCT pr2.id) as actual_processing_rules,
  AVG(br.confidence_score) as avg_business_rule_confidence,
  AVG(dv.confidence_score) as avg_validation_confidence,
  AVG(pr2.confidence_score) as avg_processing_rule_confidence
FROM `your-project-id.rag_document_processing.processing_results` pr
LEFT JOIN `your-project-id.rag_document_processing.business_rules` br ON pr.id = br.processing_id
LEFT JOIN `your-project-id.rag_document_processing.data_validations` dv ON pr.id = dv.processing_id
LEFT JOIN `your-project-id.rag_document_processing.processing_rules` pr2 ON pr.id = pr2.processing_id
GROUP BY 1,2,3,4,5,6,7,8,9;

-- High Confidence Rules View
CREATE OR REPLACE VIEW `your-project-id.rag_document_processing.high_confidence_rules` AS
SELECT 
  'business_rule' as rule_type,
  id,
  processing_id,
  document_id,
  title,
  description,
  source,
  confidence_score,
  created_timestamp
FROM `your-project-id.rag_document_processing.business_rules`
WHERE confidence_score >= 0.8

UNION ALL

SELECT 
  'data_validation' as rule_type,
  id,
  processing_id,
  document_id,
  CONCAT('Validation: ', field_name) as title,
  validation_rule as description,
  source,
  confidence_score,
  created_timestamp
FROM `your-project-id.rag_document_processing.data_validations`
WHERE confidence_score >= 0.8

UNION ALL

SELECT 
  'processing_rule' as rule_type,
  id,
  processing_id,
  document_id,
  CONCAT('Step ', step_order, ': ', LEFT(action, 50)) as title,
  CONCAT('Condition: ', condition, ' | Action: ', action) as description,
  source,
  confidence_score,
  created_timestamp
FROM `your-project-id.rag_document_processing.processing_rules`
WHERE confidence_score >= 0.8;

-- Document Processing Statistics View
CREATE OR REPLACE VIEW `your-project-id.rag_document_processing.processing_stats` AS
SELECT 
  DATE(processing_timestamp) as processing_date,
  COUNT(*) as total_documents_processed,
  COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_extractions,
  COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_extractions,
  ROUND(COUNT(CASE WHEN status = 'completed' THEN 1 END) * 100.0 / COUNT(*), 2) as success_rate,
  SUM(total_business_rules) as total_business_rules_extracted,
  SUM(total_data_validations) as total_validations_extracted,
  SUM(total_processing_rules) as total_processing_rules_extracted,
  AVG(total_business_rules) as avg_business_rules_per_doc,
  AVG(total_data_validations) as avg_validations_per_doc,
  AVG(total_processing_rules) as avg_processing_rules_per_doc
FROM `your-project-id.rag_document_processing.processing_results`
GROUP BY processing_date
ORDER BY processing_date DESC;

-- Rules by Source Analysis View
CREATE OR REPLACE VIEW `your-project-id.rag_document_processing.rules_by_source` AS
SELECT 
  source,
  'business_rule' as rule_type,
  COUNT(*) as rule_count,
  AVG(confidence_score) as avg_confidence,
  MIN(confidence_score) as min_confidence,
  MAX(confidence_score) as max_confidence,
  COUNT(CASE WHEN confidence_score >= 0.9 THEN 1 END) as high_confidence_count,
  COUNT(CASE WHEN confidence_score >= 0.8 THEN 1 END) as medium_high_confidence_count
FROM `your-project-id.rag_document_processing.business_rules`
GROUP BY source

UNION ALL

SELECT 
  source,
  'data_validation' as rule_type,
  COUNT(*) as rule_count,
  AVG(confidence_score) as avg_confidence,
  MIN(confidence_score) as min_confidence,
  MAX(confidence_score) as max_confidence,
  COUNT(CASE WHEN confidence_score >= 0.9 THEN 1 END) as high_confidence_count,
  COUNT(CASE WHEN confidence_score >= 0.8 THEN 1 END) as medium_high_confidence_count
FROM `your-project-id.rag_document_processing.data_validations`
GROUP BY source

UNION ALL

SELECT 
  source,
  'processing_rule' as rule_type,
  COUNT(*) as rule_count,
  AVG(confidence_score) as avg_confidence,
  MIN(confidence_score) as min_confidence,
  MAX(confidence_score) as max_confidence,
  COUNT(CASE WHEN confidence_score >= 0.9 THEN 1 END) as high_confidence_count,
  COUNT(CASE WHEN confidence_score >= 0.8 THEN 1 END) as medium_high_confidence_count
FROM `your-project-id.rag_document_processing.processing_rules`
GROUP BY source;

-- Create indexes for better query performance
-- Note: BigQuery automatically optimizes queries, but clustering helps with performance

-- Sample queries for testing and analytics

-- Query 1: Get all rules for a specific document
/*
SELECT 
  br.title,
  br.description,
  br.confidence_score,
  'business_rule' as type
FROM `your-project-id.rag_document_processing.business_rules` br
WHERE br.document_id = 'your-document-id'

UNION ALL

SELECT 
  CONCAT('Validation: ', dv.field_name) as title,
  dv.validation_rule as description,
  dv.confidence_score,
  'data_validation' as type
FROM `your-project-id.rag_document_processing.data_validations` dv
WHERE dv.document_id = 'your-document-id'

ORDER BY confidence_score DESC;
*/

-- Query 2: Find similar rules across documents
/*
SELECT 
  br1.title,
  br1.description,
  br1.document_id as doc1,
  br2.document_id as doc2,
  br1.confidence_score
FROM `your-project-id.rag_document_processing.business_rules` br1
JOIN `your-project-id.rag_document_processing.business_rules` br2
  ON LOWER(br1.title) = LOWER(br2.title)
  AND br1.document_id != br2.document_id
WHERE br1.confidence_score >= 0.8
ORDER BY br1.confidence_score DESC;
*/

-- Query 3: Performance analytics
/*
SELECT 
  extraction_method,
  COUNT(*) as total_extractions,
  AVG(CAST(JSON_EXTRACT_SCALAR(additional_metadata, '$.brd_length') AS INT64)) as avg_brd_length,
  AVG(CAST(JSON_EXTRACT_SCALAR(additional_metadata, '$.sp_length') AS INT64)) as avg_sp_length,
  confidence_threshold
FROM `your-project-id.rag_document_processing.extraction_metadata`
GROUP BY extraction_method, confidence_threshold
ORDER BY total_extractions DESC;
*/