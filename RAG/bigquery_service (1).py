# services/bigquery_service.py
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from models.schemas import ProcessingResult, ExtractedRules

logger = logging.getLogger(__name__)

class BigQueryService:
    """Service for interacting with Google BigQuery database"""
    
    def __init__(self):
        # Initialize BigQuery client
        # Set your GCP project ID here or via environment variable
        self.project_id = os.getenv('GCP_PROJECT_ID', 'your-project-id')
        self.dataset_id = os.getenv('BQ_DATASET_ID', 'rag_document_processing')
        
        try:
            self.client = bigquery.Client(project=self.project_id)
            logger.info(f"BigQuery client initialized for project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise
    
    async def initialize_tables(self):
        """Create BigQuery dataset and tables if they don't exist"""
        try:
            # Create dataset
            await self._create_dataset_if_not_exists()
            
            # Create tables
            await self._create_processing_results_table()
            await self._create_business_rules_table()
            await self._create_data_validations_table()
            await self._create_processing_rules_table()
            await self._create_extraction_metadata_table()
            
            logger.info("BigQuery tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery tables: {e}")
            raise
    
    async def _create_dataset_if_not_exists(self):
        """Create dataset if it doesn't exist"""
        dataset_ref = self.client.dataset(self.dataset_id)
        
        try:
            self.client.get_dataset(dataset_ref)
            logger.info(f"Dataset {self.dataset_id} already exists")
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"  # Set your preferred location
            dataset.description = "RAG Document Processing System Data"
            
            self.client.create_dataset(dataset)
            logger.info(f"Created dataset {self.dataset_id}")
    
    async def _create_processing_results_table(self):
        """Create processing results table"""
        table_id = f"{self.project_id}.{self.dataset_id}.processing_results"
        
        schema = [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("brd_filename", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("sp_filename", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("processing_timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("error_message", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("document_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("extraction_timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("total_business_rules", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("total_data_validations", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("total_processing_rules", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
        ]
        
        await self._create_table_if_not_exists(table_id, schema, "Processing Results")
    
    async def _create_business_rules_table(self):
        """Create business rules table"""
        table_id = f"{self.project_id}.{self.dataset_id}.business_rules"
        
        schema = [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("processing_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("document_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("description", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("priority", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("conditions", "STRING", mode="REPEATED"),
            bigquery.SchemaField("actions", "STRING", mode="REPEATED"),
            bigquery.SchemaField("validation_criteria", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("confidence_score", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("created_timestamp", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        await self._create_table_if_not_exists(table_id, schema, "Business Rules")
    
    async def _create_data_validations_table(self):
        """Create data validations table"""
        table_id = f"{self.project_id}.{self.dataset_id}.data_validations"
        
        schema = [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("processing_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("document_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("field_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("validation_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("validation_rule", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("error_message", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("confidence_score", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("created_timestamp", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        await self._create_table_if_not_exists(table_id, schema, "Data Validations")
    
    async def _create_processing_rules_table(self):
        """Create processing rules table"""
        table_id = f"{self.project_id}.{self.dataset_id}.processing_rules"
        
        schema = [
            bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("processing_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("document_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("step_order", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("condition", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("action", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("input_fields", "STRING", mode="REPEATED"),
            bigquery.SchemaField("output_fields", "STRING", mode="REPEATED"),
            bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("confidence_score", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("created_timestamp", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        await self._create_table_if_not_exists(table_id, schema, "Processing Rules")
    
    async def _create_extraction_metadata_table(self):
        """Create extraction metadata table"""
        table_id = f"{self.project_id}.{self.dataset_id}.extraction_metadata"
        
        schema = [
            bigquery.SchemaField("document_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("processing_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("extraction_method", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("brd_length", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("sp_length", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("processing_time", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("confidence_threshold", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("additional_metadata", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("created_timestamp", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        await self._create_table_if_not_exists(table_id, schema, "Extraction Metadata")
    
    async def _create_table_if_not_exists(self, table_id: str, schema: List[bigquery.SchemaField], description: str):
        """Create table if it doesn't exist"""
        try:
            self.client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except NotFound:
            table = bigquery.Table(table_id, schema=schema)
            table.description = description
            
            self.client.create_table(table)
            logger.info(f"Created table {table_id}")
    
    async def store_processing_result(self, processing_result: ProcessingResult):
        """Store processing result and extracted rules in BigQuery"""
        try:
            # Store main processing result
            await self._insert_processing_result(processing_result)
            
            # Store extracted rules in separate tables
            await self._insert_business_rules(processing_result)
            await self._insert_data_validations(processing_result)
            await self._insert_processing_rules(processing_result)
            await self._insert_extraction_metadata(processing_result)
            
            logger.info(f"Successfully stored processing result {processing_result.id}")
            
        except Exception as e:
            logger.error(f"Failed to store processing result: {e}")
            raise
    
    async def _insert_processing_result(self, processing_result: ProcessingResult):
        """Insert processing result into main table"""
        table_id = f"{self.project_id}.{self.dataset_id}.processing_results"
        
        rows_to_insert = [{
            "id": processing_result.id,
            "brd_filename": processing_result.brd_filename,
            "sp_filename": processing_result.sp_filename,
            "processing_timestamp": processing_result.processing_timestamp,
            "status": processing_result.status,
            "error_message": processing_result.error_message,
            "document_id": processing_result.extracted_rules.document_id,
            "extraction_timestamp": processing_result.extracted_rules.extraction_timestamp,
            "total_business_rules": len(processing_result.extracted_rules.business_rules),
            "total_data_validations": len(processing_result.extracted_rules.data_validations),
            "total_processing_rules": len(processing_result.extracted_rules.processing_rules),
            "metadata": processing_result.extracted_rules.metadata,
        }]
        
        table = self.client.get_table(table_id)
        errors = self.client.insert_rows_json(table, rows_to_insert)
        
        if errors:
            raise Exception(f"Failed to insert processing result: {errors}")
    
    async def _insert_business_rules(self, processing_result: ProcessingResult):
        """Insert business rules into business_rules table"""
        if not processing_result.extracted_rules.business_rules:
            return
        
        table_id = f"{self.project_id}.{self.dataset_id}.business_rules"
        
        rows_to_insert = []
        for rule in processing_result.extracted_rules.business_rules:
            rows_to_insert.append({
                "id": rule.id,
                "processing_id": processing_result.id,
                "document_id": processing_result.extracted_rules.document_id,
                "type": rule.type,
                "title": rule.title,
                "description": rule.description,
                "source": rule.source,
                "priority": rule.priority,
                "conditions": rule.conditions,
                "actions": rule.actions,
                "validation_criteria": rule.validation_criteria,
                "confidence_score": rule.confidence_score,
                "created_timestamp": datetime.now(),
            })
        
        table = self.client.get_table(table_id)
        errors = self.client.insert_rows_json(table, rows_to_insert)
        
        if errors:
            raise Exception(f"Failed to insert business rules: {errors}")
    
    async def _insert_data_validations(self, processing_result: ProcessingResult):
        """Insert data validations into data_validations table"""
        if not processing_result.extracted_rules.data_validations:
            return
        
        table_id = f"{self.project_id}.{self.dataset_id}.data_validations"
        
        rows_to_insert = []
        for validation in processing_result.extracted_rules.data_validations:
            rows_to_insert.append({
                "id": validation.id,
                "processing_id": processing_result.id,
                "document_id": processing_result.extracted_rules.document_id,
                "field_name": validation.field_name,
                "validation_type": validation.validation_type,
                "validation_rule": validation.validation_rule,
                "error_message": validation.error_message,
                "source": validation.source,
                "confidence_score": validation.confidence_score,
                "created_timestamp": datetime.now(),
            })
        
        table = self.client.get_table(table_id)
        errors = self.client.insert_rows_json(table, rows_to_insert)
        
        if errors:
            raise Exception(f"Failed to insert data validations: {errors}")
    
    async def _insert_processing_rules(self, processing_result: ProcessingResult):
        """Insert processing rules into processing_rules table"""
        if not processing_result.extracted_rules.processing_rules:
            return
        
        table_id = f"{self.project_id}.{self.dataset_id}.processing_rules"
        
        rows_to_insert = []
        for rule in processing_result.extracted_rules.processing_rules:
            rows_to_insert.append({
                "id": rule.id,
                "processing_id": processing_result.id,
                "document_id": processing_result.extracted_rules.document_id,
                "step_order": rule.step_order,
                "condition": rule.condition,
                "action": rule.action,
                "input_fields": rule.input_fields,
                "output_fields": rule.output_fields,
                "source": rule.source,
                "confidence_score": rule.confidence_score,
                "created_timestamp": datetime.now(),
            })
        
        table = self.client.get_table(table_id)
        errors = self.client.insert_rows_json(table, rows_to_insert)
        
        if errors:
            raise Exception(f"Failed to insert processing rules: {errors}")
    
    async def _insert_extraction_metadata(self, processing_result: ProcessingResult):
        """Insert extraction metadata"""
        table_id = f"{self.project_id}.{self.dataset_id}.extraction_metadata"
        
        metadata = processing_result.extracted_rules.metadata
        rows_to_insert = [{
            "document_id": processing_result.extracted_rules.document_id,
            "processing_id": processing_result.id,
            "extraction_method": metadata.get("extraction_method", ""),
            "brd_length": metadata.get("brd_length"),
            "sp_length": metadata.get("sp_length"),
            "processing_time": metadata.get("processing_time"),
            "confidence_threshold": metadata.get("confidence_threshold"),
            "additional_metadata": metadata,
            "created_timestamp": datetime.now(),
        }]
        
        table = self.client.get_table(table_id)
        errors = self.client.insert_rows_json(table, rows_to_insert)
        
        if errors:
            raise Exception(f"Failed to insert extraction metadata: {errors}")
    
    async def get_processing_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get processing history from BigQuery"""
        query = f"""
        SELECT 
            id,
            brd_filename,
            sp_filename,
            processing_timestamp,
            status,
            document_id,
            total_business_rules,
            total_data_validations,
            total_processing_rules,
            error_message
        FROM `{self.project_id}.{self.dataset_id}.processing_results`
        ORDER BY processing_timestamp DESC
        LIMIT {limit}
        """
        
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            
            history = []
            for row in results:
                history.append({
                    "id": row.id,
                    "brd_filename": row.brd_filename,
                    "sp_filename": row.sp_filename,
                    "processing_timestamp": row.processing_timestamp.isoformat() if row.processing_timestamp else None,
                    "status": row.status,
                    "document_id": row.document_id,
                    "total_business_rules": row.total_business_rules,
                    "total_data_validations": row.total_data_validations,
                    "total_processing_rules": row.total_processing_rules,
                    "error_message": row.error_message
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to fetch processing history: {e}")
            raise
    
    async def get_business_rules_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Get business rules for a specific document"""
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.business_rules`
        WHERE document_id = @document_id
        ORDER BY confidence_score DESC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("document_id", "STRING", document_id)
            ]
        )
        
        try:
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            rules = []
            for row in results:
                rules.append({
                    "id": row.id,
                    "type": row.type,
                    "title": row.title,
                    "description": row.description,
                    "source": row.source,
                    "priority": row.priority,
                    "conditions": row.conditions,
                    "actions": row.actions,
                    "confidence_score": row.confidence_score,
                    "created_timestamp": row.created_timestamp.isoformat() if row.created_timestamp else None
                })
            
            return rules
            
        except Exception as e:
            logger.error(f"Failed to fetch business rules: {e}")
            raise
    
    async def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics across all documents"""
        query = f"""
        SELECT 
            COUNT(*) as total_extractions,
            SUM(total_business_rules) as total_business_rules,
            SUM(total_data_validations) as total_data_validations,
            SUM(total_processing_rules) as total_processing_rules,
            AVG(total_business_rules) as avg_business_rules_per_doc,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_extractions,
            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_extractions
        FROM `{self.project_id}.{self.dataset_id}.processing_results`
        """
        
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            
            for row in results:
                return {
                    "total_extractions": row.total_extractions,
                    "total_business_rules": row.total_business_rules or 0,
                    "total_data_validations": row.total_data_validations or 0,
                    "total_processing_rules": row.total_processing_rules or 0,
                    "avg_business_rules_per_doc": float(row.avg_business_rules_per_doc) if row.avg_business_rules_per_doc else 0.0,
                    "successful_extractions": row.successful_extractions,
                    "failed_extractions": row.failed_extractions
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to fetch extraction statistics: {e}")
            raise
    
    async def search_rules(self, search_term: str, rule_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for rules containing specific terms"""
        base_query = f"""
        SELECT 
            br.id,
            br.title,
            br.description,
            br.source,
            br.priority,
            br.confidence_score,
            pr.brd_filename,
            pr.sp_filename,
            pr.processing_timestamp
        FROM `{self.project_id}.{self.dataset_id}.business_rules` br
        JOIN `{self.project_id}.{self.dataset_id}.processing_results` pr
        ON br.processing_id = pr.id
        WHERE (LOWER(br.title) LIKE @search_term 
               OR LOWER(br.description) LIKE @search_term)
        """
        
        if rule_type:
            base_query += " AND br.type = @rule_type"
        
        base_query += " ORDER BY br.confidence_score DESC LIMIT 100"
        
        query_parameters = [
            bigquery.ScalarQueryParameter("search_term", "STRING", f"%{search_term.lower()}%")
        ]
        
        if rule_type:
            query_parameters.append(
                bigquery.ScalarQueryParameter("rule_type", "STRING", rule_type)
            )
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
        
        try:
            query_job = self.client.query(base_query, job_config=job_config)
            results = query_job.result()
            
            search_results = []
            for row in results:
                search_results.append({
                    "id": row.id,
                    "title": row.title,
                    "description": row.description,
                    "source": row.source,
                    "priority": row.priority,
                    "confidence_score": row.confidence_score,
                    "brd_filename": row.brd_filename,
                    "sp_filename": row.sp_filename,
                    "processing_timestamp": row.processing_timestamp.isoformat() if row.processing_timestamp else None
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search rules: {e}")
            raise