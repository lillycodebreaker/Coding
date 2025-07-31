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
            await self._insert_extraction_metadata(