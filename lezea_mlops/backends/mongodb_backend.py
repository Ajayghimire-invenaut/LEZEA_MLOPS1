"""
MongoDB Backend for LeZeA MLOps
==============================

Handles storage of complex hierarchical data including:
- Modification trees and paths
- Resource usage statistics (per-layer, per-algorithm)
- Business metrics and costs
- Experiment metadata and summaries
- Complex JSON data that doesn't fit well in MLflow

This backend provides optimized storage and querying for LeZeA's
complex data structures with proper indexing and aggregation capabilities.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from bson import ObjectId

try:
    import pymongo
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    pymongo = None


class MongoBackend:
    """
    MongoDB backend for complex data storage and hierarchical queries
    
    This class provides:
    - Connection management with retry logic
    - Optimized collections for different data types
    - Automatic indexing for performance
    - Aggregation pipelines for analytics
    - Data validation and cleanup
    """
    
    def __init__(self, config):
        """
        Initialize MongoDB backend
        
        Args:
            config: Configuration object with MongoDB settings
        """
        if not MONGODB_AVAILABLE:
            raise RuntimeError(
                "MongoDB is not available. Install with: pip install pymongo"
            )
        
        self.config = config
        self.mongo_config = config.get_mongodb_config()
        
        # Initialize connection
        self.client = None
        self.database = None
        self.collections = {}
        
        # Connect and setup
        self._connect()
        self._setup_collections()
        self._create_indexes()
        
        print(f"✅ MongoDB backend connected: {self.mongo_config['database']}")
    
    def _connect(self):
        """Establish connection to MongoDB with retry logic"""
        try:
            # Create MongoDB client with connection options
            self.client = MongoClient(
                self.mongo_config['connection_string'],
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=30000,
                maxPoolSize=100,
                retryWrites=True,
                w="majority"
            )
            
            # Test the connection
            self.client.admin.command('ping')
            
            # Get database
            self.database = self.client[self.mongo_config['database']]
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")
    
    def _setup_collections(self):
        """Setup collections with proper configuration"""
        collection_names = self.mongo_config['collections']
        
        for collection_type, collection_name in collection_names.items():
            self.collections[collection_type] = self.database[collection_name]
    
    def _create_indexes(self):
        """Create indexes for optimal query performance"""
        try:
            # Experiments collection indexes
            if 'experiments' in self.collections:
                exp_col = self.collections['experiments']
                exp_col.create_index([("experiment_id", ASCENDING)], unique=True)
                exp_col.create_index([("timestamp", DESCENDING)])
                exp_col.create_index([("experiment_name", ASCENDING)])
                exp_col.create_index([("status", ASCENDING)])
            
            # Modifications collection indexes
            if 'modifications' in self.collections:
                mod_col = self.collections['modifications']
                mod_col.create_index([("experiment_id", ASCENDING), ("step", ASCENDING)])
                mod_col.create_index([("timestamp", DESCENDING)])
                mod_col.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])
            
            # Resources collection indexes
            if 'resources' in self.collections:
                res_col = self.collections['resources']
                res_col.create_index([("experiment_id", ASCENDING)])
                res_col.create_index([("component_type", ASCENDING)])
                res_col.create_index([("experiment_id", ASCENDING), ("component_type", ASCENDING)])
                res_col.create_index([("timestamp", DESCENDING)])
            
            # Business collection indexes
            if 'business' in self.collections:
                bus_col = self.collections['business']
                bus_col.create_index([("experiment_id", ASCENDING)])
                bus_col.create_index([("cost", DESCENDING)])
                bus_col.create_index([("timestamp", DESCENDING)])
            
            print("📊 MongoDB indexes created")
            
        except Exception as e:
            print(f"⚠️ Could not create all indexes: {e}")
    
    def store_experiment_metadata(self, experiment_id: str, metadata: Dict[str, Any]) -> str:
        """
        Store experiment metadata
        
        Args:
            experiment_id: Unique experiment identifier
            metadata: Experiment metadata dictionary
        
        Returns:
            MongoDB document ID
        """
        try:
            document = {
                'experiment_id': experiment_id,
                'timestamp': datetime.now(),
                'metadata': metadata,
                'type': 'experiment_metadata'
            }
            
            result = self.collections['experiments'].insert_one(document)
            print(f"📝 Stored experiment metadata: {experiment_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store experiment metadata: {e}")
            return None
    
    def store_lezea_config(self, experiment_id: str, lezea_config: Dict[str, Any]) -> str:
        """
        Store LeZeA configuration
        
        Args:
            experiment_id: Experiment identifier
            lezea_config: LeZeA configuration dictionary
        
        Returns:
            MongoDB document ID
        """
        try:
            document = {
                'experiment_id': experiment_id,
                'timestamp': datetime.now(),
                'lezea_config': lezea_config,
                'type': 'lezea_config'
            }
            
            result = self.collections['experiments'].insert_one(document)
            print(f"⚙️ Stored LeZeA config for: {experiment_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store LeZeA config: {e}")
            return None
    
    def store_modification_tree(self, experiment_id: str, modification_data: Dict[str, Any]) -> str:
        """
        Store model modification tree data
        
        Args:
            experiment_id: Experiment identifier
            modification_data: Modification tree data
        
        Returns:
            MongoDB document ID
        """
        try:
            document = {
                'experiment_id': experiment_id,
                'timestamp': datetime.now(),
                'modification_data': modification_data,
                'type': 'modification_tree'
            }
            
            result = self.collections['modifications'].insert_one(document)
            print(f"🌳 Stored modification tree: {experiment_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store modification tree: {e}")
            return None
    
    def store_training_step(self, experiment_id: str, step_data: Dict[str, Any]) -> str:
        """
        Store training step data
        
        Args:
            experiment_id: Experiment identifier
            step_data: Training step data
        
        Returns:
            MongoDB document ID
        """
        try:
            document = {
                'experiment_id': experiment_id,
                'timestamp': datetime.now(),
                'step_data': step_data,
                'type': 'training_step'
            }
            
            result = self.collections['modifications'].insert_one(document)
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store training step: {e}")
            return None
    
    def store_resource_usage(self, experiment_id: str, resource_data: Dict[str, Any]) -> str:
        """
        Store resource usage data
        
        Args:
            experiment_id: Experiment identifier
            resource_data: Resource usage data
        
        Returns:
            MongoDB document ID
        """
        try:
            document = {
                'experiment_id': experiment_id,
                'timestamp': datetime.now(),
                'resource_data': resource_data,
                'type': 'resource_usage'
            }
            
            result = self.collections['resources'].insert_one(document)
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store resource usage: {e}")
            return None
    
    def store_business_metrics(self, experiment_id: str, business_data: Dict[str, Any]) -> str:
        """
        Store business metrics and cost data
        
        Args:
            experiment_id: Experiment identifier
            business_data: Business metrics data
        
        Returns:
            MongoDB document ID
        """
        try:
            document = {
                'experiment_id': experiment_id,
                'timestamp': datetime.now(),
                'business_data': business_data,
                'type': 'business_metrics'
            }
            
            result = self.collections['business'].insert_one(document)
            print(f"💰 Stored business metrics: {experiment_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store business metrics: {e}")
            return None
    
    def get_experiment_data(self, experiment_id: str, data_type: str = None) -> List[Dict]:
        """
        Get all data for an experiment
        
        Args:
            experiment_id: Experiment identifier
            data_type: Optional filter by data type
        
        Returns:
            List of documents
        """
        try:
            query = {'experiment_id': experiment_id}
            if data_type:
                query['type'] = data_type
            
            # Search across all relevant collections
            results = []
            
            for collection_name, collection in self.collections.items():
                cursor = collection.find(query).sort('timestamp', DESCENDING)
                for doc in cursor:
                    doc['_id'] = str(doc['_id'])
                    doc['collection'] = collection_name
                    results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"❌ Failed to get experiment data: {e}")
            return []
    
    def get_modification_history(self, experiment_id: str, step_range: tuple = None) -> List[Dict]:
        """
        Get modification history for an experiment
        
        Args:
            experiment_id: Experiment identifier
            step_range: Optional tuple of (start_step, end_step)
        
        Returns:
            List of modification documents
        """
        try:
            query = {
                'experiment_id': experiment_id,
                'type': 'modification_tree'
            }
            
            if step_range:
                query['modification_data.step'] = {
                    '$gte': step_range[0], 
                    '$lte': step_range[1]
                }
            
            cursor = self.collections['modifications'].find(query).sort('timestamp', ASCENDING)
            
            modifications = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                modifications.append(doc)
            
            return modifications
            
        except Exception as e:
            print(f"❌ Failed to get modification history: {e}")
            return []
    
    def get_resource_statistics(self, experiment_id: str, component_type: str = None) -> List[Dict]:
        """
        Get resource usage statistics
        
        Args:
            experiment_id: Experiment identifier
            component_type: Optional filter by component type
        
        Returns:
            List of resource usage documents
        """
        try:
            query = {'experiment_id': experiment_id}
            if component_type:
                query['resource_data.component_type'] = component_type
            
            cursor = self.collections['resources'].find(query).sort('timestamp', ASCENDING)
            
            resources = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                resources.append(doc)
            
            return resources
            
        except Exception as e:
            print(f"❌ Failed to get resource statistics: {e}")
            return []
    
    def get_business_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get business metrics summary
        
        Args:
            experiment_id: Experiment identifier
        
        Returns:
            Business summary dictionary
        """
        try:
            pipeline = [
                {'$match': {'experiment_id': experiment_id}},
                {'$group': {
                    '_id': None,
                    'total_cost': {'$sum': '$business_data.cost'},
                    'entry_count': {'$sum': 1},
                    'avg_cost': {'$avg': '$business_data.cost'},
                    'min_cost': {'$min': '$business_data.cost'},
                    'max_cost': {'$max': '$business_data.cost'},
                    'first_entry': {'$min': '$timestamp'},
                    'last_entry': {'$max': '$timestamp'}
                }}
            ]
            
            result = list(self.collections['business'].aggregate(pipeline))
            
            if result:
                summary = result[0]
                summary.pop('_id', None)
                return summary
            else:
                return {'total_cost': 0, 'entry_count': 0}
                
        except Exception as e:
            print(f"❌ Failed to get business summary: {e}")
            return {}
    
    def aggregate_resource_usage(self, experiment_id: str) -> Dict[str, Any]:
        """
        Aggregate resource usage statistics
        
        Args:
            experiment_id: Experiment identifier
        
        Returns:
            Aggregated resource statistics
        """
        try:
            pipeline = [
                {'$match': {'experiment_id': experiment_id}},
                {'$group': {
                    '_id': '$resource_data.component_type',
                    'total_gpu_time': {'$sum': '$resource_data.resources.gpu_time_seconds'},
                    'total_cpu_time': {'$sum': '$resource_data.resources.cpu_time_seconds'},
                    'peak_memory': {'$max': '$resource_data.resources.peak_memory_mb'},
                    'total_io': {'$sum': '$resource_data.resources.total_io_mb'},
                    'component_count': {'$sum': 1},
                    'avg_gpu_time': {'$avg': '$resource_data.resources.gpu_time_seconds'},
                    'avg_cpu_time': {'$avg': '$resource_data.resources.cpu_time_seconds'}
                }},
                {'$sort': {'_id': 1}}
            ]
            
            result = list(self.collections['resources'].aggregate(pipeline))
            
            # Convert to more readable format
            aggregated = {}
            for item in result:
                component_type = item['_id'] or 'unknown'
                aggregated[component_type] = {
                    'total_gpu_time_seconds': item.get('total_gpu_time', 0),
                    'total_cpu_time_seconds': item.get('total_cpu_time', 0),
                    'peak_memory_mb': item.get('peak_memory', 0),
                    'total_io_mb': item.get('total_io', 0),
                    'component_count': item.get('component_count', 0),
                    'avg_gpu_time_seconds': item.get('avg_gpu_time', 0),
                    'avg_cpu_time_seconds': item.get('avg_cpu_time', 0)
                }
            
            return aggregated
            
        except Exception as e:
            print(f"❌ Failed to aggregate resource usage: {e}")
            return {}
    
    def store_experiment_summary(self, experiment_id: str, summary: Dict[str, Any]) -> str:
        """
        Store final experiment summary
        
        Args:
            experiment_id: Experiment identifier
            summary: Experiment summary data
        
        Returns:
            MongoDB document ID
        """
        try:
            # Check if summary already exists
            existing = self.collections['experiments'].find_one({
                'experiment_id': experiment_id,
                'type': 'experiment_summary'
            })
            
            document = {
                'experiment_id': experiment_id,
                'timestamp': datetime.now(),
                'summary': summary,
                'type': 'experiment_summary'
            }
            
            if existing:
                # Update existing summary
                result = self.collections['experiments'].replace_one(
                    {'_id': existing['_id']}, 
                    document
                )
                doc_id = str(existing['_id'])
                print(f"📊 Updated experiment summary: {experiment_id}")
            else:
                # Create new summary
                result = self.collections['experiments'].insert_one(document)
                doc_id = str(result.inserted_id)
                print(f"📊 Stored experiment summary: {experiment_id}")
            
            return doc_id
            
        except Exception as e:
            print(f"❌ Failed to store experiment summary: {e}")
            return None
    
    def search_experiments(self, query_filter: Dict[str, Any] = None, 
                          limit: int = 100) -> List[Dict]:
        """
        Search experiments based on criteria
        
        Args:
            query_filter: MongoDB query filter
            limit: Maximum number of results
        
        Returns:
            List of experiment documents
        """
        try:
            if query_filter is None:
                query_filter = {}
            
            cursor = self.collections['experiments'].find(query_filter).limit(limit).sort('timestamp', DESCENDING)
            
            experiments = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                experiments.append(doc)
            
            return experiments
            
        except Exception as e:
            print(f"❌ Failed to search experiments: {e}")
            return []
    
    def delete_experiment_data(self, experiment_id: str) -> Dict[str, int]:
        """
        Delete all data for an experiment
        
        Args:
            experiment_id: Experiment identifier
        
        Returns:
            Dictionary with deletion counts per collection
        """
        try:
            deletion_counts = {}
            
            for collection_name, collection in self.collections.items():
                result = collection.delete_many({'experiment_id': experiment_id})
                deletion_counts[collection_name] = result.deleted_count
            
            total_deleted = sum(deletion_counts.values())
            print(f"🗑️ Deleted {total_deleted} documents for experiment {experiment_id}")
            
            return deletion_counts
            
        except Exception as e:
            print(f"❌ Failed to delete experiment data: {e}")
            return {}
    
    def cleanup_old_data(self, days_old: int = 30) -> int:
        """
        Clean up data older than specified days
        
        Args:
            days_old: Number of days old for cleanup threshold
        
        Returns:
            Number of deleted documents
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            total_deleted = 0
            
            for collection_name, collection in self.collections.items():
                result = collection.delete_many({
                    'timestamp': {'$lt': cutoff_date}
                })
                total_deleted += result.deleted_count
                print(f"🧹 Deleted {result.deleted_count} old documents from {collection_name}")
            
            print(f"🗑️ Total cleanup: {total_deleted} documents older than {days_old} days")
            return total_deleted
            
        except Exception as e:
            print(f"❌ Failed to cleanup old data: {e}")
            return 0
    
    def export_experiment_data(self, experiment_id: str, output_file: str):
        """
        Export all experiment data to JSON file
        
        Args:
            experiment_id: Experiment identifier
            output_file: Path to output JSON file
        """
        try:
            export_data = {
                'experiment_id': experiment_id,
                'export_timestamp': datetime.now().isoformat(),
                'collections': {}
            }
            
            # Export from each collection
            for collection_name, collection in self.collections.items():
                documents = list(collection.find({'experiment_id': experiment_id}))
                
                # Convert ObjectIds to strings
                for doc in documents:
                    doc['_id'] = str(doc['_id'])
                
                export_data['collections'][collection_name] = documents
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            total_docs = sum(len(docs) for docs in export_data['collections'].values())
            print(f"📁 Exported {total_docs} documents to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to export experiment data: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all collections
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = {}
            
            for collection_name, collection in self.collections.items():
                stats[collection_name] = {
                    'document_count': collection.count_documents({}),
                    'indexes': len(list(collection.list_indexes())),
                    'estimated_size': collection.estimated_document_count()
                }
            
            # Add database-level stats
            db_stats = self.database.command("dbStats")
            stats['database'] = {
                'name': self.mongo_config['database'],
                'collections': db_stats.get('collections', 0),
                'data_size_bytes': db_stats.get('dataSize', 0),
                'storage_size_bytes': db_stats.get('storageSize', 0),
                'indexes': db_stats.get('indexes', 0)
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Failed to get collection stats: {e}")
            return {}
    
    def create_backup(self, backup_file: str, experiment_ids: List[str] = None):
        """
        Create a backup of experiment data
        
        Args:
            backup_file: Path to save backup
            experiment_ids: Optional list of specific experiment IDs to backup
        """
        try:
            backup_data = {
                'backup_timestamp': datetime.now().isoformat(),
                'database_name': self.mongo_config['database'],
                'collections': {}
            }
            
            for collection_name, collection in self.collections.items():
                query = {}
                if experiment_ids:
                    query['experiment_id'] = {'$in': experiment_ids}
                
                documents = list(collection.find(query))
                
                # Convert ObjectIds to strings
                for doc in documents:
                    doc['_id'] = str(doc['_id'])
                
                backup_data['collections'][collection_name] = documents
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            total_docs = sum(len(docs) for docs in backup_data['collections'].values())
            print(f"💾 Created backup with {total_docs} documents: {backup_file}")
            
        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("🔌 MongoDB connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()