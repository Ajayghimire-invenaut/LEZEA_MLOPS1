"""
MongoDB Backend for LeZeA Experiment Tracking System

This module provides a comprehensive MongoDB backend for storing and retrieving
experiment data, model modifications, resource usage, and business metrics
in a machine learning experimentation framework.

Author: [Your Name/Team]
Date: [Date]
Version: 1.0.0
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

# MongoDB dependencies with graceful fallback
try:
    import pymongo
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    pymongo = None  # type: ignore


def _scope_min(scope: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Normalize scope dictionary to standardized format.
    
    Args:
        scope: Optional scope dictionary containing level and entity_id
        
    Returns:
        Normalized scope dictionary with 'level' and 'entity_id' keys
        Defaults to global scope if input is None or missing keys
    """
    if not scope:
        return {"level": "global", "entity_id": "-"}
    return {
        "level": str(scope.get("level", "global")), 
        "entity_id": str(scope.get("entity_id", "-"))
    }


class MongoBackend:
    """
    MongoDB backend for complex data storage and hierarchical queries.
    
    This class provides a comprehensive interface for storing experiment data
    in MongoDB with optimized collections, indexes, and query patterns.
    
    Features:
        - Connection management with retry timeouts
        - Optimized collections and compound indexes
        - Idempotent upserts for high-frequency writes
        - Aggregation helpers for resource and business metrics
        - Optional TTL on noisy collections
        - LeZeA-specific collections and operations
        
    Attributes:
        config: Configuration object containing MongoDB settings
        client: MongoDB client instance
        database: MongoDB database instance
        collections: Dictionary of collection handles
        available: Boolean indicating backend availability
    """
    
    def __init__(self, config):
        """
        Initialize MongoDB backend with configuration.
        
        Args:
            config: Configuration object with get_mongodb_config() method
            
        Raises:
            RuntimeError: If MongoDB is not available
            ConnectionError: If connection to MongoDB fails
        """
        if not MONGODB_AVAILABLE:
            raise RuntimeError(
                "MongoDB is not available. Install with: pip install pymongo"
            )
            
        self.config = config
        self.mongo_config = config.get_mongodb_config()
        self.client: Optional[MongoClient] = None
        self.database = None
        self.collections: Dict[str, Any] = {}
        self.available: bool = False
        
        # Initialize connection and setup
        self._connect()
        self._setup_collections()
        self._create_indexes()
        self.available = True
        
        print(f"✅ MongoDB backend connected: {self.mongo_config['database']}")

    def _connect(self) -> None:
        """
        Establish connection to MongoDB with production-ready settings.
        
        Configures connection with:
        - 5 second server selection timeout
        - 30 second socket timeout
        - Connection pooling (max 100 connections)
        - Write concern majority for consistency
        - Retry writes for resilience
        
        Raises:
            ConnectionError: If connection fails or ping test fails
        """
        try:
            self.client = MongoClient(
                self.mongo_config["connection_string"],
                serverSelectionTimeoutMS=5000,  # Fast failure detection
                socketTimeoutMS=30000,          # Reasonable timeout for ops
                maxPoolSize=100,                # Support concurrent operations
                retryWrites=True,               # Automatic retry on failure
                w="majority",                   # Ensure write acknowledgment
            )
            
            # Test connection with ping
            self.client.admin.command("ping")
            self.database = self.client[self.mongo_config["database"]]
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    def _setup_collections(self) -> None:
        """
        Initialize collection handles with configurable names.
        
        Sets up handles for all collections used by the system:
        - Core experiment collections (experiments, modifications, resources)
        - Business and dataset tracking collections  
        - LeZeA-specific collections (network_lineage, population_history, etc.)
        
        Collection names can be customized via configuration.
        """
        coll_cfg = self.mongo_config.get("collections", {})
        
        # Define collection name mappings with defaults
        collection_names = {
            "experiments": coll_cfg.get("experiments", "experiments"),
            "modifications": coll_cfg.get("modifications", "modifications"),
            "resources": coll_cfg.get("resources", "resources"),
            "business": coll_cfg.get("business", "business"),
            "datasets": coll_cfg.get("datasets", "datasets"),
            "rollups": coll_cfg.get("rollups", "rollups"),
            # LeZeA-specific collections
            "network_lineage": coll_cfg.get("network_lineage", "network_lineage"),
            "population_history": coll_cfg.get("population_history", "population_history"),
            "reward_flows": coll_cfg.get("reward_flows", "reward_flows"),
            "challenge_usage": coll_cfg.get("challenge_usage", "challenge_usage"),
            "learning_relevance": coll_cfg.get("learning_relevance", "learning_relevance"),
            "component_resources": coll_cfg.get("component_resources", "component_resources"),
        }
        
        # Create collection handles
        for key, name in collection_names.items():
            self.collections[key] = self.database[name]

    def _create_indexes(self) -> None:
        """
        Create optimized indexes for query performance and data integrity.
        
        Creates compound indexes optimized for:
        - Time-series queries (timestamp descending)
        - Experiment-scoped queries
        - Unique constraints for idempotent operations
        - TTL indexes for data lifecycle management
        
        Handles index creation failures gracefully to avoid blocking startup.
        """
        try:
            # Experiments collection indexes
            self._create_experiments_indexes()
            
            # Modifications collection indexes  
            self._create_modifications_indexes()
            
            # Resources collection indexes
            self._create_resources_indexes()
            
            # Business metrics indexes
            self._create_business_indexes()
            
            # Dataset tracking indexes
            self._create_datasets_indexes()
            
            # Rollup aggregation indexes
            self._create_rollups_indexes()
            
            # LeZeA-specific collection indexes
            self._create_lezea_indexes()
            
            print("📊 MongoDB indexes created (including LeZeA collections)")
            
        except Exception as e:
            print(f"⚠️ Could not create all indexes: {e}")

    def _create_experiments_indexes(self) -> None:
        """Create indexes for experiments collection."""
        exp = self.collections["experiments"]
        exp.create_index([
            ("experiment_id", ASCENDING), 
            ("type", ASCENDING), 
            ("timestamp", DESCENDING)
        ])
        exp.create_index([("timestamp", DESCENDING)])
        exp.create_index([("metadata.name", ASCENDING)], sparse=True)
        exp.create_index([("summary.status", ASCENDING)], sparse=True)
        exp.create_index([
            ("experiment_id", ASCENDING), 
            ("type", ASCENDING)
        ], name="exp_type_unique", unique=False)

    def _create_modifications_indexes(self) -> None:
        """Create indexes for modifications collection with unique constraints."""
        mod = self.collections["modifications"]
        
        # Unique index for training steps
        mod.create_index(
            [
                ("experiment_id", ASCENDING), 
                ("type", ASCENDING),
                ("step_data.step", ASCENDING), 
                ("step_data.scope.level", ASCENDING),
                ("step_data.scope.entity_id", ASCENDING)
            ],
            name="step_scope_unique",
            unique=True,
            partialFilterExpression={"type": "training_step"},
        )
        
        # Unique index for modification trees
        mod.create_index(
            [
                ("experiment_id", ASCENDING), 
                ("type", ASCENDING),
                ("modification_data.step", ASCENDING), 
                ("modification_data.scope.level", ASCENDING),
                ("modification_data.scope.entity_id", ASCENDING)
            ],
            name="modtree_step_scope_unique",
            unique=True,
            partialFilterExpression={"type": "modification_tree"},
        )
        
        # General query indexes
        mod.create_index([("timestamp", DESCENDING)])
        mod.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])

    def _create_resources_indexes(self) -> None:
        """Create indexes for resources collection with optional TTL."""
        res = self.collections["resources"]
        res.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])
        res.create_index([("resource_data.component_type", ASCENDING)])
        
        # Optional TTL for resource cleanup
        ttl_days = int(self.mongo_config.get("resources_ttl_days", 0) or 0)
        if ttl_days > 0:
            try:
                res.create_index(
                    [("timestamp", ASCENDING)],
                    expireAfterSeconds=ttl_days * 24 * 3600,
                    name="resources_ttl",
                )
            except Exception:
                # TTL index creation is optional
                pass

    def _create_business_indexes(self) -> None:
        """Create indexes for business metrics collection."""
        bus = self.collections["business"]
        bus.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])
        bus.create_index([("business_data.cost", DESCENDING)])

    def _create_datasets_indexes(self) -> None:
        """Create indexes for datasets collection."""
        ds = self.collections["datasets"]
        ds.create_index([
            ("experiment_id", ASCENDING), 
            ("type", ASCENDING), 
            ("timestamp", DESCENDING)
        ])
        ds.create_index([("dataset_name", ASCENDING)], sparse=True)

    def _create_rollups_indexes(self) -> None:
        """Create indexes for rollups collection."""
        roll = self.collections["rollups"]
        roll.create_index([
            ("experiment_id", ASCENDING), 
            ("scope.level", ASCENDING),
            ("scope.entity_id", ASCENDING), 
            ("bucket", ASCENDING)
        ], name="rollup_scope_bucket", unique=True)

    def _create_lezea_indexes(self) -> None:
        """Create indexes for LeZeA-specific collections."""
        # Network lineage indexes
        lineage = self.collections["network_lineage"]
        lineage.create_index([
            ("experiment_id", ASCENDING), 
            ("network_id", ASCENDING)
        ], unique=True)
        lineage.create_index([("experiment_id", ASCENDING), ("generation", ASCENDING)])
        lineage.create_index([("experiment_id", ASCENDING), ("parent_ids", ASCENDING)])
        lineage.create_index([("fitness_score", DESCENDING)])

        # Population history indexes
        pop = self.collections["population_history"]
        pop.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])
        pop.create_index([("experiment_id", ASCENDING), ("generation", ASCENDING)])
        pop.create_index([("avg_fitness", DESCENDING)])

        # Reward flows indexes
        rewards = self.collections["reward_flows"]
        rewards.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])
        rewards.create_index([
            ("experiment_id", ASCENDING), 
            ("source_id", ASCENDING), 
            ("target_id", ASCENDING)
        ])
        rewards.create_index([("source_type", ASCENDING), ("target_type", ASCENDING)])
        rewards.create_index([("task_id", ASCENDING)])
        rewards.create_index([("reward_value", DESCENDING)])

        # Challenge usage indexes
        challenge = self.collections["challenge_usage"]
        challenge.create_index([
            ("experiment_id", ASCENDING), 
            ("challenge_id", ASCENDING), 
            ("timestamp", DESCENDING)
        ])
        challenge.create_index([("challenge_id", ASCENDING), ("difficulty_level", ASCENDING)])
        challenge.create_index([("usage_rate", DESCENDING)])

        # Learning relevance indexes
        relevance = self.collections["learning_relevance"]
        relevance.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])
        relevance.create_index([("sample_ids", ASCENDING)], sparse=True)
        relevance.create_index([("avg_relevance", DESCENDING)])

        # Component resources indexes
        comp_res = self.collections["component_resources"]
        comp_res.create_index([
            ("experiment_id", ASCENDING), 
            ("component_id", ASCENDING), 
            ("timestamp", DESCENDING)
        ])
        comp_res.create_index([("component_type", ASCENDING)])
        comp_res.create_index([("cpu_percent", DESCENDING)])
        comp_res.create_index([("memory_mb", DESCENDING)])

    def ping(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if database is reachable, False otherwise
        """
        try:
            if not self.client:
                return False
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def store_experiment_metadata(self, experiment_id: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Store experiment metadata as an append-only event.
        
        Args:
            experiment_id: Unique identifier for the experiment
            metadata: Dictionary containing experiment metadata
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            document = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "metadata": metadata,
                "type": "experiment_metadata",
            }
            result = self.collections["experiments"].insert_one(document)
            print(f"📝 Stored experiment metadata: {experiment_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to store experiment metadata: {e}")
            return None

    def store_lezea_config(self, experiment_id: str, lezea_config: Dict[str, Any]) -> Optional[str]:
        """
        Store LeZeA configuration with versioning over time.
        
        Args:
            experiment_id: Unique identifier for the experiment
            lezea_config: LeZeA configuration dictionary
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            document = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "lezea_config": lezea_config,
                "type": "lezea_config",
            }
            result = self.collections["experiments"].insert_one(document)
            print(f"⚙️ Stored LeZeA config for: {experiment_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to store LeZeA config: {e}")
            return None

    def store_modification_tree(self, experiment_id: str, modification_data: Dict[str, Any]) -> Optional[str]:
        """
        Store model modification tree with idempotent upsert behavior.
        
        If step is present in modification_data, performs upsert based on
        unique key: (experiment_id, type, step, scope.level, scope.entity_id)
        
        Args:
            experiment_id: Unique identifier for the experiment
            modification_data: Dictionary containing modification tree data
            
        Returns:
            'upserted', 'replaced', document ID, or None if failed
        """
        try:
            coll = self.collections["modifications"]
            scope = _scope_min(modification_data.get("scope"))
            step = modification_data.get("step")
            
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "modification_data": {**modification_data, "scope": scope},
                "type": "modification_tree",
            }
            
            # Perform upsert if step is defined
            if step is not None:
                filter_query = {
                    "experiment_id": experiment_id,
                    "type": "modification_tree",
                    "modification_data.step": step,
                    "modification_data.scope.level": scope["level"],
                    "modification_data.scope.entity_id": scope["entity_id"],
                }
                coll.update_one(filter_query, {"$set": doc}, upsert=True)
                return "upserted"
            else:
                # Insert new document if no step specified
                result = coll.insert_one(doc)
                return str(result.inserted_id)
                
        except DuplicateKeyError:
            # Handle race condition with replace operation
            try:
                coll = self.collections["modifications"]
                scope = _scope_min(modification_data.get("scope"))
                step = modification_data.get("step")
                
                filter_query = {
                    "experiment_id": experiment_id,
                    "type": "modification_tree",
                    "modification_data.step": step,
                    "modification_data.scope.level": scope["level"],
                    "modification_data.scope.entity_id": scope["entity_id"],
                }
                
                replacement_doc = {
                    "experiment_id": experiment_id,
                    "timestamp": datetime.now(),
                    "modification_data": {**modification_data, "scope": scope},
                    "type": "modification_tree",
                }
                
                coll.replace_one(filter_query, replacement_doc, upsert=True)
                return "replaced"
            except Exception:
                return None
        except Exception as e:
            print(f"❌ Failed to store modification tree: {e}")
            return None

    def store_training_step(self, experiment_id: str, step_data: Dict[str, Any]) -> Optional[str]:
        """
        Store training step data with idempotent behavior.
        
        Unique key: (experiment_id, type='training_step', step, scope.level, scope.entity_id)
        
        Args:
            experiment_id: Unique identifier for the experiment
            step_data: Dictionary containing training step data
            
        Returns:
            'upserted', document ID, or None if failed
        """
        try:
            coll = self.collections["modifications"]
            scope = _scope_min(step_data.get("scope"))
            step = step_data.get("step")
            
            # Insert without upsert if no step number
            if step is None:
                doc = {
                    "experiment_id": experiment_id,
                    "timestamp": datetime.now(),
                    "step_data": {**step_data, "scope": scope},
                    "type": "training_step",
                }
                result = coll.insert_one(doc)
                return str(result.inserted_id)
            
            # Upsert based on step and scope
            filter_query = {
                "experiment_id": experiment_id,
                "type": "training_step",
                "step_data.step": step,
                "step_data.scope.level": scope["level"],
                "step_data.scope.entity_id": scope["entity_id"],
            }
            
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "step_data": {**step_data, "scope": scope},
                "type": "training_step",
            }
            
            coll.update_one(filter_query, {"$set": doc}, upsert=True)
            return "upserted"
            
        except Exception as e:
            print(f"❌ Failed to store training step: {e}")
            return None

    def store_resource_usage(self, experiment_id: str, resource_data: Dict[str, Any]) -> Optional[str]:
        """
        Store resource usage sample as append-only event.
        
        Args:
            experiment_id: Unique identifier for the experiment
            resource_data: Dictionary containing resource usage metrics
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            document = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "resource_data": resource_data,
                "type": "resource_usage",
            }
            result = self.collections["resources"].insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to store resource usage: {e}")
            return None

    def upsert_scope_rollup(self, experiment_id: str, scope: Dict[str, Any], 
                           bucket: str, values: Dict[str, Any]) -> None:
        """
        Upsert summarized resource metrics per scope and time bucket.
        
        Args:
            experiment_id: Unique identifier for the experiment
            scope: Scope dictionary (level and entity_id)
            bucket: Time bucket identifier (e.g., '2025-08-10T10:05Z')
            values: Aggregated values dictionary
        """
        try:
            scope_normalized = _scope_min(scope)
            
            filter_query = {
                "experiment_id": experiment_id,
                "scope.level": scope_normalized["level"],
                "scope.entity_id": scope_normalized["entity_id"],
                "bucket": bucket,
            }
            
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "scope": scope_normalized,
                "bucket": bucket,
                "values": values,
            }
            
            self.collections["rollups"].update_one(filter_query, {"$set": doc}, upsert=True)
            
        except Exception as e:
            print(f"❌ Failed to upsert scope rollup: {e}")

    def store_business_metrics(self, experiment_id: str, business_data: Dict[str, Any]) -> Optional[str]:
        """
        Store business metrics and cost data.
        
        Args:
            experiment_id: Unique identifier for the experiment
            business_data: Dictionary containing business metrics and costs
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            document = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "business_data": business_data,
                "type": "business_metrics",
            }
            result = self.collections["business"].insert_one(document)
            print(f"💰 Stored business metrics: {experiment_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to store business metrics: {e}")
            return None

    def store_data_splits(self, experiment_id: str, payload: Dict[str, Any]) -> Optional[str]:
        """
        Store data split counts and metadata.
        
        Args:
            experiment_id: Unique identifier for the experiment
            payload: Dictionary containing data split information
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            document = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "type": "data_splits",
                "payload": payload,
            }
            result = self.collections["datasets"].insert_one(document)
            print(f"🧩 Stored data splits for {experiment_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to store data splits: {e}")
            return None

    def store_dataset_version(self, experiment_id: str, dataset_name: str, 
                            info: Dict[str, Any]) -> Optional[str]:
        """
        Store dataset version and fingerprint information.
        
        Args:
            experiment_id: Unique identifier for the experiment
            dataset_name: Name of the dataset
            info: Dictionary containing version and fingerprint data
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            document = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "type": "dataset_version",
                "dataset_name": dataset_name,
                "info": info,
            }
            result = self.collections["datasets"].insert_one(document)
            print(f"📦 Stored dataset version [{dataset_name}] for {experiment_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to store dataset version: {e}")
            return None

    def store_results(self, experiment_id: str, payload: Dict[str, Any]) -> Optional[str]:
        """
        Store result events with optional idempotent behavior.
        
        Handles various result types: tasker/builder rewards, RL episodes,
        classification results, generation outputs, summaries.
        
        If payload contains 'step' and 'scope', performs upsert based on
        unique key: (experiment_id, type, step, scope.level, scope.entity_id)
        
        Args:
            experiment_id: Unique identifier for the experiment
            payload: Dictionary containing result data
            
        Returns:
            'upserted', document ID, or None if failed
        """
        try:
            coll = self.collections["experiments"]
            kind = payload.get("kind", "result_event")
            scope = _scope_min(payload.get("scope"))
            step = payload.get("step", None)
            
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "type": kind,
                **payload,
                "scope": scope,
            }
            
            # Perform upsert if step and scope are defined
            if step is not None:
                filter_query = {
                    "experiment_id": experiment_id,
                    "type": kind,
                    "step": step,
                    "scope.level": scope["level"],
                    "scope.entity_id": scope["entity_id"],
                }
                coll.update_one(filter_query, {"$set": doc}, upsert=True)
                return "upserted"
            else:
                # Insert new document if no step specified
                result = coll.insert_one(doc)
                return str(result.inserted_id)
                
        except Exception as e:
            print(f"❌ Failed to store results: {e}")
            return None

    def store_experiment_summary(self, experiment_id: str, summary: Dict[str, Any]) -> Optional[str]:
        """
        Upsert final experiment summary (one document per experiment).
        
        Args:
            experiment_id: Unique identifier for the experiment
            summary: Dictionary containing experiment summary data
            
        Returns:
            'upserted' if successful, None if failed
        """
        try:
            coll = self.collections["experiments"]
            filter_query = {"experiment_id": experiment_id, "type": "experiment_summary"}
            
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "summary": summary,
                "type": "experiment_summary",
            }
            
            coll.update_one(filter_query, {"$set": doc}, upsert=True)
            print(f"📊 Upserted experiment summary: {experiment_id}")
            return "upserted"
            
        except Exception as e:
            print(f"❌ Failed to store experiment summary: {e}")
            return None

    def store_network_lineage(self, experiment_id: str, network_id: str, 
                            lineage_data: Dict[str, Any]) -> Optional[str]:
        """
        Store or update network lineage information.
        
        Args:
            experiment_id: Unique identifier for the experiment
            network_id: Unique identifier for the network
            lineage_data: Dictionary containing lineage information
            
        Returns:
            'upserted' if successful, None if failed
        """
        try:
            doc = {
                "experiment_id": experiment_id,
                "network_id": network_id,
                "timestamp": datetime.now(),
                **lineage_data
            }
            
            filter_query = {"experiment_id": experiment_id, "network_id": network_id}
            self.collections["network_lineage"].update_one(filter_query, {"$set": doc}, upsert=True)
            print(f"🧬 Stored network lineage: {network_id}")
            return "upserted"
            
        except Exception as e:
            print(f"❌ Failed to store network lineage: {e}")
            return None

    def store_population_snapshot(self, experiment_id: str, snapshot_data: Dict[str, Any]) -> Optional[str]:
        """
        Store population snapshot for evolutionary algorithms.
        
        Args:
            experiment_id: Unique identifier for the experiment
            snapshot_data: Dictionary containing population state data
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                **snapshot_data
            }
            
            result = self.collections["population_history"].insert_one(doc)
            generation = snapshot_data.get('generation', 'unknown')
            print(f"👥 Stored population snapshot: generation {generation}")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store population snapshot: {e}")
            return None

    def store_reward_flow(self, experiment_id: str, flow_data: Dict[str, Any]) -> Optional[str]:
        """
        Store reward flow between networks in the LeZeA system.
        
        Args:
            experiment_id: Unique identifier for the experiment
            flow_data: Dictionary containing reward flow data
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                **flow_data
            }
            
            result = self.collections["reward_flows"].insert_one(doc)
            source_id = flow_data.get('source_id', '?')
            target_id = flow_data.get('target_id', '?')
            print(f"💰 Stored reward flow: {source_id} → {target_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store reward flow: {e}")
            return None

    def store_challenge_usage(self, experiment_id: str, usage_data: Dict[str, Any]) -> Optional[str]:
        """
        Store challenge-specific usage rates and metrics.
        
        Args:
            experiment_id: Unique identifier for the experiment
            usage_data: Dictionary containing challenge usage data
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                **usage_data
            }
            
            result = self.collections["challenge_usage"].insert_one(doc)
            challenge_id = usage_data.get('challenge_id', 'unknown')
            print(f"📈 Stored challenge usage: {challenge_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store challenge usage: {e}")
            return None

    def store_learning_relevance(self, experiment_id: str, relevance_data: Dict[str, Any]) -> Optional[str]:
        """
        Store learning relevance data and sample analysis.
        
        Args:
            experiment_id: Unique identifier for the experiment
            relevance_data: Dictionary containing relevance metrics
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                **relevance_data
            }
            
            result = self.collections["learning_relevance"].insert_one(doc)
            sample_count = len(relevance_data.get('sample_ids', []))
            print(f"🎯 Stored learning relevance: {sample_count} samples")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store learning relevance: {e}")
            return None

    def store_component_resource(self, experiment_id: str, component_data: Dict[str, Any]) -> Optional[str]:
        """
        Store component-level resource usage metrics.
        
        Args:
            experiment_id: Unique identifier for the experiment
            component_data: Dictionary containing component resource data
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                **component_data
            }
            
            result = self.collections["component_resources"].insert_one(doc)
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store component resource: {e}")
            return None

    def store_environment(self, experiment_id: str, env: Dict[str, Any]) -> Optional[str]:
        """
        Store environment configuration and system information.
        
        Args:
            experiment_id: Unique identifier for the experiment
            env: Dictionary containing environment data
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "environment_data": env,
                "type": "environment",
            }
            
            result = self.collections["experiments"].insert_one(doc)
            print(f"🌐 Stored environment data for: {experiment_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"❌ Failed to store environment data: {e}")
            return None

    # Query and Retrieval Methods
    
    def get_experiment_data(self, experiment_id: str, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all data across collections for a specific experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            data_type: Optional filter by document type
            
        Returns:
            List of documents with collection metadata
        """
        try:
            query = {"experiment_id": experiment_id}
            if data_type:
                query["type"] = data_type
                
            results: List[Dict[str, Any]] = []
            
            # Search across all collections
            for collection_name, collection in self.collections.items():
                cursor = collection.find(query).sort("timestamp", DESCENDING)
                
                for doc in cursor:
                    # Convert ObjectId to string and add collection metadata
                    doc["_id"] = str(doc.get("_id", ""))
                    doc["collection"] = collection_name
                    results.append(doc)
                    
            return results
            
        except Exception as e:
            print(f"❌ Failed to get experiment data: {e}")
            return []

    def get_modification_history(self, experiment_id: str, 
                               step_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve modification history for an experiment with optional step filtering.
        
        Args:
            experiment_id: Unique identifier for the experiment
            step_range: Optional tuple of (min_step, max_step) for filtering
            
        Returns:
            List of modification documents sorted by timestamp
        """
        try:
            query: Dict[str, Any] = {
                "experiment_id": experiment_id, 
                "type": "modification_tree"
            }
            
            # Add step range filter if provided
            if step_range:
                query["modification_data.step"] = {
                    "$gte": step_range[0], 
                    "$lte": step_range[1]
                }
                
            cursor = self.collections["modifications"].find(query).sort("timestamp", ASCENDING)
            modifications: List[Dict[str, Any]] = []
            
            for doc in cursor:
                doc["_id"] = str(doc.get("_id", ""))
                modifications.append(doc)
                
            return modifications
            
        except Exception as e:
            print(f"❌ Failed to get modification history: {e}")
            return []

    def get_resource_statistics(self, experiment_id: str, 
                              component_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve resource usage documents with optional component filtering.
        
        Args:
            experiment_id: Unique identifier for the experiment
            component_type: Optional filter by component type
            
        Returns:
            List of resource usage documents sorted by timestamp
        """
        try:
            query: Dict[str, Any] = {"experiment_id": experiment_id}
            
            if component_type:
                query["resource_data.component_type"] = component_type
                
            cursor = self.collections["resources"].find(query).sort("timestamp", ASCENDING)
            resources: List[Dict[str, Any]] = []
            
            for doc in cursor:
                doc["_id"] = str(doc.get("_id", ""))
                resources.append(doc)
                
            return resources
            
        except Exception as e:
            print(f"❌ Failed to get resource statistics: {e}")
            return []

    def get_business_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Generate aggregated business metrics summary for an experiment.
        
        Uses MongoDB aggregation pipeline to compute:
        - Total, average, min, max costs
        - Entry count and time range
        
        Args:
            experiment_id: Unique identifier for the experiment
            
        Returns:
            Dictionary containing aggregated business metrics
        """
        try:
            pipeline = [
                {"$match": {"experiment_id": experiment_id}},
                {
                    "$group": {
                        "_id": None,
                        "total_cost": {"$sum": "$business_data.cost"},
                        "entry_count": {"$sum": 1},
                        "avg_cost": {"$avg": "$business_data.cost"},
                        "min_cost": {"$min": "$business_data.cost"},
                        "max_cost": {"$max": "$business_data.cost"},
                        "first_entry": {"$min": "$timestamp"},
                        "last_entry": {"$max": "$timestamp"},
                    }
                },
            ]
            
            result = list(self.collections["business"].aggregate(pipeline))
            
            if result:
                summary = result[0]
                summary.pop("_id", None)  # Remove aggregation ID
                return summary
                
            # Return default values if no data found
            return {"total_cost": 0, "entry_count": 0}
            
        except Exception as e:
            print(f"❌ Failed to get business summary: {e}")
            return {}

    def aggregate_resource_usage(self, experiment_id: str) -> Dict[str, Any]:
        """
        Generate aggregated resource usage statistics by component type.
        
        Uses MongoDB aggregation pipeline to compute resource totals,
        peaks, and averages grouped by component type.
        
        Args:
            experiment_id: Unique identifier for the experiment
            
        Returns:
            Dictionary with component types as keys and aggregated metrics as values
        """
        try:
            pipeline = [
                {"$match": {"experiment_id": experiment_id}},
                {
                    "$group": {
                        "_id": "$resource_data.component_type",
                        "total_gpu_time": {"$sum": "$resource_data.resources.gpu_time_seconds"},
                        "total_cpu_time": {"$sum": "$resource_data.resources.cpu_time_seconds"},
                        "peak_memory": {"$max": "$resource_data.resources.peak_memory_mb"},
                        "total_io": {"$sum": "$resource_data.resources.total_io_mb"},
                        "component_count": {"$sum": 1},
                        "avg_gpu_time": {"$avg": "$resource_data.resources.gpu_time_seconds"},
                        "avg_cpu_time": {"$avg": "$resource_data.resources.cpu_time_seconds"},
                    }
                },
                {"$sort": {"_id": 1}},  # Sort by component type
            ]
            
            result = list(self.collections["resources"].aggregate(pipeline))
            aggregated: Dict[str, Any] = {}
            
            # Transform aggregation results into structured format
            for item in result:
                component_type = item.get("_id") or "unknown"
                aggregated[component_type] = {
                    "total_gpu_time_seconds": item.get("total_gpu_time", 0),
                    "total_cpu_time_seconds": item.get("total_cpu_time", 0),
                    "peak_memory_mb": item.get("peak_memory", 0),
                    "total_io_mb": item.get("total_io", 0),
                    "component_count": item.get("component_count", 0),
                    "avg_gpu_time_seconds": item.get("avg_gpu_time", 0),
                    "avg_cpu_time_seconds": item.get("avg_cpu_time", 0),
                }
                
            return aggregated
            
        except Exception as e:
            print(f"❌ Failed to aggregate resource usage: {e}")
            return {}

    def search_experiments(self, query_filter: Optional[Dict[str, Any]] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search experiments based on flexible criteria.
        
        Args:
            query_filter: MongoDB query filter (defaults to empty for all experiments)
            limit: Maximum number of results to return
            
        Returns:
            List of experiment documents sorted by timestamp (newest first)
        """
        try:
            query = query_filter or {}
            cursor = (self.collections["experiments"]
                     .find(query)
                     .limit(limit)
                     .sort("timestamp", DESCENDING))
            
            experiments: List[Dict[str, Any]] = []
            for doc in cursor:
                doc["_id"] = str(doc.get("_id", ""))
                experiments.append(doc)
                
            return experiments
            
        except Exception as e:
            print(f"❌ Failed to search experiments: {e}")
            return []

    # Connection Management Methods
    
    def close(self) -> None:
        """
        Close MongoDB connection and cleanup resources.
        
        Should be called when the backend is no longer needed
        to properly release database connections.
        """
        if self.client:
            self.client.close()
            print("🔌 MongoDB connection closed")

    def __enter__(self) -> "MongoBackend":
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point with automatic cleanup."""
        self.close()