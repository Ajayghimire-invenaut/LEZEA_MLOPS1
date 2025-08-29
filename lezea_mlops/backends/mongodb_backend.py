from __future__ import annotations
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
try:
    import pymongo
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    pymongo = None  # type: ignore

def _scope_min(scope: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Normalize scope to {level, entity_id} (or global)."""
    if not scope:
        return {"level": "global", "entity_id": "-"}
    return {"level": str(scope.get("level", "global")), "entity_id": str(scope.get("entity_id", "-"))}

class MongoBackend:
    """
    MongoDB backend for complex data storage and hierarchical queries.
    Features:
    - Connection management with retry-ish timeouts
    - Optimized collections and compound indexes (scope + step)
    - Idempotent upserts for high-frequency writes
    - Aggregation helpers (resource/business)
    - Optional TTL on noisy collections
    - NEW: LeZeA-specific collections and operations
    """
    def __init__(self, config):
        if not MONGODB_AVAILABLE:
            raise RuntimeError("MongoDB is not available. Install with: pip install pymongo")
        self.config = config
        self.mongo_config = config.get_mongodb_config()
        self.client: Optional[MongoClient] = None
        self.database = None
        self.collections: Dict[str, Any] = {}
        self.available: bool = False
        self._connect()
        self._setup_collections()
        self._create_indexes()
        self.available = True
        print(f"✅ MongoDB backend connected: {self.mongo_config['database']}")

    def _connect(self) -> None:
        """Establish connection to MongoDB with sane timeouts."""
        try:
            self.client = MongoClient(
                self.mongo_config["connection_string"],
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=30000,
                maxPoolSize=100,
                retryWrites=True,
                w="majority",
            )
            self.client.admin.command("ping")
            self.database = self.client[self.mongo_config["database"]]
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    def _setup_collections(self) -> None:
        """Bind collection handles with configurable names."""
        coll_cfg = self.mongo_config.get("collections", {})
        names = {
            "experiments": coll_cfg.get("experiments", "experiments"),
            "modifications": coll_cfg.get("modifications", "modifications"),
            "resources": coll_cfg.get("resources", "resources"),
            "business": coll_cfg.get("business", "business"),
            "datasets": coll_cfg.get("datasets", "datasets"),
            "rollups": coll_cfg.get("rollups", "rollups"),
            "network_lineage": coll_cfg.get("network_lineage", "network_lineage"),
            "population_history": coll_cfg.get("population_history", "population_history"),
            "reward_flows": coll_cfg.get("reward_flows", "reward_flows"),
            "challenge_usage": coll_cfg.get("challenge_usage", "challenge_usage"),
            "learning_relevance": coll_cfg.get("learning_relevance", "learning_relevance"),
            "component_resources": coll_cfg.get("component_resources", "component_resources"),
        }
        for key, name in names.items():
            self.collections[key] = self.database[name]

    def _create_indexes(self) -> None:
        """Create indexes for optimal query performance and idempotency."""
        try:
            exp = self.collections["experiments"]
            exp.create_index([("experiment_id", ASCENDING), ("type", ASCENDING), ("timestamp", DESCENDING)])
            exp.create_index([("timestamp", DESCENDING)])
            exp.create_index([("metadata.name", ASCENDING)], sparse=True)
            exp.create_index([("summary.status", ASCENDING)], sparse=True)
            exp.create_index([("experiment_id", ASCENDING), ("type", ASCENDING)], name="exp_type_unique", unique=False)

            mod = self.collections["modifications"]
            mod.create_index(
                [("experiment_id", ASCENDING), ("type", ASCENDING),
                 ("step_data.step", ASCENDING), ("step_data.scope.level", ASCENDING),
                 ("step_data.scope.entity_id", ASCENDING)],
                name="step_scope_unique",
                unique=True,
                partialFilterExpression={"type": "training_step"},
            )
            mod.create_index(
                [("experiment_id", ASCENDING), ("type", ASCENDING),
                 ("modification_data.step", ASCENDING), ("modification_data.scope.level", ASCENDING),
                 ("modification_data.scope.entity_id", ASCENDING)],
                name="modtree_step_scope_unique",
                unique=True,
                partialFilterExpression={"type": "modification_tree"},
            )
            mod.create_index([("timestamp", DESCENDING)])
            mod.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])

            res = self.collections["resources"]
            res.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])
            res.create_index([("resource_data.component_type", ASCENDING)])
            ttl_days = int(self.mongo_config.get("resources_ttl_days", 0) or 0)
            if ttl_days > 0:
                try:
                    res.create_index(
                        [("timestamp", ASCENDING)],
                        expireAfterSeconds=ttl_days * 24 * 3600,
                        name="resources_ttl",
                    )
                except Exception:
                    pass

            bus = self.collections["business"]
            bus.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])
            bus.create_index([("business_data.cost", DESCENDING)])

            ds = self.collections["datasets"]
            ds.create_index([("experiment_id", ASCENDING), ("type", ASCENDING), ("timestamp", DESCENDING)])
            ds.create_index([("dataset_name", ASCENDING)], sparse=True)

            roll = self.collections["rollups"]
            roll.create_index([("experiment_id", ASCENDING), ("scope.level", ASCENDING),
                              ("scope.entity_id", ASCENDING), ("bucket", ASCENDING)],
                             name="rollup_scope_bucket", unique=True)

            lineage = self.collections["network_lineage"]
            lineage.create_index([("experiment_id", ASCENDING), ("network_id", ASCENDING)], unique=True)
            lineage.create_index([("experiment_id", ASCENDING), ("generation", ASCENDING)])
            lineage.create_index([("experiment_id", ASCENDING), ("parent_ids", ASCENDING)])
            lineage.create_index([("fitness_score", DESCENDING)])

            pop = self.collections["population_history"]
            pop.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])
            pop.create_index([("experiment_id", ASCENDING), ("generation", ASCENDING)])
            pop.create_index([("avg_fitness", DESCENDING)])

            rewards = self.collections["reward_flows"]
            rewards.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])
            rewards.create_index([("experiment_id", ASCENDING), ("source_id", ASCENDING), ("target_id", ASCENDING)])
            rewards.create_index([("source_type", ASCENDING), ("target_type", ASCENDING)])
            rewards.create_index([("task_id", ASCENDING)])
            rewards.create_index([("reward_value", DESCENDING)])

            challenge = self.collections["challenge_usage"]
            challenge.create_index([("experiment_id", ASCENDING), ("challenge_id", ASCENDING), ("timestamp", DESCENDING)])
            challenge.create_index([("challenge_id", ASCENDING), ("difficulty_level", ASCENDING)])
            challenge.create_index([("usage_rate", DESCENDING)])

            relevance = self.collections["learning_relevance"]
            relevance.create_index([("experiment_id", ASCENDING), ("timestamp", DESCENDING)])
            relevance.create_index([("sample_ids", ASCENDING)], sparse=True)
            relevance.create_index([("avg_relevance", DESCENDING)])

            comp_res = self.collections["component_resources"]
            comp_res.create_index([("experiment_id", ASCENDING), ("component_id", ASCENDING), ("timestamp", DESCENDING)])
            comp_res.create_index([("component_type", ASCENDING)])
            comp_res.create_index([("cpu_percent", DESCENDING)])
            comp_res.create_index([("memory_mb", DESCENDING)])

            print("📊 MongoDB indexes created (including LeZeA collections)")
        except Exception as e:
            print(f"⚠️ Could not create all indexes: {e}")

    def ping(self) -> bool:
        try:
            if not self.client:
                return False
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def store_experiment_metadata(self, experiment_id: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Append experiment metadata (non-unique stream)."""
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
        """Append LeZeA configuration (versioned over time)."""
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
        Store model modification tree.
        If 'step' is present, we upsert by (experiment_id, type, step, scope).
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
            if step is not None:
                filt = {
                    "experiment_id": experiment_id,
                    "type": "modification_tree",
                    "modification_data.step": step,
                    "modification_data.scope.level": scope["level"],
                    "modification_data.scope.entity_id": scope["entity_id"],
                }
                coll.update_one(filt, {"$set": doc}, upsert=True)
                return "upserted"
            else:
                result = coll.insert_one(doc)
                return str(result.inserted_id)
        except DuplicateKeyError:
            try:
                coll = self.collections["modifications"]
                scope = _scope_min(modification_data.get("scope"))
                step = modification_data.get("step")
                filt = {
                    "experiment_id": experiment_id,
                    "type": "modification_tree",
                    "modification_data.step": step,
                    "modification_data.scope.level": scope["level"],
                    "modification_data.scope.entity_id": scope["entity_id"],
                }
                coll.replace_one(filt, {
                    "experiment_id": experiment_id,
                    "timestamp": datetime.now(),
                    "modification_data": {**modification_data, "scope": scope},
                    "type": "modification_tree",
                }, upsert=True)
                return "replaced"
            except Exception:
                return None
        except Exception as e:
            print(f"❌ Failed to store modification tree: {e}")
            return None

    def store_training_step(self, experiment_id: str, step_data: Dict[str, Any]) -> Optional[str]:
        """
        Idempotent store of training step:
        unique key: (experiment_id, type='training_step', step, scope.level, scope.entity_id)
        """
        try:
            coll = self.collections["modifications"]
            scope = _scope_min(step_data.get("scope"))
            step = step_data.get("step")
            if step is None:
                doc = {
                    "experiment_id": experiment_id,
                    "timestamp": datetime.now(),
                    "step_data": {**step_data, "scope": scope},
                    "type": "training_step",
                }
                result = coll.insert_one(doc)
                return str(result.inserted_id)
            filt = {
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
            coll.update_one(filt, {"$set": doc}, upsert=True)
            return "upserted"
        except Exception as e:
            print(f"❌ Failed to store training step: {e}")
            return None

    def store_resource_usage(self, experiment_id: str, resource_data: Dict[str, Any]) -> Optional[str]:
        """Append resource usage sample."""
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

    def upsert_scope_rollup(self, experiment_id: str, scope: Dict[str, Any], bucket: str, values: Dict[str, Any]) -> None:
        """
        Upsert summarized resource metrics per scope/time bucket.
        `bucket` example: '2025-08-10T10:05Z' or '5m_2025-08-10_10:05'
        """
        try:
            scope_n = _scope_min(scope)
            filt = {
                "experiment_id": experiment_id,
                "scope.level": scope_n["level"],
                "scope.entity_id": scope_n["entity_id"],
                "bucket": bucket,
            }
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "scope": scope_n,
                "bucket": bucket,
                "values": values,
            }
            self.collections["rollups"].update_one(filt, {"$set": doc}, upsert=True)
        except Exception as e:
            print(f"❌ Failed to upsert scope rollup: {e}")

    def store_business_metrics(self, experiment_id: str, business_data: Dict[str, Any]) -> Optional[str]:
        """Append business metrics/cost data."""
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
        """Append data split counts and metadata."""
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

    def store_dataset_version(self, experiment_id: str, dataset_name: str, info: Dict[str, Any]) -> Optional[str]:
        """Append dataset version/fingerprint info."""
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
        Store result events (tasker/builder rewards, RL episode, classification, generation, summaries).
        If payload has 'step' and 'scope', we upsert on (exp, type, step, scope).
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
            if step is not None:
                filt = {
                    "experiment_id": experiment_id,
                    "type": kind,
                    "step": step,
                    "scope.level": scope["level"],
                    "scope.entity_id": scope["entity_id"],
                }
                coll.update_one(filt, {"$set": doc}, upsert=True)
                return "upserted"
            else:
                result = coll.insert_one(doc)
                return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to store results: {e}")
            return None

    def store_experiment_summary(self, experiment_id: str, summary: Dict[str, Any]) -> Optional[str]:
        """Upsert final experiment summary (one doc per experiment)."""
        try:
            coll = self.collections["experiments"]
            filt = {"experiment_id": experiment_id, "type": "experiment_summary"}
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                "summary": summary,
                "type": "experiment_summary",
            }
            coll.update_one(filt, {"$set": doc}, upsert=True)
            print(f"📊 Upserted experiment summary: {experiment_id}")
            return "upserted"
        except Exception as e:
            print(f"❌ Failed to store experiment summary: {e}")
            return None

    def store_network_lineage(self, experiment_id: str, network_id: str, lineage_data: Dict[str, Any]) -> Optional[str]:
        """Store or update network lineage information."""
        try:
            doc = {
                "experiment_id": experiment_id,
                "network_id": network_id,
                "timestamp": datetime.now(),
                **lineage_data
            }
            filt = {"experiment_id": experiment_id, "network_id": network_id}
            self.collections["network_lineage"].update_one(filt, {"$set": doc}, upsert=True)
            print(f"🧬 Stored network lineage: {network_id}")
            return "upserted"
        except Exception as e:
            print(f"❌ Failed to store network lineage: {e}")
            return None

    def store_population_snapshot(self, experiment_id: str, snapshot_data: Dict[str, Any]) -> Optional[str]:
        """Store population snapshot."""
        try:
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                **snapshot_data
            }
            result = self.collections["population_history"].insert_one(doc)
            print(f"👥 Stored population snapshot: generation {snapshot_data.get('generation', 'unknown')}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to store population snapshot: {e}")
            return None

    def store_reward_flow(self, experiment_id: str, flow_data: Dict[str, Any]) -> Optional[str]:
        """Store reward flow between networks."""
        try:
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                **flow_data
            }
            result = self.collections["reward_flows"].insert_one(doc)
            print(f"💰 Stored reward flow: {flow_data.get('source_id', '?')} → {flow_data.get('target_id', '?')}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to store reward flow: {e}")
            return None

    def store_challenge_usage(self, experiment_id: str, usage_data: Dict[str, Any]) -> Optional[str]:
        """Store challenge-specific usage rates."""
        try:
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                **usage_data
            }
            result = self.collections["challenge_usage"].insert_one(doc)
            print(f"📈 Stored challenge usage: {usage_data.get('challenge_id', 'unknown')}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to store challenge usage: {e}")
            return None

    def store_learning_relevance(self, experiment_id: str, relevance_data: Dict[str, Any]) -> Optional[str]:
        """Store learning relevance data."""
        try:
            doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(),
                **relevance_data
            }
            result = self.collections["learning_relevance"].insert_one(doc)
            print(f"🎯 Stored learning relevance: {len(relevance_data.get('sample_ids', []))} samples")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to store learning relevance: {e}")
            return None

    def store_component_resource(self, experiment_id: str, component_data: Dict[str, Any]) -> Optional[str]:
        """Store component-level resource usage."""
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
        """Store environment data."""
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

    def get_experiment_data(self, experiment_id: str, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all data across collections for an experiment."""
        try:
            query = {"experiment_id": experiment_id}
            if data_type:
                query["type"] = data_type
            results: List[Dict[str, Any]] = []
            for collection_name, collection in self.collections.items():
                cursor = collection.find(query).sort("timestamp", DESCENDING)
                for doc in cursor:
                    doc["_id"] = str(doc.get("_id", ""))
                    doc["collection"] = collection_name
                    results.append(doc)
            return results
        except Exception as e:
            print(f"❌ Failed to get experiment data: {e}")
            return []

    def get_modification_history(self, experiment_id: str, step_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """Get modification history for an experiment."""
        try:
            query: Dict[str, Any] = {"experiment_id": experiment_id, "type": "modification_tree"}
            if step_range:
                query["modification_data.step"] = {"$gte": step_range[0], "$lte": step_range[1]}
            cursor = self.collections["modifications"].find(query).sort("timestamp", ASCENDING)
            mods: List[Dict[str, Any]] = []
            for doc in cursor:
                doc["_id"] = str(doc.get("_id", ""))
                mods.append(doc)
            return mods
        except Exception as e:
            print(f"❌ Failed to get modification history: {e}")
            return []

    def get_resource_statistics(self, experiment_id: str, component_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get resource usage documents (optionally filtered by component_type)."""
        try:
            query: Dict[str, Any] = {"experiment_id": experiment_id}
            if component_type:
                query["resource_data.component_type"] = component_type
            cursor = self.collections["resources"].find(query).sort("timestamp", ASCENDING)
            out: List[Dict[str, Any]] = []
            for doc in cursor:
                doc["_id"] = str(doc.get("_id", ""))
                out.append(doc)
            return out
        except Exception as e:
            print(f"❌ Failed to get resource statistics: {e}")
            return []

    def get_business_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Aggregate business metrics for an experiment."""
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
                summary.pop("_id", None)
                return summary
            return {"total_cost": 0, "entry_count": 0}
        except Exception as e:
            print(f"❌ Failed to get business summary: {e}")
            return {}

    def aggregate_resource_usage(self, experiment_id: str) -> Dict[str, Any]:
        """Aggregate resource usage statistics across components."""
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
                {"$sort": {"_id": 1}},
            ]
            result = list(self.collections["resources"].aggregate(pipeline))
            aggregated: Dict[str, Any] = {}
            for item in result:
                ctype = item.get("_id") or "unknown"
                aggregated[ctype] = {
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

    def search_experiments(self, query_filter: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Search experiments based on criteria in 'experiments'."""
        try:
            q = query_filter or {}
            cursor = self.collections["experiments"].find(q).limit(limit).sort("timestamp", DESCENDING)
            out: List[Dict[str, Any]] = []
            for doc in cursor:
                doc["_id"] = str(doc.get("_id", ""))
                out.append(doc)
            return out
        except Exception as e:
            print(f"❌ Failed to search experiments: {e}")
            return []

    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            print("🔌 MongoDB connection closed")

    def __enter__(self) -> "MongoBackend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()