"""
PostgreSQL Backend for LeZeA MLOps
==================================

Handles PostgreSQL operations for MLflow metadata storage and structured data:
- MLflow backend store (experiments, runs, metrics, parameters)
- Structured experiment metadata queries
- Performance optimization for large-scale experiments
- Data integrity and backup management
- Advanced analytics and reporting queries

This backend provides optimized PostgreSQL operations with connection pooling,
transaction management, and LeZeA-specific query optimizations.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import RealDictCursor, execute_values
    from psycopg2.errors import OperationalError, DatabaseError
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None


class PostgresBackend:
    """
    PostgreSQL backend for MLflow metadata and structured data storage
    
    This class provides:
    - Connection pooling for optimal performance
    - Transaction management with rollback support
    - Optimized queries for MLflow metadata
    - Advanced analytics and reporting capabilities
    - Data integrity and backup operations
    - Performance monitoring and optimization
    """
    
    def __init__(self, config):
        """
        Initialize PostgreSQL backend
        
        Args:
            config: Configuration object with PostgreSQL settings
        """
        if not POSTGRES_AVAILABLE:
            raise RuntimeError(
                "PostgreSQL adapter is not available. Install with: pip install psycopg2-binary"
            )
        
        self.config = config
        self.postgres_config = config.get_postgres_config()
        
        # Connection pool configuration
        self.connection_pool = None
        self._init_connection_pool()
        
        # Test connection
        self._test_connection()
        
        # Initialize MLflow schema if needed
        self._ensure_mlflow_schema()
        
        print(f"✅ PostgreSQL backend connected: {self.postgres_config['database']}")
    
    def _init_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password'],
                cursor_factory=RealDictCursor
            )
            
        except Exception as e:
            raise ConnectionError(f"Failed to create PostgreSQL connection pool: {e}")
    
    def _test_connection(self):
        """Test PostgreSQL connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()
                    print(f"📊 PostgreSQL version: {version['version'].split(',')[0]}")
                    
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def _ensure_mlflow_schema(self):
        """Ensure MLflow database schema exists"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Check if MLflow tables exist
                    cursor.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('experiments', 'runs', 'metrics', 'params', 'tags')
                    """)
                    
                    existing_tables = [row['table_name'] for row in cursor.fetchall()]
                    
                    if len(existing_tables) < 5:
                        print("⚠️ MLflow tables not found. Run 'mlflow db upgrade' to initialize schema.")
                    else:
                        print("✅ MLflow schema verified")
                        self._create_custom_indexes()
                        
        except Exception as e:
            print(f"⚠️ Could not verify MLflow schema: {e}")
    
    def _create_custom_indexes(self):
        """Create custom indexes for better performance"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Custom indexes for better query performance
                    indexes = [
                        "CREATE INDEX IF NOT EXISTS idx_runs_experiment_start_time ON runs(experiment_id, start_time DESC);",
                        "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);",
                        "CREATE INDEX IF NOT EXISTS idx_metrics_run_key_step ON metrics(run_uuid, key, step);",
                        "CREATE INDEX IF NOT EXISTS idx_params_run_key ON params(run_uuid, key);",
                        "CREATE INDEX IF NOT EXISTS idx_tags_run_key ON tags(run_uuid, key);",
                        "CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);",
                        "CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name);"
                    ]
                    
                    for index_sql in indexes:
                        try:
                            cursor.execute(index_sql)
                        except Exception as e:
                            print(f"⚠️ Could not create index: {e}")
                    
                    print("📊 Custom indexes created")
                    
        except Exception as e:
            print(f"⚠️ Could not create custom indexes: {e}")
    
    def get_experiment_summary(self, experiment_id: str = None, experiment_name: str = None) -> Dict[str, Any]:
        """
        Get comprehensive experiment summary
        
        Args:
            experiment_id: MLflow experiment ID
            experiment_name: Experiment name
        
        Returns:
            Dictionary with experiment summary
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Build query based on provided identifiers
                    if experiment_id:
                        exp_condition = "e.experiment_id = %s"
                        exp_param = experiment_id
                    elif experiment_name:
                        exp_condition = "e.name = %s"
                        exp_param = experiment_name
                    else:
                        raise ValueError("Either experiment_id or experiment_name must be provided")
                    
                    # Get experiment info and run statistics
                    query = f"""
                        SELECT 
                            e.experiment_id,
                            e.name as experiment_name,
                            e.creation_time,
                            e.last_update_time,
                            COUNT(r.run_uuid) as total_runs,
                            COUNT(CASE WHEN r.status = 'FINISHED' THEN 1 END) as completed_runs,
                            COUNT(CASE WHEN r.status = 'FAILED' THEN 1 END) as failed_runs,
                            COUNT(CASE WHEN r.status = 'RUNNING' THEN 1 END) as running_runs,
                            MIN(r.start_time) as first_run_start,
                            MAX(r.end_time) as last_run_end,
                            AVG(CASE WHEN r.end_time IS NOT NULL AND r.start_time IS NOT NULL 
                                THEN r.end_time - r.start_time END) as avg_duration_ms
                        FROM experiments e
                        LEFT JOIN runs r ON e.experiment_id = r.experiment_id
                        WHERE {exp_condition}
                        GROUP BY e.experiment_id, e.name, e.creation_time, e.last_update_time
                    """
                    
                    cursor.execute(query, (exp_param,))
                    result = cursor.fetchone()
                    
                    if not result:
                        return {'error': 'Experiment not found'}
                    
                    # Convert to dictionary and format timestamps
                    summary = dict(result)
                    
                    # Format timestamps
                    if summary['creation_time']:
                        summary['creation_time'] = datetime.fromtimestamp(summary['creation_time'] / 1000).isoformat()
                    if summary['last_update_time']:
                        summary['last_update_time'] = datetime.fromtimestamp(summary['last_update_time'] / 1000).isoformat()
                    if summary['first_run_start']:
                        summary['first_run_start'] = datetime.fromtimestamp(summary['first_run_start'] / 1000).isoformat()
                    if summary['last_run_end']:
                        summary['last_run_end'] = datetime.fromtimestamp(summary['last_run_end'] / 1000).isoformat()
                    
                    # Convert duration to seconds
                    if summary['avg_duration_ms']:
                        summary['avg_duration_seconds'] = summary['avg_duration_ms'] / 1000
                    
                    return summary
                    
        except Exception as e:
            print(f"❌ Failed to get experiment summary: {e}")
            return {'error': str(e)}
    
    def get_run_metrics_history(self, run_id: str, metrics: List[str] = None) -> Dict[str, List]:
        """
        Get metrics history for a specific run
        
        Args:
            run_id: MLflow run ID
            metrics: List of metric names to retrieve (None for all)
        
        Returns:
            Dictionary with metrics history
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Build query with optional metric filter
                    base_query = """
                        SELECT key, value, timestamp, step
                        FROM metrics 
                        WHERE run_uuid = %s
                    """
                    
                    params = [run_id]
                    
                    if metrics:
                        placeholders = ','.join(['%s'] * len(metrics))
                        base_query += f" AND key IN ({placeholders})"
                        params.extend(metrics)
                    
                    base_query += " ORDER BY key, step"
                    
                    cursor.execute(base_query, params)
                    rows = cursor.fetchall()
                    
                    # Group by metric name
                    metrics_history = {}
                    for row in rows:
                        metric_name = row['key']
                        if metric_name not in metrics_history:
                            metrics_history[metric_name] = []
                        
                        metrics_history[metric_name].append({
                            'value': row['value'],
                            'timestamp': datetime.fromtimestamp(row['timestamp'] / 1000).isoformat(),
                            'step': row['step']
                        })
                    
                    return metrics_history
                    
        except Exception as e:
            print(f"❌ Failed to get run metrics history: {e}")
            return {}
    
    def get_experiment_leaderboard(self, experiment_id: str, metric_name: str, 
                                  limit: int = 10, ascending: bool = False) -> List[Dict]:
        """
        Get experiment leaderboard sorted by a specific metric
        
        Args:
            experiment_id: MLflow experiment ID
            metric_name: Metric name to sort by
            limit: Number of top runs to return
            ascending: Sort order (False for descending)
        
        Returns:
            List of top run dictionaries
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    order_direction = "ASC" if ascending else "DESC"
                    
                    query = f"""
                        SELECT 
                            r.run_uuid,
                            r.name as run_name,
                            r.status,
                            r.start_time,
                            r.end_time,
                            m.value as metric_value,
                            (r.end_time - r.start_time) as duration_ms
                        FROM runs r
                        INNER JOIN metrics m ON r.run_uuid = m.run_uuid
                        WHERE r.experiment_id = %s 
                        AND m.key = %s
                        AND r.status = 'FINISHED'
                        ORDER BY m.value {order_direction}
                        LIMIT %s
                    """
                    
                    cursor.execute(query, (experiment_id, metric_name, limit))
                    rows = cursor.fetchall()
                    
                    leaderboard = []
                    for i, row in enumerate(rows, 1):
                        entry = dict(row)
                        entry['rank'] = i
                        entry['metric_name'] = metric_name
                        
                        # Format timestamps
                        if entry['start_time']:
                            entry['start_time'] = datetime.fromtimestamp(entry['start_time'] / 1000).isoformat()
                        if entry['end_time']:
                            entry['end_time'] = datetime.fromtimestamp(entry['end_time'] / 1000).isoformat()
                        
                        # Convert duration to seconds
                        if entry['duration_ms']:
                            entry['duration_seconds'] = entry['duration_ms'] / 1000
                        
                        leaderboard.append(entry)
                    
                    return leaderboard
                    
        except Exception as e:
            print(f"❌ Failed to get experiment leaderboard: {e}")
            return []
    
    def search_runs_advanced(self, experiment_ids: List[str] = None, 
                           filter_conditions: Dict[str, Any] = None,
                           order_by: str = None, limit: int = 100) -> List[Dict]:
        """
        Advanced run search with complex filtering
        
        Args:
            experiment_ids: List of experiment IDs to search
            filter_conditions: Dictionary of filter conditions
            order_by: Column to order by
            limit: Maximum number of results
        
        Returns:
            List of run dictionaries
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Build base query
                    query = """
                        SELECT DISTINCT
                            r.run_uuid,
                            r.experiment_id,
                            r.name as run_name,
                            r.status,
                            r.start_time,
                            r.end_time,
                            r.artifact_uri
                        FROM runs r
                    """
                    
                    joins = []
                    where_conditions = []
                    params = []
                    
                    # Add experiment filter
                    if experiment_ids:
                        placeholders = ','.join(['%s'] * len(experiment_ids))
                        where_conditions.append(f"r.experiment_id IN ({placeholders})")
                        params.extend(experiment_ids)
                    
                    # Add filter conditions
                    if filter_conditions:
                        # Metric filters
                        if 'metrics' in filter_conditions:
                            for i, (metric_name, operator, value) in enumerate(filter_conditions['metrics']):
                                alias = f"m{i}"
                                joins.append(f"LEFT JOIN metrics {alias} ON r.run_uuid = {alias}.run_uuid")
                                where_conditions.append(f"{alias}.key = %s AND {alias}.value {operator} %s")
                                params.extend([metric_name, value])
                        
                        # Parameter filters
                        if 'params' in filter_conditions:
                            for i, (param_name, operator, value) in enumerate(filter_conditions['params']):
                                alias = f"p{i}"
                                joins.append(f"LEFT JOIN params {alias} ON r.run_uuid = {alias}.run_uuid")
                                where_conditions.append(f"{alias}.key = %s AND {alias}.value {operator} %s")
                                params.extend([param_name, value])
                        
                        # Tag filters
                        if 'tags' in filter_conditions:
                            for i, (tag_name, operator, value) in enumerate(filter_conditions['tags']):
                                alias = f"t{i}"
                                joins.append(f"LEFT JOIN tags {alias} ON r.run_uuid = {alias}.run_uuid")
                                where_conditions.append(f"{alias}.key = %s AND {alias}.value {operator} %s")
                                params.extend([tag_name, value])
                        
                        # Status filter
                        if 'status' in filter_conditions:
                            where_conditions.append("r.status = %s")
                            params.append(filter_conditions['status'])
                        
                        # Date range filter
                        if 'start_time_after' in filter_conditions:
                            where_conditions.append("r.start_time >= %s")
                            params.append(int(filter_conditions['start_time_after'].timestamp() * 1000))
                        
                        if 'start_time_before' in filter_conditions:
                            where_conditions.append("r.start_time <= %s")
                            params.append(int(filter_conditions['start_time_before'].timestamp() * 1000))
                    
                    # Combine query parts
                    if joins:
                        query += " " + " ".join(joins)
                    
                    if where_conditions:
                        query += " WHERE " + " AND ".join(where_conditions)
                    
                    # Add ordering
                    if order_by:
                        query += f" ORDER BY r.{order_by} DESC"
                    else:
                        query += " ORDER BY r.start_time DESC"
                    
                    # Add limit
                    query += f" LIMIT {limit}"
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Format results
                    runs = []
                    for row in rows:
                        run_data = dict(row)
                        
                        # Format timestamps
                        if run_data['start_time']:
                            run_data['start_time'] = datetime.fromtimestamp(run_data['start_time'] / 1000).isoformat()
                        if run_data['end_time']:
                            run_data['end_time'] = datetime.fromtimestamp(run_data['end_time'] / 1000).isoformat()
                        
                        runs.append(run_data)
                    
                    return runs
                    
        except Exception as e:
            print(f"❌ Failed to search runs: {e}")
            return []
    
    def get_metric_statistics(self, experiment_id: str, metric_name: str) -> Dict[str, Any]:
        """
        Get statistical summary for a metric across all runs in an experiment
        
        Args:
            experiment_id: MLflow experiment ID
            metric_name: Name of the metric
        
        Returns:
            Dictionary with metric statistics
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = """
                        SELECT 
                            COUNT(*) as run_count,
                            AVG(m.value) as mean_value,
                            MIN(m.value) as min_value,
                            MAX(m.value) as max_value,
                            STDDEV(m.value) as std_value,
                            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY m.value) as q25,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY m.value) as median,
                            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY m.value) as q75
                        FROM runs r
                        INNER JOIN metrics m ON r.run_uuid = m.run_uuid
                        WHERE r.experiment_id = %s 
                        AND m.key = %s
                        AND r.status = 'FINISHED'
                    """
                    
                    cursor.execute(query, (experiment_id, metric_name))
                    result = cursor.fetchone()
                    
                    if result and result['run_count'] > 0:
                        stats = dict(result)
                        stats['metric_name'] = metric_name
                        stats['experiment_id'] = experiment_id
                        return stats
                    else:
                        return {'error': 'No data found for metric'}
                    
        except Exception as e:
            print(f"❌ Failed to get metric statistics: {e}")
            return {'error': str(e)}
    
    def get_parameter_analysis(self, experiment_id: str) -> Dict[str, Any]:
        """
        Analyze parameter usage across all runs in an experiment
        
        Args:
            experiment_id: MLflow experiment ID
        
        Returns:
            Dictionary with parameter analysis
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get parameter usage statistics
                    query = """
                        SELECT 
                            p.key as param_name,
                            COUNT(*) as usage_count,
                            COUNT(DISTINCT p.value) as unique_values,
                            array_agg(DISTINCT p.value ORDER BY p.value) as all_values
                        FROM runs r
                        INNER JOIN params p ON r.run_uuid = p.run_uuid
                        WHERE r.experiment_id = %s
                        GROUP BY p.key
                        ORDER BY usage_count DESC
                    """
                    
                    cursor.execute(query, (experiment_id,))
                    rows = cursor.fetchall()
                    
                    parameter_analysis = {
                        'experiment_id': experiment_id,
                        'total_parameters': len(rows),
                        'parameters': []
                    }
                    
                    for row in rows:
                        param_info = dict(row)
                        
                        # Analyze parameter type
                        sample_values = param_info['all_values'][:5]  # First 5 values
                        param_info['sample_values'] = sample_values
                        param_info['parameter_type'] = self._infer_parameter_type(sample_values)
                        
                        parameter_analysis['parameters'].append(param_info)
                    
                    return parameter_analysis
                    
        except Exception as e:
            print(f"❌ Failed to get parameter analysis: {e}")
            return {'error': str(e)}
    
    def _infer_parameter_type(self, values: List[str]) -> str:
        """Infer parameter type from sample values"""
        if not values:
            return 'unknown'
        
        # Check if all values are numeric
        try:
            [float(v) for v in values]
            return 'numeric'
        except ValueError:
            pass
        
        # Check if all values are boolean-like
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
        if all(v.lower() in boolean_values for v in values):
            return 'boolean'
        
        # Check if values look like categorical
        if len(set(values)) < len(values) or len(values) <= 5:
            return 'categorical'
        
        return 'text'
    
    def cleanup_old_runs(self, experiment_id: str, keep_runs: int = 100) -> int:
        """
        Clean up old runs keeping only the most recent ones
        
        Args:
            experiment_id: MLflow experiment ID
            keep_runs: Number of runs to keep
        
        Returns:
            Number of runs deleted
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get old run IDs to delete
                    cursor.execute("""
                        SELECT run_uuid
                        FROM runs
                        WHERE experiment_id = %s
                        ORDER BY start_time DESC
                        OFFSET %s
                    """, (experiment_id, keep_runs))
                    
                    old_runs = [row['run_uuid'] for row in cursor.fetchall()]
                    
                    if not old_runs:
                        return 0
                    
                    # Delete old runs and related data
                    placeholders = ','.join(['%s'] * len(old_runs))
                    
                    # Delete metrics
                    cursor.execute(f"DELETE FROM metrics WHERE run_uuid IN ({placeholders})", old_runs)
                    metrics_deleted = cursor.rowcount
                    
                    # Delete params
                    cursor.execute(f"DELETE FROM params WHERE run_uuid IN ({placeholders})", old_runs)
                    params_deleted = cursor.rowcount
                    
                    # Delete tags
                    cursor.execute(f"DELETE FROM tags WHERE run_uuid IN ({placeholders})", old_runs)
                    tags_deleted = cursor.rowcount
                    
                    # Delete runs
                    cursor.execute(f"DELETE FROM runs WHERE run_uuid IN ({placeholders})", old_runs)
                    runs_deleted = cursor.rowcount
                    
                    print(f"🧹 Cleaned up {runs_deleted} old runs ({metrics_deleted} metrics, {params_deleted} params, {tags_deleted} tags)")
                    return runs_deleted
                    
        except Exception as e:
            print(f"❌ Failed to cleanup old runs: {e}")
            return 0
    
    def backup_experiment(self, experiment_id: str, backup_file: str):
        """
        Create a backup of experiment data
        
        Args:
            experiment_id: MLflow experiment ID
            backup_file: Path to save backup
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    backup_data = {
                        'backup_timestamp': datetime.now().isoformat(),
                        'experiment_id': experiment_id,
                        'tables': {}
                    }
                    
                    # Backup experiment info
                    cursor.execute("SELECT * FROM experiments WHERE experiment_id = %s", (experiment_id,))
                    backup_data['tables']['experiments'] = [dict(row) for row in cursor.fetchall()]
                    
                    # Backup runs
                    cursor.execute("SELECT * FROM runs WHERE experiment_id = %s", (experiment_id,))
                    runs = [dict(row) for row in cursor.fetchall()]
                    backup_data['tables']['runs'] = runs
                    
                    run_ids = [run['run_uuid'] for run in runs]
                    
                    if run_ids:
                        placeholders = ','.join(['%s'] * len(run_ids))
                        
                        # Backup metrics
                        cursor.execute(f"SELECT * FROM metrics WHERE run_uuid IN ({placeholders})", run_ids)
                        backup_data['tables']['metrics'] = [dict(row) for row in cursor.fetchall()]
                        
                        # Backup params
                        cursor.execute(f"SELECT * FROM params WHERE run_uuid IN ({placeholders})", run_ids)
                        backup_data['tables']['params'] = [dict(row) for row in cursor.fetchall()]
                        
                        # Backup tags
                        cursor.execute(f"SELECT * FROM tags WHERE run_uuid IN ({placeholders})", run_ids)
                        backup_data['tables']['tags'] = [dict(row) for row in cursor.fetchall()]
                    
                    # Save backup
                    with open(backup_file, 'w') as f:
                        json.dump(backup_data, f, indent=2, default=str)
                    
                    total_records = sum(len(table_data) for table_data in backup_data['tables'].values())
                    print(f"💾 Created backup with {total_records} records: {backup_file}")
                    
        except Exception as e:
            print(f"❌ Failed to backup experiment: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics and health information
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    stats = {}
                    
                    # Table sizes
                    cursor.execute("""
                        SELECT 
                            schemaname,
                            tablename,
                            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                            pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                        FROM pg_tables 
                        WHERE schemaname = 'public' 
                        AND tablename IN ('experiments', 'runs', 'metrics', 'params', 'tags')
                        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                    """)
                    
                    stats['table_sizes'] = [dict(row) for row in cursor.fetchall()]
                    
                    # Record counts
                    tables = ['experiments', 'runs', 'metrics', 'params', 'tags']
                    stats['record_counts'] = {}
                    
                    for table in tables:
                        cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                        stats['record_counts'][table] = cursor.fetchone()['count']
                    
                    # Database size
                    cursor.execute("""
                        SELECT pg_size_pretty(pg_database_size(current_database())) as database_size,
                               pg_database_size(current_database()) as database_size_bytes
                    """)
                    db_size = cursor.fetchone()
                    stats['database_size'] = dict(db_size)
                    
                    # Connection info
                    cursor.execute("SELECT COUNT(*) as active_connections FROM pg_stat_activity WHERE datname = current_database()")
                    stats['active_connections'] = cursor.fetchone()['active_connections']
                    
                    return stats
                    
        except Exception as e:
            print(f"❌ Failed to get database stats: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            print("🔌 PostgreSQL connection pool closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()