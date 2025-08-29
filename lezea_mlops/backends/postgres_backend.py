"""
PostgreSQL Backend for LeZeA MLOps System

This module provides a comprehensive PostgreSQL backend for MLflow metadata storage
and structured data operations. It handles connection pooling, transaction management,
advanced analytics, and database optimization for large-scale machine learning experiments.

Features:
    - MLflow backend store (experiments, runs, metrics, parameters)  
    - Structured experiment metadata queries and analytics
    - Performance optimization with connection pooling and custom indexes
    - Data integrity management and backup operations
    - Advanced reporting and statistical analysis capabilities
    - Database health monitoring and maintenance operations


"""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# PostgreSQL dependencies with graceful fallback
try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.errors import DatabaseError, OperationalError
    from psycopg2.extras import RealDictCursor, execute_values
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None  # type: ignore


class PostgresBackend:
    """
    Professional PostgreSQL backend for MLflow metadata and structured data storage.
    
    This class provides a robust, production-ready interface for PostgreSQL operations
    with connection pooling, transaction management, and advanced query capabilities.
    
    Key Features:
        - Thread-safe connection pooling for optimal performance
        - Automatic transaction management with rollback support
        - Optimized queries for MLflow metadata operations
        - Advanced analytics and reporting capabilities
        - Data integrity and backup operations
        - Performance monitoring and database optimization
        - Custom indexing for improved query performance
        
    Attributes:
        config: Configuration object containing PostgreSQL settings
        postgres_config: Parsed PostgreSQL configuration dictionary
        connection_pool: Thread-safe connection pool instance
    """

    def __init__(self, config):
        """
        Initialize PostgreSQL backend with connection pooling and schema validation.
        
        Args:
            config: Configuration object with get_postgres_config() method
            
        Raises:
            RuntimeError: If PostgreSQL adapter is not available
            ConnectionError: If database connection or pool creation fails
        """
        if not POSTGRES_AVAILABLE:
            raise RuntimeError(
                "PostgreSQL adapter is not available. Install with: pip install psycopg2-binary"
            )

        self.config = config
        self.postgres_config = config.get_postgres_config()
        self.connection_pool = None

        # Initialize connection infrastructure
        self._init_connection_pool()
        self._test_connection()
        self._ensure_mlflow_schema()

        print(f"PostgreSQL backend connected: {self.postgres_config['database']}")

    def _init_connection_pool(self):
        """
        Initialize PostgreSQL connection pool with production-ready settings.
        
        Creates a threaded connection pool with configurable min/max connections
        and proper cursor factory for dictionary-based results.
        
        Raises:
            ConnectionError: If connection pool creation fails
        """
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,                                    # Minimum connections
                maxconn=20,                                   # Maximum connections
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password'],
                cursor_factory=RealDictCursor               # Enable dict-like results
            )

        except Exception as e:
            raise ConnectionError(f"Failed to create PostgreSQL connection pool: {e}")

    def _test_connection(self):
        """
        Test PostgreSQL connection and retrieve version information.
        
        Performs a simple query to verify connectivity and logs database version
        for debugging and compatibility verification.
        
        Raises:
            ConnectionError: If connection test fails
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version();")
                    version_info = cursor.fetchone()
                    
                    # Extract major version for logging
                    version_str = version_info['version'].split(',')[0]
                    print(f"PostgreSQL version: {version_str}")

        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections with automatic transaction handling.
        
        Provides connection from pool with automatic commit on success and
        rollback on exception. Ensures proper connection cleanup.
        
        Yields:
            psycopg2.connection: Database connection with RealDictCursor
            
        Raises:
            Exception: Re-raises any database operation exceptions after rollback
        """
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
        """
        Verify MLflow database schema exists and create performance indexes.
        
        Checks for required MLflow tables and creates custom indexes for
        improved query performance. Provides guidance if schema is missing.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Check for essential MLflow tables
                    cursor.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('experiments', 'runs', 'metrics', 'params', 'tags')
                    """)

                    existing_tables = [row['table_name'] for row in cursor.fetchall()]

                    if len(existing_tables) < 5:
                        print("MLflow tables not found. Run 'mlflow db upgrade' to initialize schema.")
                    else:
                        print("MLflow schema verified")
                        self._create_custom_indexes()

        except Exception as e:
            print(f"Could not verify MLflow schema: {e}")

    def _create_custom_indexes(self):
        """
        Create custom database indexes for optimized query performance.
        
        Adds indexes specifically designed for common MLflow query patterns
        including time-based filtering, experiment grouping, and key-based lookups.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Performance-optimized indexes for common query patterns
                    performance_indexes = [
                        "CREATE INDEX IF NOT EXISTS idx_runs_experiment_start_time ON runs(experiment_id, start_time DESC);",
                        "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);",
                        "CREATE INDEX IF NOT EXISTS idx_metrics_run_key_step ON metrics(run_uuid, key, step);",
                        "CREATE INDEX IF NOT EXISTS idx_params_run_key ON params(run_uuid, key);",
                        "CREATE INDEX IF NOT EXISTS idx_tags_run_key ON tags(run_uuid, key);",
                        "CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);",
                        "CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name);"
                    ]

                    created_count = 0
                    for index_sql in performance_indexes:
                        try:
                            cursor.execute(index_sql)
                            created_count += 1
                        except Exception as e:
                            print(f"Could not create index: {e}")

                    print(f"Custom indexes created: {created_count}/{len(performance_indexes)}")

        except Exception as e:
            print(f"Could not create custom indexes: {e}")

    # Experiment Analysis and Reporting Methods

    def get_experiment_summary(self, experiment_id: str = None, experiment_name: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive experiment summary with run statistics.
        
        Provides detailed analytics including run counts by status, timing information,
        and performance metrics for experiment evaluation.
        
        Args:
            experiment_id: MLflow experiment ID for lookup
            experiment_name: Alternative experiment name for lookup
            
        Returns:
            Dictionary containing experiment summary and statistics
            
        Raises:
            ValueError: If neither experiment_id nor experiment_name provided
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Determine query condition based on provided identifier
                    if experiment_id:
                        exp_condition = "e.experiment_id = %s"
                        exp_param = experiment_id
                    elif experiment_name:
                        exp_condition = "e.name = %s"
                        exp_param = experiment_name
                    else:
                        raise ValueError("Either experiment_id or experiment_name must be provided")

                    # Comprehensive experiment analytics query
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

                    # Format response with proper timestamp conversion
                    summary = dict(result)

                    # Convert MLflow timestamps (milliseconds) to ISO format
                    timestamp_fields = ['creation_time', 'last_update_time', 'first_run_start', 'last_run_end']
                    for field in timestamp_fields:
                        if summary[field]:
                            summary[field] = datetime.fromtimestamp(summary[field] / 1000).isoformat()

                    # Convert duration from milliseconds to seconds
                    if summary['avg_duration_ms']:
                        summary['avg_duration_seconds'] = summary['avg_duration_ms'] / 1000

                    return summary

        except Exception as e:
            print(f"Failed to get experiment summary: {e}")
            return {'error': str(e)}

    def get_run_metrics_history(self, run_id: str, metrics: List[str] = None) -> Dict[str, List]:
        """
        Retrieve time-series metrics history for a specific run.
        
        Returns complete metrics evolution over time with optional filtering
        by metric names for focused analysis.
        
        Args:
            run_id: MLflow run UUID
            metrics: Optional list of metric names to filter (None for all)
            
        Returns:
            Dictionary with metric names as keys and time-series data as values
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Build dynamic query with optional metric filtering
                    base_query = """
                        SELECT key, value, timestamp, step
                        FROM metrics 
                        WHERE run_uuid = %s
                    """

                    query_params = [run_id]

                    # Add metric name filter if specified
                    if metrics:
                        placeholders = ','.join(['%s'] * len(metrics))
                        base_query += f" AND key IN ({placeholders})"
                        query_params.extend(metrics)

                    base_query += " ORDER BY key, step"

                    cursor.execute(base_query, query_params)
                    rows = cursor.fetchall()

                    # Group metrics by name for time-series structure
                    metrics_history = {}
                    for row in rows:
                        metric_name = row['key']
                        if metric_name not in metrics_history:
                            metrics_history[metric_name] = []

                        # Format timestamp and add to series
                        metrics_history[metric_name].append({
                            'value': row['value'],
                            'timestamp': datetime.fromtimestamp(row['timestamp'] / 1000).isoformat(),
                            'step': row['step']
                        })

                    return metrics_history

        except Exception as e:
            print(f"Failed to get run metrics history: {e}")
            return {}

    def get_experiment_leaderboard(self, experiment_id: str, metric_name: str,
                                  limit: int = 10, ascending: bool = False) -> List[Dict]:
        """
        Generate experiment leaderboard ranked by specified metric.
        
        Creates a ranked list of top-performing runs based on a target metric
        with comprehensive run metadata for performance analysis.
        
        Args:
            experiment_id: MLflow experiment ID
            metric_name: Target metric for ranking
            limit: Number of top runs to return
            ascending: Sort order (False for descending/higher is better)
            
        Returns:
            List of ranked run dictionaries with performance metadata
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    sort_direction = "ASC" if ascending else "DESC"

                    # Comprehensive leaderboard query with timing data
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
                        ORDER BY m.value {sort_direction}
                        LIMIT %s
                    """

                    cursor.execute(query, (experiment_id, metric_name, limit))
                    rows = cursor.fetchall()

                    # Format leaderboard with rankings and metadata
                    leaderboard = []
                    for rank, row in enumerate(rows, 1):
                        entry = dict(row)
                        entry['rank'] = rank
                        entry['metric_name'] = metric_name

                        # Convert timestamps from MLflow format
                        if entry['start_time']:
                            entry['start_time'] = datetime.fromtimestamp(entry['start_time'] / 1000).isoformat()
                        if entry['end_time']:
                            entry['end_time'] = datetime.fromtimestamp(entry['end_time'] / 1000).isoformat()

                        # Convert duration to seconds for readability
                        if entry['duration_ms']:
                            entry['duration_seconds'] = entry['duration_ms'] / 1000

                        leaderboard.append(entry)

                    return leaderboard

        except Exception as e:
            print(f"Failed to get experiment leaderboard: {e}")
            return []

    def search_runs_advanced(self, experiment_ids: List[str] = None,
                           filter_conditions: Dict[str, Any] = None,
                           order_by: str = None, limit: int = 100) -> List[Dict]:
        """
        Execute advanced run search with complex multi-table filtering.
        
        Supports sophisticated filtering across metrics, parameters, tags,
        and run metadata with dynamic query construction.
        
        Args:
            experiment_ids: List of experiment IDs to search within
            filter_conditions: Complex filter dictionary supporting:
                - metrics: List of (name, operator, value) tuples
                - params: List of (name, operator, value) tuples  
                - tags: List of (name, operator, value) tuples
                - status: Run status filter
                - start_time_after/before: Date range filters
            order_by: Column name for result ordering
            limit: Maximum number of results
            
        Returns:
            List of run dictionaries matching filter criteria
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Dynamic query construction
                    base_query = """
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
                    query_params = []

                    # Add experiment ID filtering
                    if experiment_ids:
                        placeholders = ','.join(['%s'] * len(experiment_ids))
                        where_conditions.append(f"r.experiment_id IN ({placeholders})")
                        query_params.extend(experiment_ids)

                    # Process complex filter conditions
                    if filter_conditions:
                        self._build_filter_conditions(
                            filter_conditions, joins, where_conditions, query_params
                        )

                    # Assemble complete query
                    complete_query = base_query
                    if joins:
                        complete_query += " " + " ".join(joins)
                    if where_conditions:
                        complete_query += " WHERE " + " AND ".join(where_conditions)

                    # Add ordering and limit
                    if order_by:
                        complete_query += f" ORDER BY r.{order_by} DESC"
                    else:
                        complete_query += " ORDER BY r.start_time DESC"
                    complete_query += f" LIMIT {limit}"

                    cursor.execute(complete_query, query_params)
                    rows = cursor.fetchall()

                    # Format results with proper timestamp conversion
                    runs = []
                    for row in rows:
                        run_data = dict(row)

                        # Convert MLflow timestamps
                        if run_data['start_time']:
                            run_data['start_time'] = datetime.fromtimestamp(run_data['start_time'] / 1000).isoformat()
                        if run_data['end_time']:
                            run_data['end_time'] = datetime.fromtimestamp(run_data['end_time'] / 1000).isoformat()

                        runs.append(run_data)

                    return runs

        except Exception as e:
            print(f"Failed to search runs: {e}")
            return []

    def _build_filter_conditions(self, filter_conditions: Dict[str, Any], 
                               joins: List[str], where_conditions: List[str], 
                               query_params: List[Any]):
        """
        Build dynamic SQL filter conditions from filter specification.
        
        Helper method for constructing complex WHERE clauses and JOINs
        based on filter_conditions dictionary.
        
        Args:
            filter_conditions: Filter specification dictionary
            joins: List to append JOIN clauses to
            where_conditions: List to append WHERE conditions to
            query_params: List to append query parameters to
        """
        # Metric-based filtering
        if 'metrics' in filter_conditions:
            for i, (metric_name, operator, value) in enumerate(filter_conditions['metrics']):
                alias = f"m{i}"
                joins.append(f"LEFT JOIN metrics {alias} ON r.run_uuid = {alias}.run_uuid")
                where_conditions.append(f"{alias}.key = %s AND {alias}.value {operator} %s")
                query_params.extend([metric_name, value])

        # Parameter-based filtering
        if 'params' in filter_conditions:
            for i, (param_name, operator, value) in enumerate(filter_conditions['params']):
                alias = f"p{i}"
                joins.append(f"LEFT JOIN params {alias} ON r.run_uuid = {alias}.run_uuid")
                where_conditions.append(f"{alias}.key = %s AND {alias}.value {operator} %s")
                query_params.extend([param_name, value])

        # Tag-based filtering
        if 'tags' in filter_conditions:
            for i, (tag_name, operator, value) in enumerate(filter_conditions['tags']):
                alias = f"t{i}"
                joins.append(f"LEFT JOIN tags {alias} ON r.run_uuid = {alias}.run_uuid")
                where_conditions.append(f"{alias}.key = %s AND {alias}.value {operator} %s")
                query_params.extend([tag_name, value])

        # Status filtering
        if 'status' in filter_conditions:
            where_conditions.append("r.status = %s")
            query_params.append(filter_conditions['status'])

        # Date range filtering
        if 'start_time_after' in filter_conditions:
            timestamp_ms = int(filter_conditions['start_time_after'].timestamp() * 1000)
            where_conditions.append("r.start_time >= %s")
            query_params.append(timestamp_ms)

        if 'start_time_before' in filter_conditions:
            timestamp_ms = int(filter_conditions['start_time_before'].timestamp() * 1000)
            where_conditions.append("r.start_time <= %s")
            query_params.append(timestamp_ms)

    # Statistical Analysis Methods

    def get_metric_statistics(self, experiment_id: str, metric_name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive statistical summary for experiment metrics.
        
        Computes descriptive statistics including mean, standard deviation,
        quartiles, and distribution characteristics for metric analysis.
        
        Args:
            experiment_id: MLflow experiment ID
            metric_name: Target metric for analysis
            
        Returns:
            Dictionary with statistical measures and distribution data
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Comprehensive statistical analysis query
                    statistical_query = """
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

                    cursor.execute(statistical_query, (experiment_id, metric_name))
                    result = cursor.fetchone()

                    if result and result['run_count'] > 0:
                        stats = dict(result)
                        stats['metric_name'] = metric_name
                        stats['experiment_id'] = experiment_id
                        return stats
                    else:
                        return {'error': 'No data found for metric'}

        except Exception as e:
            print(f"Failed to get metric statistics: {e}")
            return {'error': str(e)}

    def get_parameter_analysis(self, experiment_id: str) -> Dict[str, Any]:
        """
        Analyze parameter usage patterns across experiment runs.
        
        Examines parameter distribution, uniqueness, and types to provide
        insights into hyperparameter exploration patterns.
        
        Args:
            experiment_id: MLflow experiment ID
            
        Returns:
            Dictionary with parameter usage analysis and type inference
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Parameter usage analysis query
                    analysis_query = """
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

                    cursor.execute(analysis_query, (experiment_id,))
                    rows = cursor.fetchall()

                    # Build comprehensive parameter analysis
                    analysis_result = {
                        'experiment_id': experiment_id,
                        'total_parameters': len(rows),
                        'parameters': []
                    }

                    for row in rows:
                        param_info = dict(row)

                        # Analyze parameter characteristics
                        sample_values = param_info['all_values'][:5]  # Sample for type inference
                        param_info['sample_values'] = sample_values
                        param_info['parameter_type'] = self._infer_parameter_type(sample_values)

                        analysis_result['parameters'].append(param_info)

                    return analysis_result

        except Exception as e:
            print(f"Failed to get parameter analysis: {e}")
            return {'error': str(e)}

    def _infer_parameter_type(self, values: List[str]) -> str:
        """
        Infer parameter data type from sample values.
        
        Analyzes sample parameter values to determine likely data type
        for better understanding of hyperparameter space.
        
        Args:
            values: List of sample parameter values as strings
            
        Returns:
            Inferred parameter type: 'numeric', 'boolean', 'categorical', 'text', or 'unknown'
        """
        if not values:
            return 'unknown'

        # Test for numeric type
        try:
            [float(v) for v in values]
            return 'numeric'
        except (ValueError, TypeError):
            pass

        # Test for boolean-like values
        boolean_indicators = {'true', 'false', '1', '0', 'yes', 'no'}
        if all(v.lower() in boolean_indicators for v in values):
            return 'boolean'

        # Test for categorical (limited unique values)
        if len(set(values)) < len(values) or len(values) <= 5:
            return 'categorical'

        return 'text'

    # Database Management Methods

    def cleanup_old_runs(self, experiment_id: str, keep_runs: int = 100) -> int:
        """
        Clean up old experiment runs while preserving recent ones.
        
        Removes older runs and associated data (metrics, parameters, tags)
        based on start time, keeping only the most recent runs.
        
        Args:
            experiment_id: MLflow experiment ID
            keep_runs: Number of recent runs to preserve
            
        Returns:
            Number of runs successfully deleted
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Identify old runs to delete
                    cursor.execute("""
                        SELECT run_uuid
                        FROM runs
                        WHERE experiment_id = %s
                        ORDER BY start_time DESC
                        OFFSET %s
                    """, (experiment_id, keep_runs))

                    old_run_ids = [row['run_uuid'] for row in cursor.fetchall()]

                    if not old_run_ids:
                        return 0

                    # Execute cascading deletion
                    placeholders = ','.join(['%s'] * len(old_run_ids))

                    # Delete associated data first (foreign key constraints)
                    cursor.execute(f"DELETE FROM metrics WHERE run_uuid IN ({placeholders})", old_run_ids)
                    metrics_deleted = cursor.rowcount

                    cursor.execute(f"DELETE FROM params WHERE run_uuid IN ({placeholders})", old_run_ids)
                    params_deleted = cursor.rowcount

                    cursor.execute(f"DELETE FROM tags WHERE run_uuid IN ({placeholders})", old_run_ids)
                    tags_deleted = cursor.rowcount

                    # Delete run records
                    cursor.execute(f"DELETE FROM runs WHERE run_uuid IN ({placeholders})", old_run_ids)
                    runs_deleted = cursor.rowcount

                    print(f"Cleaned up {runs_deleted} old runs ({metrics_deleted} metrics, {params_deleted} params, {tags_deleted} tags)")
                    return runs_deleted

        except Exception as e:
            print(f"Failed to cleanup old runs: {e}")
            return 0

    def backup_experiment(self, experiment_id: str, backup_file: str):
        """
        Create comprehensive backup of experiment data.
        
        Exports all experiment-related data including runs, metrics,
        parameters, and tags to JSON format for backup or migration.
        
        Args:
            experiment_id: MLflow experiment ID to backup
            backup_file: File path for backup output
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Initialize backup structure
                    backup_data = {
                        'backup_timestamp': datetime.now().isoformat(),
                        'experiment_id': experiment_id,
                        'tables': {}
                    }

                    # Backup experiment metadata
                    cursor.execute("SELECT * FROM experiments WHERE experiment_id = %s", (experiment_id,))
                    backup_data['tables']['experiments'] = [dict(row) for row in cursor.fetchall()]

                    # Backup all runs
                    cursor.execute("SELECT * FROM runs WHERE experiment_id = %s", (experiment_id,))
                    runs = [dict(row) for row in cursor.fetchall()]
                    backup_data['tables']['runs'] = runs

                    # Backup related data if runs exist
                    run_ids = [run['run_uuid'] for run in runs]
                    if run_ids:
                        self._backup_run_data(cursor, run_ids, backup_data)

                    # Write backup to file
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(backup_data, f, indent=2, default=str)

                    total_records = sum(len(table_data) for table_data in backup_data['tables'].values())
                    print(f"Created backup with {total_records} records: {backup_file}")

        except Exception as e:
            print(f"Failed to backup experiment: {e}")

    def _backup_run_data(self, cursor, run_ids: List[str], backup_data: Dict):
        """
        Backup run-related data (metrics, parameters, tags).
        
        Helper method for comprehensive run data backup including
        all associated metadata and measurements.
        
        Args:
            cursor: Database cursor for queries
            run_ids: List of run UUIDs to backup
            backup_data: Backup structure to populate
        """
        placeholders = ','.join(['%s'] * len(run_ids))

        # Backup metrics
        cursor.execute(f"SELECT * FROM metrics WHERE run_uuid IN ({placeholders})", run_ids)
        backup_data['tables']['metrics'] = [dict(row) for row in cursor.fetchall()]

        # Backup parameters
        cursor.execute(f"SELECT * FROM params WHERE run_uuid IN ({placeholders})", run_ids)
        backup_data['tables']['params'] = [dict(row) for row in cursor.fetchall()]

        # Backup tags
        cursor.execute(f"SELECT * FROM tags WHERE run_uuid IN ({placeholders})", run_ids)
        backup_data['tables']['tags'] = [dict(row) for row in cursor.fetchall()]

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Generate comprehensive database statistics and health information.
        
        Provides detailed metrics about database size, table statistics,
        connection counts, and overall system health for monitoring.
        
        Returns:
            Dictionary containing database statistics and health metrics
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    database_stats = {}

                    # Analyze table sizes and storage usage
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

                    database_stats['table_sizes'] = [dict(row) for row in cursor.fetchall()]

                    # Generate record count statistics
                    mlflow_tables = ['experiments', 'runs', 'metrics', 'params', 'tags']
                    database_stats['record_counts'] = {}

                    for table_name in mlflow_tables:
                        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                        result = cursor.fetchone()
                        database_stats['record_counts'][table_name] = result['count']

                    # Calculate overall database size
                    cursor.execute("""
                        SELECT pg_size_pretty(pg_database_size(current_database())) as database_size,
                               pg_database_size(current_database()) as database_size_bytes
                    """)
                    size_info = cursor.fetchone()
                    database_stats['database_size'] = dict(size_info)

                    # Monitor active connections
                    cursor.execute("""
                        SELECT COUNT(*) as active_connections 
                        FROM pg_stat_activity 
                        WHERE datname = current_database()
                    """)
                    connection_info = cursor.fetchone()
                    database_stats['active_connections'] = connection_info['active_connections']

                    return database_stats

        except Exception as e:
            print(f"Failed to get database stats: {e}")
            return {'error': str(e)}

    def ping(self) -> bool:
        """
        Test database connectivity and responsiveness.
        
        Returns:
            True if database is accessible and responsive, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return cursor.fetchone() is not None
        except Exception:
            return False

    def execute_custom_query(self, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """
        Execute custom SQL query with parameter binding.
        
        Provides interface for executing arbitrary SQL queries with
        proper parameter binding for security and flexibility.
        
        Args:
            query: SQL query string with parameter placeholders
            params: Optional list of parameters for query binding
            
        Returns:
            List of result dictionaries
            
        Note:
            Use with caution - ensure queries are safe and validated
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params or [])
                    
                    # Handle different query types
                    if cursor.description:
                        # SELECT queries return results
                        return [dict(row) for row in cursor.fetchall()]
                    else:
                        # Non-SELECT queries return row count
                        return [{'affected_rows': cursor.rowcount}]

        except Exception as e:
            print(f"Failed to execute custom query: {e}")
            return [{'error': str(e)}]

    # Connection Management and Cleanup

    def close(self):
        """
        Close all database connections and cleanup resources.
        
        Properly shuts down the connection pool and releases
        all database connections for clean application shutdown.
        """
        if self.connection_pool:
            self.connection_pool.closeall()
            print("PostgreSQL connection pool closed")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic resource cleanup."""
        self.close()