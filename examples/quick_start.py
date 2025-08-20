#!/usr/bin/env python3
"""
Enhanced LeZeA MLOps - Complete Quick Start Example
==================================================

Comprehensive example showcasing ALL LeZeA MLOps features including the newly implemented ones.

Shows:
- Experiment start/end with enhanced tracking
- LeZeA config + constraints + network genealogy
- Enhanced data usage with challenge tracking
- Component-level resource attribution
- Modification trees with real-time stats
- Tasker ↔ Builder reward flows
- Population tracking and evolution
- Advanced analytics and recommendations

NEW FEATURES DEMONSTRATED:
- Network lineage and genealogy tracking (1.4)
- Challenge-specific data usage rates (1.5.6) 
- Component-level resource attribution (1.5.4)
- Real-time modification stats (1.5.2-1.5.3)
- Tasker-Builder reward flows (1.6.1)
- Enhanced learning relevance (1.5.7)

Usage:
    python examples/enhanced_quick_start.py
"""

import os
import time
import math
import random
import tempfile
import uuid
from datetime import datetime
from lezea_mlops import ExperimentTracker
from lezea_mlops.monitoring.gpu_monitor import ComponentType
from lezea_mlops.monitoring.data_usage import DifficultyLevel
from lezea_mlops.modification.trees import ModificationType, AcceptanceReason


def simulate_network_evolution(tracker):
    """Demonstrate network genealogy and population tracking."""
    print("\n🧬 Simulating Network Evolution...")
    
    # Register initial population
    networks = {}
    for i in range(4):  # 4 taskers
        network_id = f"tasker_{i:02d}"
        networks[network_id] = {
            "type": "tasker",
            "generation": 0,
            "fitness": random.uniform(0.3, 0.7)
        }
        
        # Register with tracker
        tracker.register_network(
            network_id=network_id,
            network_type=tracker.NetworkType.TASKER,
            generation=0,
            layer_configs=[
                tracker.LayerSeedConfig(
                    layer_id=f"layer_{j}",
                    layer_type="dense",
                    seed=42 + i * 10 + j,
                    initialization_method="xavier",
                    parameters={"units": 64 + j * 32}
                ) for j in range(3)
            ]
        )
        tracker.update_network_fitness(network_id, networks[network_id]["fitness"])
    
    # Register builder
    builder_id = "builder_main"
    networks[builder_id] = {"type": "builder", "generation": 0, "fitness": 0.8}
    tracker.register_network(
        network_id=builder_id,
        network_type=tracker.NetworkType.BUILDER,
        generation=0
    )
    tracker.update_network_fitness(builder_id, 0.8)
    
    # Log initial population snapshot
    fitness_scores = [net["fitness"] for net in networks.values()]
    tracker.log_population_snapshot(
        tasker_count=4,
        builder_count=1,
        generation=0,
        fitness_scores=fitness_scores,
        diversity_metric=0.8,
        step=0
    )
    
    # Simulate evolution over generations
    for generation in range(1, 4):
        # Create offspring from best performers
        best_taskers = sorted(
            [(nid, net) for nid, net in networks.items() if net["type"] == "tasker"],
            key=lambda x: x[1]["fitness"],
            reverse=True
        )[:2]
        
        for i, (parent_id, parent_net) in enumerate(best_taskers):
            child_id = f"tasker_gen{generation}_{i:02d}"
            child_fitness = parent_net["fitness"] + random.uniform(-0.1, 0.2)
            child_fitness = max(0.1, min(0.95, child_fitness))
            
            networks[child_id] = {
                "type": "tasker",
                "generation": generation,
                "fitness": child_fitness
            }
            
            # Register with genealogy
            tracker.register_network(
                network_id=child_id,
                network_type=tracker.NetworkType.TASKER,
                parent_ids=[parent_id],
                generation=generation
            )
            tracker.update_network_fitness(child_id, child_fitness)
            
            # Track network modification
            tracker.track_network_modification(
                network_id=child_id,
                modification_type="mutation",
                details={"parent": parent_id, "mutation_rate": 0.1}
            )
        
        # Log population snapshot
        current_fitness = [net["fitness"] for net in networks.values()]
        diversity = 0.8 - generation * 0.1  # Decreasing diversity over time
        
        tracker.log_population_snapshot(
            tasker_count=len([n for n in networks.values() if n["type"] == "tasker"]),
            builder_count=1,
            generation=generation,
            fitness_scores=current_fitness,
            diversity_metric=max(0.3, diversity),
            step=generation * 5
        )
    
    return networks


def simulate_challenge_based_training(tracker, networks):
    """Demonstrate challenge-specific data usage and learning relevance."""
    print("\n🎯 Simulating Challenge-Based Training...")
    
    # Register challenges with different difficulty levels
    challenges = {
        "image_classification": {
            "difficulty": DifficultyLevel.EASY,
            "samples": [f"img_class_{i}" for i in range(100)]
        },
        "object_detection": {
            "difficulty": DifficultyLevel.MEDIUM, 
            "samples": [f"obj_det_{i}" for i in range(80)]
        },
        "semantic_segmentation": {
            "difficulty": DifficultyLevel.HARD,
            "samples": [f"sem_seg_{i}" for i in range(60)]
        },
        "3d_reconstruction": {
            "difficulty": DifficultyLevel.EXPERT,
            "samples": [f"3d_recon_{i}" for i in range(40)]
        }
    }
    
    # Register challenges with data usage tracker
    if tracker.data_usage:
        for challenge_id, challenge_info in challenges.items():
            tracker.data_usage.register_challenge(
                challenge_id=challenge_id,
                difficulty_level=challenge_info["difficulty"],
                sample_ids=challenge_info["samples"],
                description=f"Challenge: {challenge_id.replace('_', ' ').title()}",
                importance_multiplier=1.0 + challenge_info["difficulty"].value.count("e") * 0.2
            )
    
    # Simulate training with challenge-specific usage
    for step in range(1, 16):
        # Select a challenge for this step
        challenge_id = random.choice(list(challenges.keys()))
        challenge_info = challenges[challenge_id]
        
        # Sample data from the challenge
        batch_size = random.randint(8, 16)
        sample_ids = random.sample(challenge_info["samples"], min(batch_size, len(challenge_info["samples"])))
        
        # Simulate training metrics
        base_difficulty = len(challenge_info["difficulty"].value) * 0.1
        loss = 1.0 + base_difficulty - (step * 0.05) + random.uniform(-0.1, 0.1)
        loss = max(0.1, loss)
        
        accuracy = max(0.1, min(0.95, 1.0 - loss * 0.8 + random.uniform(-0.05, 0.05)))
        
        # Calculate importance weights (harder samples get higher weights)
        importance_weights = {
            sample_id: 1.0 + base_difficulty + random.uniform(-0.2, 0.2)
            for sample_id in sample_ids
        }
        
        # Log challenge usage rate
        tracker.log_challenge_usage_rate(
            challenge_id=challenge_id,
            difficulty_level=challenge_info["difficulty"].value,
            usage_rate=len(sample_ids) / len(challenge_info["samples"]),
            sample_count=len(sample_ids),
            importance_weights=importance_weights,
            step=step
        )
        
        # Calculate relevance scores based on performance
        performance_gain = max(0, 0.8 - loss)
        relevance_scores = {
            sample_id: performance_gain + random.uniform(-0.1, 0.1)
            for sample_id in sample_ids
        }
        
        # Log learning relevance
        tracker.log_learning_relevance(
            sample_ids=sample_ids,
            relevance_scores=relevance_scores,
            challenge_rankings={challenge_id: step},  # Simple ranking by step
            step=step
        )
        
        # Regular training step logging
        tracker.log_training_step(
            step,
            loss=loss,
            accuracy=accuracy,
            challenge_id=challenge_id,
            sample_ids=sample_ids,
            split="train",
            performance_metrics={"challenge_performance": performance_gain}
        )
        
        print(f"  Step {step:2d}: {challenge_id:20s} | Loss: {loss:.3f} | Acc: {accuracy:.3f}")


def simulate_component_tracking(tracker, networks):
    """Demonstrate component-level resource attribution."""
    print("\n📊 Simulating Component Resource Tracking...")
    
    # Track different components during operations
    components = [
        ("tasker_001", ComponentType.TASKER),
        ("builder_main", ComponentType.BUILDER), 
        ("algorithm_dqn", ComponentType.ALGORITHM),
        ("network_conv", ComponentType.NETWORK),
        ("layer_dense1", ComponentType.LAYER)
    ]
    
    # Register components with GPU monitor
    if tracker.gpu_monitor:
        for comp_id, comp_type in components:
            tracker.gpu_monitor.register_component(comp_id, comp_type)
        
        # Simulate component operations
        for step in range(5):
            time.sleep(0.1)  # Allow some time for monitoring
            
            # Simulate operations on components
            for comp_id, comp_type in components:
                # Record operations
                operation_count = random.randint(5, 20)
                tracker.gpu_monitor.record_component_operation(comp_id, operation_count)
                
                # Log component-specific resources via tracker
                tracker.log_component_resources(
                    component_id=comp_id,
                    component_type=comp_type.value,
                    cpu_percent=random.uniform(10, 80),
                    memory_mb=random.uniform(100, 2000),
                    gpu_util_percent=random.uniform(20, 90),
                    io_operations=random.randint(100, 1000),
                    step=step
                )
        
        # Get component analysis
        if hasattr(tracker.gpu_monitor, 'get_component_analysis'):
            analysis = tracker.gpu_monitor.get_component_analysis()
            if "error" not in analysis:
                print(f"  📈 Component analysis: {analysis['total_components']} components tracked")
                if analysis.get('imbalance_detected'):
                    print("  ⚠️  Resource imbalance detected between components")


def simulate_modification_tracking(tracker):
    """Demonstrate real-time modification stats and acceptance tracking."""
    print("\n🌳 Simulating Modification Tree Evolution...")
    
    modifications = []
    
    # Start a modification session
    with tracker.modification_trees[0].start_session("architecture_optimization", step=10) as session:
        # Add various types of modifications
        modification_types = [
            ("increase_layer_width", ModificationType.ARCHITECTURE, {"delta": 32, "layer": "conv1"}),
            ("adjust_learning_rate", ModificationType.HYPERPARAMETER, {"lr": 0.001, "decay": 0.95}),
            ("add_dropout", ModificationType.REGULARIZATION, {"rate": 0.3, "layer": "dense1"}),
            ("change_activation", ModificationType.ACTIVATION, {"from": "relu", "to": "swish"}),
            ("add_batch_norm", ModificationType.OPTIMIZATION, {"momentum": 0.9, "eps": 1e-5}),
        ]
        
        # Create modification tree
        parent_id = None
        for i, (op, mod_type, params) in enumerate(modification_types):
            node = session.add_modification(
                op=op,
                params=params,
                parent_id=parent_id,
                component_id=f"component_{i}",
                modification_type=mod_type,
                step=10 + i
            )
            
            # Simulate evaluation
            time.sleep(0.01)  # Small delay to simulate evaluation time
            
            # Random evaluation results
            performance_gain = random.uniform(-0.1, 0.3)
            accepted = performance_gain > 0.05
            
            reason = AcceptanceReason.PERFORMANCE_GAIN if accepted else AcceptanceReason.NO_IMPROVEMENT
            if performance_gain < -0.05:
                reason = AcceptanceReason.PERFORMANCE_DEGRADATION
            
            session.set_acceptance(
                node.id,
                accepted=accepted,
                score=0.8 + performance_gain,
                reason=reason,
                impact_score=performance_gain,
                evaluation_time_ms=random.uniform(10, 100)
            )
            
            modifications.append({
                "op": op,
                "accepted": accepted,
                "score": 0.8 + performance_gain,
                "type": mod_type.value
            })
            
            if accepted:
                parent_id = node.id  # Chain successful modifications
        
        print(f"  🔄 Created {len(modifications)} modifications")
        accepted_count = sum(1 for m in modifications if m["accepted"])
        print(f"  ✅ Accepted: {accepted_count}, Rejected: {len(modifications) - accepted_count}")
    
    # Log modification tree with tracker
    tracker.log_modification_tree(
        step=15,
        modifications=[
            {
                "type": m["op"],
                "accepted": m["accepted"],
                "score": m["score"],
                "modification_type": m["type"]
            } for m in modifications
        ],
        statistics={
            "total_modifications": len(modifications),
            "accepted_modifications": accepted_count,
            "rejected_modifications": len(modifications) - accepted_count,
            "acceptance_rate": accepted_count / len(modifications),
            "avg_evaluation_time_ms": 45.2
        }
    )


def simulate_reward_flows(tracker, networks):
    """Demonstrate Tasker ↔ Builder reward flows."""
    print("\n💰 Simulating Tasker-Builder Reward Flows...")
    
    # Get network IDs
    tasker_ids = [nid for nid, net in networks.items() if net["type"] == "tasker"]
    builder_id = "builder_main"
    
    # Simulate task assignments and evaluations
    for step in range(1, 11):
        # Each tasker performs tasks and gets evaluated by builder
        for tasker_id in tasker_ids[:3]:  # Use first 3 taskers
            # Simulate task performance
            base_performance = networks[tasker_id]["fitness"]
            task_performance = base_performance + random.uniform(-0.2, 0.2)
            task_performance = max(0.1, min(1.0, task_performance))
            
            # Builder evaluates tasker
            evaluation_score = task_performance + random.uniform(-0.1, 0.1)
            evaluation_score = max(0.0, min(1.0, evaluation_score))
            
            # Log reward flow from builder to tasker (evaluation)
            tracker.log_reward_flow(
                source_id=builder_id,
                target_id=tasker_id,
                source_type=tracker.NetworkType.BUILDER,
                target_type=tracker.NetworkType.TASKER,
                reward_value=evaluation_score,
                task_id=f"task_{step}_{tasker_id}",
                performance_metrics={
                    "task_completion_time": random.uniform(0.5, 2.0),
                    "accuracy": task_performance,
                    "efficiency": random.uniform(0.6, 1.0)
                },
                step=step
            )
            
            # Log reward flow from tasker to builder (task completion)
            completion_reward = task_performance * 0.8
            tracker.log_reward_flow(
                source_id=tasker_id,
                target_id=builder_id,
                source_type=tracker.NetworkType.TASKER,
                target_type=tracker.NetworkType.BUILDER,
                reward_value=completion_reward,
                task_id=f"task_{step}_{tasker_id}",
                performance_metrics={"completion_quality": task_performance},
                step=step
            )
        
        # Log aggregated results
        step_rewards = {f"tasker_{i}": random.uniform(0.3, 0.9) for i in range(3)}
        tracker.log_tasker_rewards(step_rewards, step=step)
        
        builder_rewards = {tasker_id: random.uniform(0.4, 0.8) for tasker_id in tasker_ids[:3]}
        tracker.log_builder_rewards(builder_rewards, step=step)
    
    print(f"  🔄 Simulated {len(tasker_ids) * 10} reward flow interactions")


def demonstrate_analytics_and_recommendations(tracker):
    """Show advanced analytics and recommendations."""
    print("\n📈 Generating Analytics and Recommendations...")
    
    # Get experiment summary with LeZeA data
    summary = tracker.get_experiment_summary()
    print(f"  📊 Experiment Summary Generated")
    print(f"     - Total training steps: {summary['training_steps']}")
    print(f"     - LeZeA networks: {summary['lezea_summary']['networks_registered']}")
    print(f"     - Population snapshots: {summary['lezea_summary']['population_snapshots']}")
    print(f"     - Reward flows: {summary['lezea_summary']['reward_flows']}")
    
    # Get recommendations
    recommendations = tracker.get_recommendations()
    print(f"\n💡 Recommendations ({len(recommendations)}):")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"     {i}. {rec}")
    
    # Population fitness summary
    if hasattr(tracker, 'get_population_fitness_summary'):
        fitness_summary = tracker.get_population_fitness_summary()
        if "error" not in fitness_summary:
            print(f"\n🏆 Population Fitness:")
            print(f"     - Networks tracked: {fitness_summary['total_networks']}")
            print(f"     - Overall average: {fitness_summary['overall_avg']:.3f}")
            print(f"     - Overall best: {fitness_summary['overall_max']:.3f}")


def main():
    print("🚀 Enhanced LeZeA MLOps - Complete Feature Demo")
    print("=" * 55)
    print("Demonstrating ALL LeZeA MLOps features including:")
    print("• Network genealogy and population tracking") 
    print("• Challenge-specific data usage and learning relevance")
    print("• Component-level resource attribution")
    print("• Real-time modification statistics")
    print("• Tasker-Builder reward flows")
    print("• Advanced analytics and recommendations")
    print()

    # Start enhanced experiment with all features
    with ExperimentTracker(
        "enhanced_lezea_demo", 
        purpose="Complete LeZeA MLOps feature demonstration",
        auto_start=True
    ) as tracker:
        
        # Enhanced LeZeA configuration
        tracker.log_lezea_config(
            tasker_pop_size=6,
            builder_pop_size=2, 
            algorithm_type="Enhanced_DQN",
            start_network_id="tasker_00",
            hyperparameters={
                "lr": 1e-3, 
                "batch": 32, 
                "gamma": 0.99,
                "epsilon_decay": 0.995,
                "target_update": 100
            },
            seeds={
                "global": 42, 
                "tasker": 1337, 
                "builder": 2023,
                "env": 9999
            },
            init_scheme="xavier_uniform",
        )
        
        # Enhanced constraints
        tracker.log_constraints(
            max_runtime=300,  # 5 minutes
            max_steps=50,
            max_episodes=100
        )

        # Enhanced dataset configuration
        tracker.log_data_splits(
            train=2000, 
            val=400, 
            test=600,
            extra={"validation_splits": 5, "stratified": True}
        )
        
        dataset_info = tracker.log_dataset_version(
            "enhanced_dataset", 
            dataset_root="data/enhanced/",
            preprocess_code_path=None  # Would be real path in practice
        )

        # 1. Network Evolution and Population Tracking
        networks = simulate_network_evolution(tracker)
        
        # 2. Challenge-Based Training with Enhanced Data Usage
        simulate_challenge_based_training(tracker, networks)
        
        # 3. Component Resource Tracking  
        simulate_component_tracking(tracker, networks)
        
        # 4. Modification Tree Evolution
        simulate_modification_tracking(tracker)
        
        # 5. Reward Flow Simulation
        simulate_reward_flows(tracker, networks)

        # Enhanced results logging
        print("\n📊 Logging Enhanced Results...")
        
        # Multi-class classification results
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 0, 2, 1, 1, 2, 0, 1, 1]
        results = tracker.log_classification_results(
            y_true, y_pred, 
            labels=[0, 1, 2], 
            split="test",
            step=25
        )
        print(f"  ✅ Classification accuracy: {results['accuracy']:.3f}")
        
        # Generation outputs
        generation_samples = [
            {"text": "Generated sample 1", "score": 0.85},
            {"text": "Generated sample 2", "score": 0.78},
            {"text": "Generated sample 3", "score": 0.92},
        ]
        gen_results = tracker.log_generation_outputs(
            generation_samples, 
            name="text_generation",
            step=25
        )
        print(f"  ✅ Generation avg score: {gen_results['avg_score']:.3f}")
        
        # RL episode results
        for episode in range(1, 6):
            episode_reward = random.uniform(100, 500)
            episode_steps = random.randint(50, 200)
            actions = [random.choice(['up', 'down', 'left', 'right']) for _ in range(episode_steps)]
            
            tracker.log_rl_episode(
                episode=episode,
                total_reward=episode_reward,
                steps=episode_steps,
                actions=actions,
                step=20 + episode
            )
        print(f"  ✅ Logged 5 RL episodes")

        # Enhanced checkpoint with metadata
        with tempfile.NamedTemporaryFile("w", suffix=".ckpt", delete=False) as f:
            f.write(f"Enhanced checkpoint with metadata\n")
            f.write(f"Networks: {len(networks)}\n") 
            f.write(f"Generation: 3\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            ckpt_path = f.name
        
        try:
            tracker.log_checkpoint(
                ckpt_path, 
                step=25, 
                role="enhanced_model",
                metadata={
                    "model_type": "enhanced_lezea",
                    "networks_count": len(networks),
                    "best_fitness": max(net["fitness"] for net in networks.values()),
                    "architecture": "multi_network_ensemble"
                }
            )
            print("  💾 Enhanced checkpoint logged")
        finally:
            os.unlink(ckpt_path)

        # Business metrics
        tracker.log_business_metrics(
            cost=15.50,
            comments="Enhanced LeZeA demo with full feature set",
            conclusion="All features working correctly, ready for production"
        )

        # Final analytics and recommendations
        demonstrate_analytics_and_recommendations(tracker)

        print(f"\n🎉 Enhanced LeZeA MLOps demo completed successfully!")
        print(f"   ✅ All features demonstrated and working")
        print(f"   ✅ Data logged to all available backends")
        print(f"   ✅ Ready for production LeZeA experiments")


if __name__ == "__main__":
    main()