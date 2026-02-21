#!/usr/bin/env python3
"""
NaVQA Evaluation Script for ReMEmbR Agent

This script evaluates the ReMEmbR Agent using the NaVQA dataset (67 questions).
It calls the /remembr_agent/query service and evaluates responses based on question type.

Question Types:
- Spatial (S01-S20): Evaluated by Euclidean distance from ground truth coordinates (20 questions)
- Temporal Point-in-time (T01-T10): Records time responses (10 questions)
- Temporal Duration (D01-D05): Records duration responses (5 questions)
- Descriptive Binary (B01-B20): Evaluates yes/no accuracy (20 questions)
- Descriptive Text (X01-X12): Records text responses (12 questions)

Ground truth coordinates and object presence are verified against small_house.world.

Usage:
    ros2 run agent navqa_evaluator --ros-args -p output_file:=/path/to/results.json
"""

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import rclpy
from rclpy.node import Node

from agent_msgs.srv import Query


# NaVQA Dataset - Questions based on small_house.world
# Ground truth coordinates are in (x, y, z) format from Gazebo world file
NAVQA_DATASET = {
    # ============== Spatial Questions (based on small_house.world models) ==============
    "S01": {
        "question": "Where is the refrigerator?",
        "type": "spatial",
        "ground_truth": (8.70, -1.03, 0.00),  # Refrigerator_01_001
    },
    "S02": {
        "question": "Where is the bed?",
        "type": "spatial",
        "ground_truth": (-6.17, 2.03, 0.00),  # Bed_01_001
    },
    "S03": {
        "question": "Where is the sofa?",
        "type": "spatial",
        "ground_truth": (0.78, -0.41, 0.00),  # SofaC_01_001
    },
    "S04": {
        "question": "Where is the kitchen table?",
        "type": "spatial",
        "ground_truth": (6.55, 0.95, 0.00),  # KitchenTable_01_001 - also called dining table
    },
    "S05": {
        "question": "Where is the TV?",
        "type": "spatial",
        "ground_truth": (0.82, -5.38, 0.00),  # TV_01_001 (living room)
    },
    "S06": {
        "question": "Where is the desk?",
        "type": "spatial",
        "ground_truth": (-8.99, 2.06, 0.00),  # ReadingDesk_01_001 - desk mentioned 40 times
    },
    "S07": {
        "question": "Where is the wardrobe?",
        "type": "spatial",
        "ground_truth": (-3.15, 2.48, 0.00),  # Wardrobe_01_001
    },
    "S08": {
        "question": "Where is the stove?",
        "type": "spatial",
        "ground_truth": (9.04, -3.35, 0.00),  # CookingBench_01_001 - stove/oven area
    },
    "S09": {
        "question": "Where is the air conditioner?",
        "type": "spatial",
        "ground_truth": (-2.00, -5.23, 0.00),  # AirconditionerB_01_001
    },
    "S10": {
        "question": "Where is the curtain?",
        "type": "spatial",
        "ground_truth": (-9.17, 0.20, 0.00),  # Curtain_01_001
    },
    "S11": {
        "question": "Where is the shoe rack?",
        "type": "spatial",
        "ground_truth": (4.30, -5.17, 0.00),  # ShoeRack_01_001
    },
    "S12": {
        "question": "Where is the trash can?",
        "type": "spatial",
        "ground_truth": (2.36, -0.80, 0.00),  # Trash_01_001
    },
    "S13": {
        "question": "Where is the coffee table?",
        "type": "spatial",
        "ground_truth": (1.51, -1.73, 0.00),  # CoffeeTable_01_001
    },
    "S14": {
        "question": "Where is the TV stand?",
        "type": "spatial",
        "ground_truth": (0.63, -5.18, 0.00),  # TVCabinet_01_001 - described as wooden table with TV
    },
    "S15": {
        "question": "Where is the kitchen cabinet?",
        "type": "spatial",
        "ground_truth": (8.00, -3.84, 0.00),  # KitchenCabinet_01_001
    },
    "S16": {
        "question": "Where is the nightstand?",
        "type": "spatial",
        "ground_truth": (-7.73, 2.86, 0.00),  # NightStand_01_001
    },
    "S17": {
        "question": "Where is the balcony table?",
        "type": "spatial",
        "ground_truth": (-0.56, 4.11, 0.00),  # BalconyTable_01_001
    },
    "S18": {
        "question": "Where is the bedroom TV?",
        "type": "spatial",
        "ground_truth": (-6.20, -1.39, 0.00),  # TV_02_001
    },
    "S19": {
        "question": "Where is the gym equipment?",
        "type": "spatial",
        "ground_truth": (3.48, 3.17, 0.00),  # FitnessEquipment_01_001
    },
    "S20": {
        "question": "Where is the carpet?",
        "type": "spatial",
        "ground_truth": (0.79, -1.11, 0.00),  # Carpet_01_001
    },
    # ============== Temporal Point-in-time Questions (T01-T10) ==============
    "T01": {
        "question": "When did you last see the refrigerator?",
        "type": "temporal_point",
        "ground_truth": None,  # Will record agent's response
    },
    "T02": {
        "question": "When did you last see the bed?",
        "type": "temporal_point",
        "ground_truth": None,
    },
    "T03": {
        "question": "When did you last see the sofa?",
        "type": "temporal_point",
        "ground_truth": None,
    },
    "T04": {
        "question": "When did you last see the kitchen table?",
        "type": "temporal_point",
        "ground_truth": None,
    },
    "T05": {
        "question": "When did you last see the TV?",
        "type": "temporal_point",
        "ground_truth": None,
    },
    "T06": {
        "question": "When did you pass by the kitchen?",
        "type": "temporal_point",
        "ground_truth": None,
    },
    "T07": {
        "question": "When did you pass by the bedroom?",
        "type": "temporal_point",
        "ground_truth": None,
    },
    "T08": {
        "question": "When did you pass by the living room?",
        "type": "temporal_point",
        "ground_truth": None,
    },
    "T09": {
        "question": "When did you pass by the bathroom?",
        "type": "temporal_point",
        "ground_truth": None,
    },
    "T10": {
        "question": "When did you pass by the entrance?",
        "type": "temporal_point",
        "ground_truth": None,
    },
    # ============== Temporal Duration Questions (D01-D05) ==============
    "D01": {
        "question": "How long did you stay in the kitchen?",
        "type": "temporal_duration",
        "ground_truth": None,
    },
    "D02": {
        "question": "How long did you stay in the bedroom?",
        "type": "temporal_duration",
        "ground_truth": None,
    },
    "D03": {
        "question": "How long did you stay in the living room?",
        "type": "temporal_duration",
        "ground_truth": None,
    },
    "D04": {
        "question": "How long did you stay in the bathroom?",
        "type": "temporal_duration",
        "ground_truth": None,
    },
    "D05": {
        "question": "How long did you stay near the entrance?",
        "type": "temporal_duration",
        "ground_truth": None,
    },
    # ============== Descriptive Binary Questions (B01-B20) ==============
    "B01": {
        "question": "Is there a refrigerator in the kitchen?",
        "type": "binary",
        "ground_truth": "yes",
    },
    "B02": {
        "question": "Is there a bed in the bedroom?",
        "type": "binary",
        "ground_truth": "yes",
    },
    "B03": {
        "question": "Is there a sofa in the living room?",
        "type": "binary",
        "ground_truth": "yes",
    },
    "B04": {
        "question": "Is there a table in the kitchen?",
        "type": "binary",
        "ground_truth": "yes",  # Table mentioned 177 times, kitchen 165 times
    },
    "B05": {
        "question": "Is there a TV in the living room?",
        "type": "binary",
        "ground_truth": "yes",
    },
    "B06": {
        "question": "Is there a bookshelf in the living room?",
        "type": "binary",
        "ground_truth": "yes",  # Bookshelf mentioned 24 times in living room/gym area
    },
    "B07": {
        "question": "Is there a washing machine in the bathroom?",
        "type": "binary",
        "ground_truth": "no",
    },
    "B08": {
        "question": "Is there a microwave in the kitchen?",
        "type": "binary",
        "ground_truth": "yes",  # Microwave mentioned in kitchen captions
    },
    "B09": {
        "question": "Is there a nightstand in the bedroom?",
        "type": "binary",
        "ground_truth": "yes",  # Nightstand + bedroom mentioned 13 times
    },
    "B10": {
        "question": "Is there a wardrobe in the bedroom?",
        "type": "binary",
        "ground_truth": "yes",
    },
    "B11": {
        "question": "Is there a bathtub in the bathroom?",
        "type": "binary",
        "ground_truth": "no",  # No bathtub model in small_house.world
    },
    "B12": {
        "question": "Is there a toilet in the bathroom?",
        "type": "binary",
        "ground_truth": "no",  # No toilet model in small_house.world
    },
    "B13": {
        "question": "Is there a sink in the kitchen?",
        "type": "binary",
        "ground_truth": "yes",  # Sink mentioned 30 times in kitchen captions
    },
    "B14": {
        "question": "Is there an oven in the living room?",
        "type": "binary",
        "ground_truth": "no",
    },
    "B15": {
        "question": "Is there an air conditioner in the bedroom?",
        "type": "binary",
        "ground_truth": "no",
    },
    "B16": {
        "question": "Is there a plant in the living room?",
        "type": "binary",
        "ground_truth": "no",  # No plant model in small_house.world
    },
    "B17": {
        "question": "Is there a mirror in the bathroom?",
        "type": "binary",
        "ground_truth": "no",  # No mirror model in small_house.world
    },
    "B18": {
        "question": "Is there a lamp in the bedroom?",
        "type": "binary",
        "ground_truth": "yes",  # Lamp mentioned in bedroom captions (nightstand lamp)
    },
    "B19": {
        "question": "Is there a curtain in the bedroom?",
        "type": "binary",
        "ground_truth": "yes",
    },
    "B20": {
        "question": "Is there a shoe rack in the entrance?",
        "type": "binary",
        "ground_truth": "yes",
    },
    # ============== Descriptive Text Questions (X01-X12) ==============
    "X01": {
        "question": "What objects are in the kitchen?",
        "type": "descriptive",
        "ground_truth": None,  # Will record agent's response
    },
    "X02": {
        "question": "What objects are in the bedroom?",
        "type": "descriptive",
        "ground_truth": None,
    },
    "X03": {
        "question": "What objects are in the living room?",
        "type": "descriptive",
        "ground_truth": None,
    },
    "X04": {
        "question": "What objects are in the bathroom?",
        "type": "descriptive",
        "ground_truth": None,
    },
    "X05": {
        "question": "What color is the sofa?",
        "type": "descriptive",
        "ground_truth": None,  # Expected: gray (most common sofa color in captions)
    },
    "X06": {
        "question": "What color is the bed?",
        "type": "descriptive",
        "ground_truth": None,
    },
    "X07": {
        "question": "What is on the kitchen table?",
        "type": "descriptive",
        "ground_truth": None,
    },
    "X08": {
        "question": "What is on the desk?",
        "type": "descriptive",
        "ground_truth": None,
    },
    "X09": {
        "question": "What is next to the refrigerator?",
        "type": "descriptive",
        "ground_truth": None,
    },
    "X10": {
        "question": "What is next to the TV?",
        "type": "descriptive",
        "ground_truth": None,
    },
    "X11": {
        "question": "What does the living room look like?",
        "type": "descriptive",
        "ground_truth": None,  # Expected: sofa, coffee table, TV, etc.
    },
    "X12": {
        "question": "What does the bedroom look like?",
        "type": "descriptive",
        "ground_truth": None,  # Expected: bed, nightstand, lamp, light blue walls
    },
}


class NaVQAEvaluator(Node):
    """ROS2 Node for evaluating ReMEmbR Agent with NaVQA dataset."""

    def __init__(self):
        super().__init__("navqa_evaluator")

        # Declare parameters
        self.declare_parameter("output_file", "navqa_results.json")
        self.declare_parameter("timeout_sec", 60.0)  # Increased from 30 to 60 seconds
        self.declare_parameter("question_ids", "")  # Empty = all questions

        # Get parameters
        self.output_file = self.get_parameter("output_file").value
        self.timeout_sec = self.get_parameter("timeout_sec").value
        question_ids_param = self.get_parameter("question_ids").value

        # Parse question IDs if specified
        if question_ids_param:
            self.question_ids = [q.strip() for q in question_ids_param.split(",")]
        else:
            self.question_ids = list(NAVQA_DATASET.keys())

        # Create service client
        self.query_client = self.create_client(Query, "/remembr_agent/query")

        self.get_logger().info("NaVQA Evaluator initialized")
        self.get_logger().info(f"Output file: {self.output_file}")
        self.get_logger().info(f"Questions to evaluate: {len(self.question_ids)}")

    def wait_for_service(self) -> bool:
        """Wait for the agent service to become available."""
        self.get_logger().info("Waiting for /remembr_agent/query service...")
        if not self.query_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("Service not available after 10 seconds")
            return False
        self.get_logger().info("Service is available")
        return True

    def call_agent(self, question: str) -> Optional[dict]:
        """Call the ReMEmbR agent with a question."""
        request = Query.Request()
        request.question = question

        future = self.query_client.call_async(request)

        # Spin until complete or timeout
        start_time = self.get_clock().now()
        while rclpy.ok():
            # Use spin_once with a small timeout to avoid blocking
            rclpy.spin_once(self, timeout_sec=0.5)

            # Check elapsed time BEFORE checking future.done()
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > self.timeout_sec:
                self.get_logger().warning(
                    f"Timeout ({self.timeout_sec}s) waiting for response to: {question}"
                )
                future.cancel()
                return None

            if future.done():
                break

        if not future.done():
            self.get_logger().error(f"Future not done after loop exit for: {question}")
            return None

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed for: {question} - {e}")
            return None

        if response is None:
            self.get_logger().error(f"Service call returned None for: {question}")
            return None

        return {
            "success": response.success,
            "type": response.type,
            "text": response.text,
            "binary": response.binary,
            "position": list(response.position) if response.position else [],
            "orientation": response.orientation,
            "time": response.time,
            "duration": response.duration,
            "error": response.error,
        }

    @staticmethod
    def calculate_euclidean_distance(
        predicted: tuple | list, ground_truth: tuple | list
    ) -> float:
        """Calculate Euclidean distance between two 3D points."""
        if len(predicted) < 3 or len(ground_truth) < 3:
            return float("inf")

        return math.sqrt(
            (predicted[0] - ground_truth[0]) ** 2
            + (predicted[1] - ground_truth[1]) ** 2
            + (predicted[2] - ground_truth[2]) ** 2
        )

    @staticmethod
    def extract_binary_answer(text: str) -> Optional[str]:
        """Extract yes/no answer from response text."""
        text_lower = text.lower().strip()

        # Check for explicit yes/no patterns
        if re.search(r"\byes\b", text_lower):
            return "yes"
        if re.search(r"\bno\b", text_lower):
            return "no"

        # Check for affirmative/negative phrases
        affirmative_patterns = [
            r"\bthere is\b",
            r"\byes,?\s+there",
            r"\bfound\b",
            r"\bexists?\b",
            r"\bpresent\b",
            r"\bcan see\b",
            r"\bsaw\b",
        ]
        negative_patterns = [
            r"\bthere is no\b",
            r"\bthere isn't\b",
            r"\bnot found\b",
            r"\bdoesn't exist\b",
            r"\bdon't see\b",
            r"\bcannot find\b",
            r"\bno .* found\b",
        ]

        for pattern in negative_patterns:
            if re.search(pattern, text_lower):
                return "no"

        for pattern in affirmative_patterns:
            if re.search(pattern, text_lower):
                return "yes"

        return None

    def evaluate_spatial(
        self, result: dict, ground_truth: tuple
    ) -> dict[str, Any]:
        """Evaluate a spatial question result."""
        evaluation = {
            "predicted_position": None,
            "ground_truth_position": list(ground_truth),
            "euclidean_distance": None,
            "within_1m": False,
            "within_2m": False,
            "within_5m": False,
        }

        if not result or not result.get("success"):
            return evaluation

        position = result.get("position", [])
        if len(position) >= 2:
            # Ensure we have 3D coordinates (add z=0 if missing)
            if len(position) == 2:
                position = [position[0], position[1], 0.0]
            evaluation["predicted_position"] = position

            distance = self.calculate_euclidean_distance(position, ground_truth)
            evaluation["euclidean_distance"] = round(distance, 4)
            evaluation["within_1m"] = distance <= 1.0
            evaluation["within_2m"] = distance <= 2.0
            evaluation["within_5m"] = distance <= 5.0

        return evaluation

    def evaluate_binary(
        self, result: dict, ground_truth: str
    ) -> dict[str, Any]:
        """Evaluate a binary (yes/no) question result."""
        evaluation = {
            "predicted_answer": None,
            "ground_truth_answer": ground_truth,
            "correct": False,
        }

        if not result or not result.get("success"):
            return evaluation

        text = result.get("text", "")
        predicted = self.extract_binary_answer(text)
        evaluation["predicted_answer"] = predicted
        evaluation["correct"] = predicted == ground_truth

        return evaluation

    def evaluate_temporal(self, result: dict) -> dict[str, Any]:
        """Evaluate a temporal question result."""
        evaluation = {
            "response_time": None,
            "response_text": None,
            "has_time_info": False,
        }

        if not result or not result.get("success"):
            return evaluation

        evaluation["response_text"] = result.get("text", "")
        time_value = result.get("time", 0.0)
        if time_value > 0:
            evaluation["response_time"] = time_value
            evaluation["has_time_info"] = True

        return evaluation

    def evaluate_descriptive(self, result: dict) -> dict[str, Any]:
        """Evaluate a descriptive question result."""
        evaluation = {
            "response_text": None,
            "response_length": 0,
        }

        if not result or not result.get("success"):
            return evaluation

        text = result.get("text", "")
        evaluation["response_text"] = text
        evaluation["response_length"] = len(text)

        return evaluation

    def run_evaluation(self) -> dict:
        """Run the full NaVQA evaluation."""
        results = []
        start_time = datetime.now()

        self.get_logger().info(f"Starting evaluation of {len(self.question_ids)} questions")

        for idx, question_id in enumerate(self.question_ids):
            if question_id not in NAVQA_DATASET:
                self.get_logger().warning(f"Unknown question ID: {question_id}")
                continue

            question_data = NAVQA_DATASET[question_id]
            question = question_data["question"]
            q_type = question_data["type"]
            ground_truth = question_data["ground_truth"]

            self.get_logger().info(
                f"[{idx + 1}/{len(self.question_ids)}] Evaluating {question_id}: {question}"
            )

            # Call agent
            response = self.call_agent(question)

            # Build result entry
            result_entry = {
                "id": question_id,
                "question": question,
                "type": q_type,
                "ground_truth": (
                    list(ground_truth) if isinstance(ground_truth, tuple) else ground_truth
                ),
                "response": response,
                "evaluation": {},
            }

            # Type-specific evaluation
            if q_type == "spatial":
                result_entry["evaluation"] = self.evaluate_spatial(response, ground_truth)
            elif q_type == "binary":
                result_entry["evaluation"] = self.evaluate_binary(response, ground_truth)
            elif q_type in ("temporal_point", "temporal_duration"):
                result_entry["evaluation"] = self.evaluate_temporal(response)
            elif q_type == "descriptive":
                result_entry["evaluation"] = self.evaluate_descriptive(response)

            results.append(result_entry)

            # Log progress for spatial questions
            if q_type == "spatial" and result_entry["evaluation"].get("euclidean_distance"):
                dist = result_entry["evaluation"]["euclidean_distance"]
                self.get_logger().info(f"  -> Euclidean distance: {dist:.2f}m")

        end_time = datetime.now()

        # Generate summary
        summary = self.generate_summary(results)

        return {
            "evaluation_time": start_time.isoformat(),
            "evaluation_duration_seconds": (end_time - start_time).total_seconds(),
            "total_questions": len(results),
            "results": results,
            "summary": summary,
        }

    def generate_summary(self, results: list) -> dict:
        """Generate evaluation summary statistics."""
        summary = {
            "spatial": {
                "total": 0,
                "successful": 0,
                "distances": [],
                "avg_distance": None,
                "within_1m": 0,
                "within_2m": 0,
                "within_5m": 0,
            },
            "temporal_point": {
                "total": 0,
                "with_time_info": 0,
            },
            "temporal_duration": {
                "total": 0,
                "with_time_info": 0,
            },
            "binary": {
                "total": 0,
                "correct": 0,
                "accuracy": None,
            },
            "descriptive": {
                "total": 0,
                "avg_response_length": None,
            },
        }

        for result in results:
            q_type = result["type"]
            evaluation = result["evaluation"]

            if q_type == "spatial":
                summary["spatial"]["total"] += 1
                dist = evaluation.get("euclidean_distance")
                if dist is not None and dist != float("inf"):
                    summary["spatial"]["successful"] += 1
                    summary["spatial"]["distances"].append(dist)
                    if evaluation.get("within_1m"):
                        summary["spatial"]["within_1m"] += 1
                    if evaluation.get("within_2m"):
                        summary["spatial"]["within_2m"] += 1
                    if evaluation.get("within_5m"):
                        summary["spatial"]["within_5m"] += 1

            elif q_type == "temporal_point":
                summary["temporal_point"]["total"] += 1
                if evaluation.get("has_time_info"):
                    summary["temporal_point"]["with_time_info"] += 1

            elif q_type == "temporal_duration":
                summary["temporal_duration"]["total"] += 1
                if evaluation.get("has_time_info"):
                    summary["temporal_duration"]["with_time_info"] += 1

            elif q_type == "binary":
                summary["binary"]["total"] += 1
                if evaluation.get("correct"):
                    summary["binary"]["correct"] += 1

            elif q_type == "descriptive":
                summary["descriptive"]["total"] += 1

        # Calculate averages
        if summary["spatial"]["distances"]:
            summary["spatial"]["avg_distance"] = round(
                sum(summary["spatial"]["distances"]) / len(summary["spatial"]["distances"]),
                4,
            )
        # Remove raw distances from summary (too verbose)
        del summary["spatial"]["distances"]

        if summary["binary"]["total"] > 0:
            summary["binary"]["accuracy"] = round(
                summary["binary"]["correct"] / summary["binary"]["total"], 4
            )

        descriptive_lengths = [
            r["evaluation"].get("response_length", 0)
            for r in results
            if r["type"] == "descriptive" and r["evaluation"].get("response_length")
        ]
        if descriptive_lengths:
            summary["descriptive"]["avg_response_length"] = round(
                sum(descriptive_lengths) / len(descriptive_lengths), 2
            )

        return summary

    def save_results(self, results: dict) -> None:
        """Save evaluation results to JSON file."""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.get_logger().info(f"Results saved to: {output_path}")


def main(args=None):
    """Main entry point for NaVQA evaluator."""
    rclpy.init(args=args)

    evaluator = NaVQAEvaluator()

    try:
        if not evaluator.wait_for_service():
            evaluator.get_logger().error("Cannot proceed without agent service")
            return

        results = evaluator.run_evaluation()
        evaluator.save_results(results)

        # Print summary to console
        summary = results["summary"]
        evaluator.get_logger().info("=" * 50)
        evaluator.get_logger().info("EVALUATION SUMMARY")
        evaluator.get_logger().info("=" * 50)

        evaluator.get_logger().info(
            f"Spatial: {summary['spatial']['successful']}/{summary['spatial']['total']} "
            f"with position data, avg distance: {summary['spatial']['avg_distance']}m"
        )
        evaluator.get_logger().info(
            f"  - Within 1m: {summary['spatial']['within_1m']}"
        )
        evaluator.get_logger().info(
            f"  - Within 2m: {summary['spatial']['within_2m']}"
        )
        evaluator.get_logger().info(
            f"  - Within 5m: {summary['spatial']['within_5m']}"
        )

        evaluator.get_logger().info(
            f"Binary: {summary['binary']['correct']}/{summary['binary']['total']} correct "
            f"(accuracy: {summary['binary']['accuracy']})"
        )

        evaluator.get_logger().info(
            f"Temporal Point: {summary['temporal_point']['with_time_info']}/"
            f"{summary['temporal_point']['total']} with time info"
        )
        evaluator.get_logger().info(
            f"Temporal Duration: {summary['temporal_duration']['with_time_info']}/"
            f"{summary['temporal_duration']['total']} with time info"
        )
        evaluator.get_logger().info(
            f"Descriptive: {summary['descriptive']['total']} responses, "
            f"avg length: {summary['descriptive']['avg_response_length']} chars"
        )

        evaluator.get_logger().info("=" * 50)

    except KeyboardInterrupt:
        evaluator.get_logger().info("Evaluation interrupted by user")
    finally:
        evaluator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
