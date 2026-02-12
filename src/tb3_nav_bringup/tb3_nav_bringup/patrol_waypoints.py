"""
Waypoint definitions for TurtleBot3 patrol navigation in small_house.world.

Coordinates are based on the Gazebo world coordinate system.
Each waypoint includes a name, position (x, y), and orientation (yaw).

Waypoints are optimized for object observation:
- Robot positioned 1.5-2m from target objects
- Yaw angle calculated to face each object directly
- Multiple waypoints per area for extended observation time

Floor Layout (approximate boundaries):
- Living Room: x: -2 to 3, y: -5 to 2
- Kitchen: x: 5 to 9, y: -5 to 2
- Bedroom: x: -9 to -2.5, y: -5 to 3
- Fitness/Balcony: x: -2 to 5, y: 2 to 5.5
- Hallway: x: 3 to 5, y: -5 to -3

Key obstacles to avoid:
- Sofa: (0.78, -0.41) ~1.5x0.8m
- Bed: (-6.17, 2.03) ~2x1.5m
- Wardrobe: (-3.15, 2.48) ~1x0.5m
- Kitchen table/chairs: (6.55, 0.95) area
- Cooking bench: (9.04, -3.35) wall-mounted
- Room wall: (-6.1, -1.5) divides bedroom
"""

import math


def _calc_yaw(robot_x: float, robot_y: float, obj_x: float, obj_y: float) -> float:
    """Calculate yaw angle for robot to face target object."""
    return math.atan2(obj_y - robot_y, obj_x - robot_x)


# Object positions from NaVQA dataset
_OBJECTS = {
    # Living Room
    "sofa": (0.78, -0.41),
    "coffee_table": (1.51, -1.73),
    "tv_living": (0.82, -5.38),
    "carpet": (0.79, -1.11),
    "trash_living": (2.36, -0.80),
    "tv_cabinet": (0.63, -5.18),
    "vase": (3.06, -5.41),
    "security_camera": (0.84, -5.00),
    # Kitchen
    "refrigerator": (8.70, -1.03),
    "kitchen_table": (6.55, 0.95),
    "cooking_bench": (9.04, -3.35),
    "kitchen_cabinet": (8.00, -3.84),
    "tableware": (7.15, 0.98),
    # Bedroom
    "bed": (-6.17, 2.03),
    "tv_bedroom": (-6.20, -1.39),
    "reading_desk": (-8.99, 2.06),
    "wardrobe": (-3.15, 2.48),
    "nightstand_1": (-7.73, 2.86),
    "nightstand_2": (-4.41, 2.86),
    "trash_bedroom": (-8.70, 1.00),
    "ball_bedroom": (-6.95, -4.22),
    "curtain": (-9.17, 0.20),
    # Fitness / Balcony
    "fitness_equipment": (3.48, 3.17),
    "dumbbell": (2.51, 2.72),
    "ball_fitness": (3.30, 4.23),
    "balcony_table": (-0.56, 4.11),
    "balcony_chair_1": (-1.38, 4.10),
    "balcony_chair_2": (0.33, 4.10),
    # Entrance / Hallway
    "door": (6.00, -5.55),
    "shoe_rack": (4.30, -5.17),
    "board": (2.78, -5.39),
}


def _waypoint(name: str, x: float, y: float, target: str) -> dict:
    """Create waypoint facing target object."""
    obj = _OBJECTS[target]
    return {"name": name, "x": x, "y": y, "yaw": _calc_yaw(x, y, obj[0], obj[1])}


# Main patrol waypoints (optimized for object observation)
# Each waypoint is positioned to observe specific objects with calculated yaw
# Adjusted to avoid obstacles and stay within accessible floor areas
WAYPOINTS = [
    # ============== Living Room Area ==============
    # Observe Sofa from front (avoid being on carpet/table)
    _waypoint("sofa_observer", 0.78, -2.5, "sofa"),
    # Observe Coffee Table and Tablet (from side, avoid sofa)
    _waypoint("coffee_table_observer", -0.5, -2.0, "coffee_table"),
    # Observe TV from middle of room
    _waypoint("tv_living_observer_1", 1.0, -2.0, "tv_living"),
    # Observe TV from closer (stay in open area)
    _waypoint("tv_living_observer_2", 1.0, -3.5, "tv_living"),
    # Observe Trash in living room (from open hallway area)
    _waypoint("trash_living_observer", 2.5, -3.0, "trash_living"),

    # ============== Entrance / Hallway ==============
    # Observe Board (from hallway center)
    _waypoint("board_observer", 2.5, -4.0, "board"),
    # Observe Shoe Rack (from hallway)
    _waypoint("shoe_rack_observer", 4.0, -4.0, "shoe_rack"),
    # Observe Entrance Door (from inside hallway)
    _waypoint("door_observer", 5.0, -4.0, "door"),

    # ============== Kitchen Area ==============
    # Observe Kitchen Table and Tableware (from entrance side)
    _waypoint("kitchen_table_observer_1", 5.5, 0.0, "kitchen_table"),
    _waypoint("kitchen_table_observer_2", 6.5, -0.5, "kitchen_table"),
    # Observe Kitchen Cabinet (from open area)
    _waypoint("kitchen_cabinet_observer", 6.5, -3.0, "kitchen_cabinet"),

    # ============== Fitness / Balcony Area ==============
    # Observe Fitness Equipment (from open floor)
    _waypoint("fitness_observer_1", 2.5, 2.5, "fitness_equipment"),
    # Observe Dumbbell (from side)
    _waypoint("dumbbell_observer", 1.5, 2.5, "dumbbell"),
    # Observe Ball in fitness area (from below)
    _waypoint("ball_fitness_observer", 2.5, 3.5, "ball_fitness"),
    # Observe Balcony Table and Chairs (from fitness side)
    _waypoint("balcony_table_observer", 0.5, 3.0, "balcony_table"),

    # ============== Bedroom Area ==============
    # Transition through hallway to bedroom door
    {"name": "hallway_to_bedroom", "x": -2.0, "y": -3.5, "yaw": 1.57},
    # Observe Ball in bedroom (from open floor near door)
    _waypoint("ball_bedroom_observer", -5.0, -3.5, "ball_bedroom"),
    # Observe Wardrobe (from bedroom floor, avoid wall)
    _waypoint("wardrobe_observer_1", -4.0, 1.0, "wardrobe"),
    # Observe Bed from foot (stay in open area between room wall and bed)
    _waypoint("bed_observer_1", -6.0, 0.5, "bed"),
    # Observe Nightstands (from foot of bed)
    _waypoint("nightstand_2_observer", -4.5, 1.0, "nightstand_2"),
    # Observe Reading Desk (from accessible side)
    _waypoint("reading_desk_observer_1", -8.0, 1.0, "reading_desk"),
    # Observe Trash in bedroom (near desk area)
    _waypoint("trash_bedroom_observer", -8.0, 0.0, "trash_bedroom"),
]

# Starting point (robot's initial position in the hallway)
START_POINT = {"name": "start_point", "x": -3.5, "y": -4.5, "yaw": 1.58}

# Summary statistics
WAYPOINT_COUNT = len(WAYPOINTS)
ESTIMATED_PATROL_TIME_MINUTES = WAYPOINT_COUNT * 0.5  # ~30 seconds per waypoint
