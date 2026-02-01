"""
Waypoint definitions for TurtleBot3 patrol navigation in small_house.world.

Coordinates are based on the Gazebo world coordinate system.
Each waypoint includes a name, position (x, y), and orientation (yaw).
"""

# Main patrol waypoints (15 locations in the house)
WAYPOINTS = [
    # Living Room Area
    {"name": "living_room", "x": 0.9, "y": -1.5, "yaw": 0.0},
    {"name": "sofa_front", "x": 1.5, "y": -0.4, "yaw": 3.14},
    {"name": "tv_front", "x": 1.0, "y": -4.0, "yaw": 1.57},

    # Sunroom / Balcony Area
    {"name": "sunroom", "x": -0.5, "y": 4.0, "yaw": 1.57},
    {"name": "fitness_area", "x": 3.0, "y": 3.5, "yaw": 0.0},
    {"name": "balcony_window", "x": 1.0, "y": 5.0, "yaw": 1.57},

    # Kitchen Area
    {"name": "kitchen_table", "x": 5.5, "y": 0.0, "yaw": 0.0},
    {"name": "kitchen_cooking", "x": 7.5, "y": -3.0, "yaw": 3.14},
    {"name": "refrigerator", "x": 8.0, "y": -1.0, "yaw": 1.57},

    # Entrance / Hallway Area
    {"name": "entrance", "x": 4.3, "y": -4.5, "yaw": 0.0},
    {"name": "hallway_center", "x": -0.5, "y": -4.5, "yaw": 0.0},
    {"name": "hallway_to_bedroom", "x": -3.5, "y": -2.0, "yaw": 1.57},

    # Bedroom Area
    {"name": "bedroom", "x": -6.2, "y": 0.5, "yaw": 0.0},
    {"name": "bedroom_desk", "x": -8.3, "y": 1.5, "yaw": 3.14},
    {"name": "wardrobe", "x": -3.5, "y": 2.0, "yaw": -1.57},
]

# Starting point (robot's initial position in the hallway)
START_POINT = {"name": "start_point", "x": -3.5, "y": -4.5, "yaw": 1.58}
