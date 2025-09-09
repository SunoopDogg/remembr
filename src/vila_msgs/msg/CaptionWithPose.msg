# CaptionWithPose.msg
# Custom message combining VILA caption with robot pose data

# VILA-generated caption from image segment
string caption

# Robot position
float64 position_x
float64 position_y
float64 position_z

# Robot orientation - yaw angle in radians (-π to π)
float64 theta

# Timestamp of when the caption was generated
builtin_interfaces/Time timestamp

# Number of images used to generate this caption
uint32 image_count

# Duration of the segment in seconds
float64 duration
