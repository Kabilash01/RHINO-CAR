def estimate_headway(sensor_value: float, box_height: float, use_sensor: bool = True) -> float:
    """
    Estimate headway using both sensor data and YOLO box height.
    
    Parameters:
    - sensor_value: Distance in cm from LiDAR or ultrasonic sensor.
    - box_height: Height of bounding box (pixels) from YOLO.
    - use_sensor: Whether to prioritize sensor input.

    Returns:
    - headway (in meters)
    """
    if use_sensor and sensor_value and 20 <= sensor_value <= 500:
        # Valid sensor reading range (20cm to 5m)
        return round(sensor_value / 100.0, 2)  # convert cm to meters
    elif box_height > 0:
        # Fallback: estimate distance from visual height
        # This factor can be tuned per camera calibration
        return round(1000 / (box_height + 1e-3), 2)
    else:
        # No usable input — assume max distance
        return 100.0
