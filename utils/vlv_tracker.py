
# vlv_tracker.py

# This module holds the latest tracked VLV (km/h) and allows sharing across modules

vlv_global_kmph = 40.0  # Default fallback value


def update_vlv(new_vlv_kmph):
    global vlv_global_kmph
    vlv_global_kmph = new_vlv_kmph


def get_vlv():
    return vlv_global_kmph
