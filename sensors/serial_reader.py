import serial
import time

def read_serial_data(port="COM3", baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"[INFO] Connected to {port} at {baudrate} baud.")
        time.sleep(2)  # wait for serial to initialize

        while True:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    try:
                        parts = line.split(',')
                        headway = float(parts[0])
                        vsv = float(parts[1])
                        visibility = parts[2] if len(parts) > 2 else "clear"

                        print(f"[SENSOR] Headway: {headway} m | VSV: {vsv} km/h | Visibility: {visibility}")
                        
                        # Return these values to your main program
                        return headway, vsv, visibility

                    except ValueError:
                        print("[WARN] Invalid data format.")
            time.sleep(0.1)

    except serial.SerialException as e:
        print(f"[ERROR] Could not open serial port: {e}")
        return None, None, None

if __name__ == "__main__":
    read_serial_data()
