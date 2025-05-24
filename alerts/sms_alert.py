class SmsAlert:
    def __init__(self, location=""):
        self.location = location

    def run(self):
        print(f"[SMS] 🚨 Emergency alert sent from {self.location} (SIMULATED)")
