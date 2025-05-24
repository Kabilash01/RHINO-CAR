class EmailAlert:
    def __init__(self, location=""):
        self.location = location

    def run(self):
        print(f"[EMAIL] 📧 Emergency alert sent from {self.location} (SIMULATED)")
