import pandas as pd
from pathlib import Path

class KnowledgeBase:
    def __init__(self, stats_file="clash_royale_cards.xlsx"):
        # Look for file in the same directory as this script
        self.stats_file = Path(__file__).parent / stats_file
        self.data = pd.DataFrame()
        self._load_data()

    def _load_data(self):
        """Loads the stats file (CSV or Excel)."""
        if not self.stats_file.exists():
            print(f"[WARNING] Stats file '{self.stats_file}' not found.")
            print("Please download 'clash_royale_cards.xlsx' from Kaggle and save it here.")
            # Create dummy data for testing
            self.data = pd.DataFrame([
                {"card": "knight", "hp": 1452, "damage": 167, "target": "ground", "range": "melee"},
                {"card": "archer", "hp": 254, "damage": 89, "target": "air_ground", "range": 5},
                {"card": "giant", "hp": 3275, "damage": 211, "target": "buildings", "range": "melee"}
            ])
            return

        try:
            if self.stats_file.suffix == '.xlsx':
                self.data = pd.read_excel(self.stats_file)
            else:
                self.data = pd.read_csv(self.stats_file)
            
            # Normalize column names to lowercase
            self.data.columns = [c.lower().strip() for c in self.data.columns]
            print(f"[INFO] Loaded stats for {len(self.data)} cards.")
        except Exception as e:
            print(f"[ERROR] Failed to load stats: {e}")

    def get_card_stats(self, card_name):
        """Returns the stats dictionary for a given card name (fuzzy match)."""
        if self.data.empty:
            return {}
        
        # Simple exact match first
        row = self.data[self.data['card'] == card_name.lower()]
        
        # If not found, try contains
        if row.empty:
            row = self.data[self.data['card'].astype(str).str.contains(card_name.lower())]

        if not row.empty:
            return row.iloc[0].to_dict()
        else:
            return None

if __name__ == "__main__":
    kb = KnowledgeBase()
    print("Knight Stats:", kb.get_card_stats("knight"))
