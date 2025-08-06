import pandas as pd
from googletrans import Translator

# ==== CONFIGURATION ====
INPUT_FILE = "SA_Aqar.csv"  # Your original file
OUTPUT_FILE = "SA_Aqar_English.csv"  # Translated file
# =======================

def main():
    print("Loading CSV...")
    df = pd.read_csv(INPUT_FILE)

    translator = Translator()

    def translate_text(text):
        if pd.isna(text):
            return text
        try:
            return translator.translate(str(text), src='auto', dest='en').text
        except Exception as e:
            print(f"Error translating '{text}': {e}")
            return text

    print("Translating column names...")
    df.columns = [translate_text(col) for col in df.columns]

    print("Translating cell values (this may take a while)...")
    for col in df.columns:
        df[col] = df[col].map(translate_text)

    print("Saving translated file...")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Done! Translated file saved as '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
