# @title
import os
import json
import time
from openai import OpenAI
#from google.colab import drive

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("Error! Check your OPENAI_API_KEY!")

INPUT_FILE = "../data/ank_raw.txt"
OUTPUT_FILE = "../data/ank_dataset.json"

client = OpenAI(api_key=API_KEY)

def generate_full_dataset():
    print(f"Reading file {INPUT_FILE}...")

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            full_text = f.read()
    except FileNotFoundError:
        print("❌ FILE NOT FOUND! Create ank_full.txt and paste the text.")
        return

    print(f"✅ File read successfully. Length: {len(full_text)} characters.")

    chunk_size = 15000
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    print(f"Split into {len(chunks)} parts. Starting FULL processing...")

    full_dataset = []

    chunks_to_process = chunks

    for i, chunk in enumerate(chunks_to_process):
        print(f"Part {i+1}/{len(chunks_to_process)}...")

        prompt = f"""
        Aš tau duodu gabalą iš LR Administracinių nusižengimų kodekso.
        Tavo tikslas: rasti nusižengimus ir baudas.

        Sukurk JSON QA poras:
        Instruction: Klausimas (pvz. "Kokia bauda už chuliganišką vairavimą?")
        Output: Atsakymas (pvz. "Pagal ANK 420 str., bauda yra ...")

        GRIEŽTAI:
        1. Tik validus JSON.
        2. Ignoruok turinį (menu, dates), jei tai ne įstatymas.
        3. Sukurk 3-6 poras (kuo daugiau, tuo geriau).

        Tekstas:
        {chunk}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )

            data = json.loads(response.choices[0].message.content)

            items = []
            if isinstance(data, list): items = data
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list): items = v; break

            if items:
                full_dataset.extend(items)
                print(f"   ✅ Added {len(items)} pairs. (Total: {len(full_dataset)})")
            else:
                print("   ⚠️ Empty (possibly technical or non-legal text).")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            time.sleep(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, ensure_ascii=False, indent=2)

    print(f"\nMISSION ACCOMPLISHED! Full dataset created.")
    print(f"Path: {OUTPUT_FILE}")
    print(f"Total samples: {len(full_dataset)}")

generate_full_dataset()
