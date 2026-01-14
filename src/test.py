from unsloth import FastLanguageModel
import torch

MODEL_NAME = "lukashm/LitLex-Llama-LT-v1"

def run_test():
    print(f"⏳ Loading model from: {MODEL_NAME}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

    questions = [
        "Kokia bauda gresia už greičio viršijimą daugiau kaip 50 km/h?",
        "Ką daryti, jei kaimynai triukšmauja naktį? Kokia bauda?",
        "Kokia bauda už nelegalų darbą?",
        "Kokia bauda už triukšmavimą vakaro metu?"
    ]

    print("\n⚖️ LITLEX AI TEISINĖ KONSULTACIJA ⚖️\n")

    for q in questions:
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    q,
                    "",
                )
            ], return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.batch_decode(outputs)[0]

        clean_response = response.split("### Response:\n")[1].replace("<|end_of_text|>", "").strip()

        print(f"👤 Klausimas: {q}")
        print(f"🤖 Atsakymas: {clean_response}")
        print("-" * 50)


if __name__ == "__main__":
    run_test()
