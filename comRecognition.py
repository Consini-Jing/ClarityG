import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from typing import List, Tuple, Union

MODEL_PATH = "ClarityG/experiment/results/command-detection-roberta-balanced"
BATCH_SIZE = 32
MAX_LENGTH = 128
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def model_is_command(text: str, threshold: float = 0.9) -> Tuple[bool, float]:
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prob = probs[0, 1].item()  

    return prob >= threshold, prob
def detect_commands_in_file(file_path: str, output_path: str, threshold: float = 0.9):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    results = []
    commands = []

    for i in range(0, len(lines), BATCH_SIZE):
        batch = lines[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            batch_probs = probs[:, 1].tolist()

        for text, prob in zip(batch, batch_probs):
            is_command = prob >= threshold
            results.append([text, is_command, prob])
            if is_command:
                commands.append(text)

    all_lines_with_index = []
    token_cursor = 0
    for text, is_command, _ in results:
        all_lines_with_index.append((text, token_cursor, is_command))
        token_cursor += len(text.split())

    with open(output_path, "w", encoding="utf-8") as f_out:
        for text, is_command, prob in results:
            status = "[COMMAND]" if is_command else "[TEXT]"
            f_out.write(f"{status} {text} (Confidence: {prob:.4f})\n")

    return commands, all_lines_with_index


if __name__ == "__main__":
    results,_=detect_commands_in_file(
        file_path="ClarityG/datasets/reports_cmd/01Threat Spotlight Cyber Criminal Adoption of IPFS for Phishing, Malware Campaigns.txt",
        output_path="command_results.txt",
        threshold=0.9
    )
    print(results)


