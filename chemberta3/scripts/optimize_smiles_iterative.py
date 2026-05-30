from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from rdkit import Chem
import pandas as pd
from pathlib import Path
from rdkit.Chem import QED , rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity

model_name = "DeepChem/MoLFormer-c3-1.1B"
valid_tokens = ["C","c","N","n","O","o","S","s","F","Cl","Br","I","=","#","-"]
num_steps = 3


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, model


def load_smiles(n=10):
    df = pd.read_csv(Path.home() / "Downloads" / "chemberta3" / "chemberta3_benchmarking" / "data" / "data_preprocessing" / "splits" / "hiv" / "test_cleaned.csv")
    return [df["smiles"].iloc[i] for i in range(n)]


def get_top_predictions(tokenizer, model, masked_smiles):
    inputs = tokenizer(masked_smiles, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_logits = logits[0, mask_token_index, :]
    return torch.topk(mask_logits, 50, dim=1).indices[0].tolist()


def get_candidate_smiles(tokenizer, masked_smiles, top_k_preds):
    candidate_smiles_list = []
    print("\nTop Predictions:")
    for k in top_k_preds:
        predicted_token = tokenizer.decode([k]).strip()
        if predicted_token not in valid_tokens:
            continue
        print(predicted_token)
        candidate_smiles = masked_smiles.replace(tokenizer.mask_token, predicted_token)
        candidate_smiles_list.append(candidate_smiles)
    return candidate_smiles_list


def validate_candidates(candidate_smiles_list, fingerprint_generator, ori_fingerprint, best_combined_score, best_smiles, search):
    number_valid = 0
    number_invalid = 0
    file = "greedy_search_results.txt" if search == "greedy" else "beam_search_results.txt"
    print("\nValidation Results:")
    with open(file, "a") as f:
        for candidate_smiles in candidate_smiles_list:
            molecule = Chem.MolFromSmiles(candidate_smiles)
            if molecule is not None:
                number_valid += 1
                score = QED.qed(molecule)
                candidate_fingerprint = fingerprint_generator.GetFingerprint(molecule)
                similarity = TanimotoSimilarity(ori_fingerprint, candidate_fingerprint)
                combined_score = score + similarity
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_smiles = candidate_smiles
                if score > 0.5 and similarity > 0.5:
                    print(f"Valid: {candidate_smiles} (QED: {score:.2f}, Similarity: {similarity:.2f})")
                    f.write(f"Valid: {candidate_smiles} (QED: {score:.2f}, Similarity: {similarity:.2f})\n")
                if score > 0.5 and similarity <= 0.5:
                    print(f"High QED but low similarity: {candidate_smiles} (QED: {score:.2f}, Similarity: {similarity:.2f})")
                    f.write(f"High QED but low similarity: {candidate_smiles} (QED: {score:.2f}, Similarity: {similarity:.2f})\n")
                if score <= 0.5 and similarity > 0.5:
                    print(f"Low QED but high similarity: {candidate_smiles} (QED: {score:.2f}, Similarity: {similarity:.2f})")
                    f.write(f"Low QED but high similarity: {candidate_smiles} (QED: {score:.2f}, Similarity: {similarity:.2f})\n")
                if score <= 0.5 and similarity <= 0.5:
                    print(f"Low QED and low similarity: {candidate_smiles} (QED: {score:.2f}, Similarity: {similarity:.2f})")
                    f.write(f"Low QED and low similarity: {candidate_smiles} (QED: {score:.2f}, Similarity: {similarity:.2f})\n")
            else:
                print(f"Rejected: {candidate_smiles}\n")
                f.write(f"Rejected: {candidate_smiles}\n")
                number_invalid += 1
        print(f"Total Valid: {number_valid}, Total Invalid: {number_invalid}")
        if number_valid + number_invalid > 0:
            print(f"Validity Rate: {number_valid / (number_valid + number_invalid):.2f}")
            f.write(f"Validity Rate: {number_valid / (number_valid + number_invalid):.2f}\n")
        else:
            print("No valid candidates to calculate validity rate.")
            f.write("No valid candidates to calculate validity rate.\n")
    return best_combined_score, best_smiles, number_valid, number_invalid


def greedy_search(tokenizer, model, smiles, num_steps):
    current_smiles = smiles
    ori_molecule = Chem.MolFromSmiles(current_smiles)
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    ori_fingerprint = fingerprint_generator.GetFingerprint(ori_molecule)
    total_valid = 0
    total_invalid = 0

    for i in range(num_steps):
        print(f"\n--- Step {i + 1} ---")
        tokens = tokenizer.tokenize(current_smiles)
        if len(tokens) < 2:
            print("Too short to mask, skipping.")
            continue
        best_combined_score = -1
        best_smiles = current_smiles

        for j in range(len(tokens)):
            if tokens[j] not in valid_tokens:
                continue
            temp_tokens = tokens.copy()
            temp_tokens[j] = tokenizer.mask_token
            masked_smiles = tokenizer.convert_tokens_to_string(temp_tokens)
            print("\nMasked Smiles")
            print(masked_smiles)

            top_k_preds = get_top_predictions(tokenizer, model, masked_smiles)
            candidate_smiles_list = get_candidate_smiles(tokenizer, masked_smiles, top_k_preds)
            best_combined_score, best_smiles, nv, ni = validate_candidates(
                candidate_smiles_list, fingerprint_generator, ori_fingerprint, best_combined_score, best_smiles, search="greedy"
            )
            total_valid += nv
            total_invalid += ni

        current_smiles = best_smiles
        print(f"\nBest Smiles after Step {i + 1}: {current_smiles} (Combined Score: {best_combined_score:.2f})")

    return current_smiles, total_valid, total_invalid

def beam_search(tokenizer, model, smiles, num_steps):
    beam_width = 3
    beam = [(smiles, -1)]  # (smiles, combined_score)
    ori_molecule = Chem.MolFromSmiles(smiles)
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    ori_fingerprint = fingerprint_generator.GetFingerprint(ori_molecule)
    total_valid = 0
    total_invalid = 0

    for i in range(num_steps):
        print(f"\n--- Beam Search Step {i + 1} ---")
        candidates = []
        for current_smiles, _ in beam:
            tokens = tokenizer.tokenize(current_smiles)
            if len(tokens) < 2:
                continue
            for j in range(len(tokens)):
                if tokens[j] not in valid_tokens:
                    continue
                temp_tokens = tokens.copy()
                temp_tokens[j] = tokenizer.mask_token
                masked_smiles = tokenizer.convert_tokens_to_string(temp_tokens)
                print("\nMasked Smiles")
                print(masked_smiles)

                top_k_preds = get_top_predictions(tokenizer, model, masked_smiles)
                candidate_smiles_list = get_candidate_smiles(tokenizer, masked_smiles, top_k_preds)
                best_score, best_candidate, nv, ni = validate_candidates(
                    candidate_smiles_list, fingerprint_generator, ori_fingerprint, -1, current_smiles, search="beam"
                )
                total_valid += nv
                total_invalid += ni
                if best_score > -1:
                    candidates.append((best_candidate, best_score))
        seen = {}
        for smiles, score in candidates:
            if smiles not in seen or score > seen[smiles]:
                seen[smiles] = score
        candidates = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_width]
        print(f"\nTop {beam_width} Candidates after Step {i + 1}:")
        final_molecules = []
        for candidate_smiles, combined_score in beam:
            final_molecules.append(candidate_smiles)
            print(f"{candidate_smiles} (Combined Score: {combined_score:.2f})")

    return final_molecules, total_valid, total_invalid

def unique_smiles(generated_smiles):
    unique = set(generated_smiles)
    return len(unique) / len(generated_smiles) if generated_smiles else 0

def novel_smiles(generated_smiles, original_smiles):
    original_set = set(original_smiles)
    novel = [s for s in generated_smiles if s not in original_set]
    return len(novel) / len(generated_smiles) if generated_smiles else 0


def main():
    tokenizer, model = load_model(model_name)
    smiles_list = load_smiles(10)
    greedy_generated_smiles = []
    beam_generated_smiles = []
    greedy_total_valid = 0
    greedy_total_invalid = 0
    beam_total_valid = 0
    beam_total_invalid = 0

    for smiles in smiles_list:
        print(f"\n=== SMILES: {smiles} ===")
        greedy_results, gv, gi = greedy_search(tokenizer, model, smiles, num_steps)
        beam_results, bv, bi = beam_search(tokenizer, model, smiles, num_steps)
        greedy_generated_smiles.append(greedy_results)
        beam_generated_smiles.append(beam_results)
        greedy_total_valid += gv
        greedy_total_invalid += gi
        beam_total_valid += bv
        beam_total_invalid += bi

    flat_beam = []
    for result in beam_generated_smiles:
        flat_beam.extend(result)

    greedy_validity_rate = greedy_total_valid / (greedy_total_valid + greedy_total_invalid) if (greedy_total_valid + greedy_total_invalid) > 0 else 0
    beam_validity_rate = beam_total_valid / (beam_total_valid + beam_total_invalid) if (beam_total_valid + beam_total_invalid) > 0 else 0

    print("\n=== Final Evaluation ===")
    with open("final_results.txt", "w") as f:
        f.write("Greedy Search Generated SMILES:\n")
        print(f"Greedy Search Unique SMILES: {unique_smiles(greedy_generated_smiles):.2f}")
        print(f"Greedy Search Novel SMILES: {novel_smiles(greedy_generated_smiles, smiles_list):.2f}")
        print(f"Greedy Search Total Valid: {greedy_total_valid}, Total Invalid: {greedy_total_invalid}, Validity Rate: {greedy_validity_rate:.2f}")
        f.write(f"Greedy Search Unique SMILES: {unique_smiles(greedy_generated_smiles):.2f}\n")
        f.write(f"Greedy Search Novel SMILES: {novel_smiles(greedy_generated_smiles, smiles_list):.2f}\n")
        f.write(f"Greedy Search Total Valid: {greedy_total_valid}, Total Invalid: {greedy_total_invalid}, Validity Rate: {greedy_validity_rate:.2f}\n")
        f.write("\nBeam Search Generated SMILES:\n")
        print(f"Beam Search Unique SMILES: {unique_smiles(flat_beam):.2f}")
        print(f"Beam Search Novel SMILES: {novel_smiles(flat_beam, smiles_list):.2f}")
        print(f"Beam Search Total Valid: {beam_total_valid}, Total Invalid: {beam_total_invalid}, Validity Rate: {beam_validity_rate:.2f}")
        f.write(f"Beam Search Unique SMILES: {unique_smiles(flat_beam):.2f}\n")
        f.write(f"Beam Search Novel SMILES: {novel_smiles(flat_beam, smiles_list):.2f}\n")
        f.write(f"Beam Search Total Valid: {beam_total_valid}, Total Invalid: {beam_total_invalid}, Validity Rate: {beam_validity_rate:.2f}\n")


if __name__ == "__main__":
    main()
