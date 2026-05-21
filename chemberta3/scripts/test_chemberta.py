from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from rdkit import Chem
import pandas as pd
from pathlib import Path
from rdkit.Chem import QED , rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity
#Load Model
model_name = "DeepChem/MoLFormer-c3-1.1B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

#Valid Tokens
valid_tokens = ["C","c","N","n","O","o","S","s","F","Cl","Br","I","=","#","-"]

df = pd.read_csv(Path.home() / "Downloads" / "chemberta3" / "chemberta3_benchmarking" / "data" / "data_preprocessing" / "splits" / "hiv" / "test_cleaned.csv")

smiles_list = []
for i in range(10):
    smiles_list.append(df["smiles"].iloc[i])

for i in smiles_list:
    print(f"\n=== SMILES: {i} ===")

    #Tokenize Smiles
    tokens = tokenizer.tokenize(i)
    ori_molecule = Chem.MolFromSmiles(i)
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    ori_fingerprint = fingerprint_generator.GetFingerprint(ori_molecule)
    if len(tokens) < 2:
        print("Too short to mask, skipping.")
        continue

    #Mask Smiles
    for j in range(len(tokens)):
        if tokens[j] not in valid_tokens:
           continue
        temp_tokens = tokens.copy()
        temp_tokens[j] = tokenizer.mask_token
        masked_smiles = tokenizer.convert_tokens_to_string(temp_tokens)
        print("\nMasked Smiles")
        print(masked_smiles)
        inputs = tokenizer(masked_smiles, return_tensors="pt")

        #Run Inference
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits

        #Extract top k predictions
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        mask_logits = logits[0, mask_token_index, :]
        top_k_preds = torch.topk(mask_logits, 50, dim=1).indices[0].tolist()

        print("\nTop Predictions:")
        candidate_smiles_list = []
        for k in top_k_preds:
            predicted_token = tokenizer.decode([k]).strip()
            if predicted_token not in valid_tokens:
                continue
            print(predicted_token)
            candidate_smiles = masked_smiles.replace(tokenizer.mask_token, predicted_token)
            candidate_smiles_list.append(candidate_smiles)

        #Validate using RDKit
        print("\nValidation Results:")
        for candidate_smiles in candidate_smiles_list:
            molecule = Chem.MolFromSmiles(candidate_smiles)
            with open("validation_results.txt", "a") as f:
                if molecule is not None:
                    score = QED.qed(molecule)
                    candidate_fingerprint = fingerprint_generator.GetFingerprint(molecule)
                    similarity = TanimotoSimilarity(ori_fingerprint, candidate_fingerprint)
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
                    print(f"Rejected: {candidate_smiles}")
                    f.write(f"Rejected: {candidate_smiles}")