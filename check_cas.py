import pandas as pd
import pubchempy as pcp
import re
from tqdm import tqdm
import time
import os

INPUT_FILE = ''
OUTPUT_FILE = ''
SMILES_COL = 'SMILES'

def check_cas_for_smiles(smiles, timeout=20):

    try:
        compounds = pcp.get_compounds(smiles, 'smiles', timeout=timeout)
        
        if not compounds:
            return 0, None
        
        cid = compounds[0].cid
        c = pcp.Compound.from_cid(cid)
        
        cas_pattern = r'^\d{2,7}-\d{2}-\d$'
        
        for syn in c.synonyms:
            if re.match(cas_pattern, str(syn)):
                cas_numbers = [s for s in c.synonyms if re.match(cas_pattern, str(s))]
                return 1, cas_numbers[:3]
        
        return 0, None
        
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return 0, None

def batch_check_cas(input_file, output_file, smiles_col, delay=0.4):
    
    print(f"\nLoading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found. Available columns: {list(df.columns)}")
    
    print(f"Found {len(df)} molecules")
    
    has_cas = []
    cas_numbers = []
    
    print("\nQuerying PubChem for CAS numbers...")
    for smiles in tqdm(df[smiles_col], desc="Processing"):
        result, cas_list = check_cas_for_smiles(str(smiles).strip())
        has_cas.append(result)
        cas_numbers.append(cas_list[0] if cas_list else None)
        time.sleep(delay)
    
    df['has_cas'] = has_cas
    df['cas_number'] = cas_numbers
    
    print(f"\nSaving results to: {output_file}")
    df.to_csv(output_file, index=False)

    print(f"Total molecules: {len(df)}")
    print(f"Has CAS number: {sum(has_cas)}")
    print(f"No CAS number: {len(df) - sum(has_cas)}")
    print(f"Percentage with CAS: {sum(has_cas)/len(df)*100:.1f}%")
    
    return df

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
    else:
        batch_check_cas(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            smiles_col=SMILES_COL
        )
