import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union

import numpy as np
import torch
import yaml

from chai_lab.chai1 import run_inference
from chai_lab.data.dataset.inference_dataset import read_inputs
from chai_lab.data.dataset.msas.colabfold import generate_colabfold_msas
from chai_lab.data.parsing.structure.entity_type import EntityType

def parse_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict) or 'sequences' not in config:
        raise ValueError("Config must contain a 'sequences' key")
    
    return config

def create_fasta_from_config(config: Dict[str, Any], tmp_dir: Path) -> Path:
    """Generate a FASTA file from the configuration."""
    fasta_entries = []
    
    for entry in config['sequences']:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("Each sequence entry must have exactly one entity type")
        
        entity_type = list(entry.keys())[0]
        entity_data = entry[entity_type]
        
        # Validate required fields
        if 'id' not in entity_data:
            raise ValueError(f"Missing 'id' field in {entity_type} entry")
        
        # Convert single ID to list for consistency
        chain_ids = entity_data['id'] if isinstance(entity_data['id'], list) else [entity_data['id']]
        
        if entity_type in ['protein', 'dna', 'rna']:
            if 'sequence' not in entity_data:
                raise ValueError(f"Missing 'sequence' field in {entity_type} entry")
            
            for i, chain_id in enumerate(chain_ids):
                # Format header to match expected format: >protein|name=Cx43_Xenopus-laevis_1
                header = f">{entity_type}|name={entity_type}_{chain_id}"
                if len(chain_ids) > 1:
                    header += f"_{i+1}"
                sequence = entity_data['sequence']
                fasta_entries.extend([header, sequence])
        
        elif entity_type == 'ligand':
            if 'smiles' not in entity_data and 'ccd' not in entity_data:
                raise ValueError("Ligand entry must have either 'smiles' or 'ccd' field")
            if 'smiles' in entity_data and 'ccd' in entity_data:
                raise ValueError("Ligand entry cannot have both 'smiles' and 'ccd' fields")
            
            for i, chain_id in enumerate(chain_ids):
                # Format header for ligand
                header = f">ligand|name=ligand_{chain_id}"
                if len(chain_ids) > 1:
                    header += f"_{i+1}"
                if 'smiles' in entity_data:
                    sequence = entity_data['smiles']
                else:
                    sequence = f"CCD:{entity_data['ccd']}"
                fasta_entries.extend([header, sequence])
    
    fasta_content = '\n'.join(fasta_entries)
    fasta_path = tmp_dir / "input.fasta"
    fasta_path.write_text(fasta_content)
    
    return fasta_path

def collect_msas(config: Dict[str, Any], msa_dir: Path) -> None:
    """Generate MSAs for all protein sequences."""
    protein_seqs = []
    
    for entry in config['sequences']:
        entity_type = list(entry.keys())[0]
        entity_data = entry[entity_type]
        
        if entity_type == 'protein':
            protein_seqs.append(entity_data['sequence'])
    
    if protein_seqs:
        generate_colabfold_msas(protein_seqs=protein_seqs, msa_dir=msa_dir)

def main():
    parser = argparse.ArgumentParser(description="Run structure prediction with YAML config")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--device", default="cuda:0", help="Device to run inference on")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: ./outputs/TIMESTAMP)")
    args = parser.parse_args()
    
    # Create temporary directory for intermediate files
    tmp_dir = Path(tempfile.mkdtemp())
    
    # Parse config and create FASTA
    config = parse_config(args.config)
    fasta_path = create_fasta_from_config(config, tmp_dir)
    
    # Set up MSA directory and generate MSAs
    msa_dir = tmp_dir / "msas"
    msa_dir.mkdir()
    collect_msas(config, msa_dir)
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./outputs/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        seed=42,
        device=torch.device(args.device),
        use_esm_embeddings=True,
        msa_directory=msa_dir,
    )
    
    # Save results
    cif_paths = candidates.cif_paths
    scores = [rd.aggregate_score for rd in candidates.ranking_data]
    
    # Load detailed scores for sample 2
    scores = np.load(output_dir.joinpath("scores.model_idx_2.npz"))
    
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
