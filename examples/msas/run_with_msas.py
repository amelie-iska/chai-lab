import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from chai_lab.chai1 import run_inference
from chai_lab.data.dataset.inference_dataset import read_inputs
from chai_lab.data.dataset.msas.colabfold import generate_colabfold_msas
from chai_lab.data.parsing.structure.entity_type import EntityType

tmp_dir = Path(tempfile.mkdtemp())

# Prepare input fasta
example_fasta = """
>protein|name=Cx43_Xenopus-laevis_1
MGDWSALGRLLDKVQAYSTAGGKVWLSVLFIFRILLLGTAVESAWGDEQSAFVCNTQQPGCENVCYDKSFPISHVRFWVLQIIFVSTPTLLYLAHVFYLMRKEEKLNRKEEELKMVQNEGGNVDMHLKQIEIKKFKYGLEEHGKVKMRGGLLRTYIISILFKSVFEVGFIIIQWYMYGFSLSAIYTCKRDPCPHQVDCFLSRPTEKTIFIWFMLIVSIVSLALNIIELFYVTYKSIKDGIKGKKDPFSATNDAVISGKECGSPKYAYFNGCSSPTAPMSPPGYKLVTGERNPSSCRNYNKQASEQNWANYSAEQNRMGQAGSTISNTHAQPFDFSDEHQNTKKMAPGHEMQPLTILDQRPSSRASSHASSRPRPDDLEI
>protein|name=Cx43_Xenopus-laevis_2
MGDWSALGRLLDKVQAYSTAGGKVWLSVLFIFRILLLGTAVESAWGDEQSAFVCNTQQPGCENVCYDKSFPISHVRFWVLQIIFVSTPTLLYLAHVFYLMRKEEKLNRKEEELKMVQNEGGNVDMHLKQIEIKKFKYGLEEHGKVKMRGGLLRTYIISILFKSVFEVGFIIIQWYMYGFSLSAIYTCKRDPCPHQVDCFLSRPTEKTIFIWFMLIVSIVSLALNIIELFYVTYKSIKDGIKGKKDPFSATNDAVISGKECGSPKYAYFNGCSSPTAPMSPPGYKLVTGERNPSSCRNYNKQASEQNWANYSAEQNRMGQAGSTISNTHAQPFDFSDEHQNTKKMAPGHEMQPLTILDQRPSSRASSHASSRPRPDDLEI
>protein|name=example-peptide
MGTFEEVP
>ligand|name=example-ligand-as-smiles
CC(C)C[C@H](NC(=O)[C@H](CO)NC(=O)[C@@H](N)Cc1ccccc1)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N1CCC[C@H]1C(=O)O
""".strip()
fasta_path = tmp_dir / "example.fasta"
fasta_path.write_text(example_fasta)

# Generate MSAs
msa_dir = tmp_dir / "msas"
msa_dir.mkdir()
protein_seqs = [
    input.sequence
    for input in read_inputs(fasta_path)
    if input.entity_type == EntityType.PROTEIN.value
]
generate_colabfold_msas(protein_seqs=protein_seqs, msa_dir=msa_dir)

# Generate structure
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"./outputs/{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    # 'default' setup
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    device=torch.device("cuda:0"),
    use_esm_embeddings=True,
    msa_directory=msa_dir,
)
cif_paths = candidates.cif_paths
scores = [rd.aggregate_score for rd in candidates.ranking_data]

# Load pTM, ipTM, pLDDTs and clash scores for sample 2
scores = np.load(output_dir.joinpath("scores.model_idx_2.npz"))
