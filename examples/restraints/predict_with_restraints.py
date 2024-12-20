import logging
from pathlib import Path

import torch

from chai_lab.chai1 import run_inference

logging.basicConfig(level=logging.INFO)

example_fasta = """
>protein|7SYZ_1_prot
MMADSKLVSLNNNLSGKIKDQGKVIKNYYGTMDIKKINDGLLDSKILGAFNTVIALLGSIIIIVMNIMIIQNYTRTTDNQALIKESLQSVQQQIKALTDKIGTEIGPKVSLIDTSSTITIPANIGLLGSKISQSTSSINENVNDKCKFTLPPLKIHECNISCPNPLPFREYRPISQGVSDLVGLPNQICLQKTTSTILKPRLISYTLPINTREGVCITDPLLAVDNGFFAYSHLEKIGSCTRGIAKQRIIGVGEVLDRGDKVPSMFMTNVWTPPNPSTIHHCSSTYHEDFYYTLCAVSHVGDPILNSTSWTESLSLIRLAVRPKSDSGDYNQKYIAITKVERGKYDKVMPYGPSGIKQGDTLYFPAVGFLPRTEFQYNDSNCPIIHCKYSKAENCRLSMGVNSKSHYILRSGLLKYNLSLGGDIILQFIEIADNRLTIGSPSKIYNSLGQPVFYQASYSWDTMIKLGDVDTVDPLRVQWRNNSVISRPGQSQCPRFNVCPEVCWEGTYNDAFLIDRLNWVSAGVYLNSNQTAENPVFAVFKDNEILYQVPLAEDDTNAQKTITDCFLLENVIWCISLVEIYDTGDSVIRPKLFAVKIPAQCSES
>protein|7SYZ_2_heavy
QIQLVQSGPELKKPGETVKISCTTSGYTFTNYGLNWVKQAPGKGFKWMAWINTYTGEPTYADDFKGRFAFSLETSASTTYLQINNLKNEDMSTYFCARSGYYDGLKAMDYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRDC
>protein|7SYZ_3_light
DVLMIQTPLSLPVSLGDQASISCRSSQSLIHINGNTYLEWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSHVPFTFGAGTKLELKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNECVY
""".strip()

fasta_path = Path("/tmp/example.fasta")
fasta_path.write_text(example_fasta)

output_dir = Path("/tmp/outputs")

# We provide two example sets of restraints:
# contact.restraints - specifies residue-residue contacts
# pocket.restraints - specifies residue-chain contacts
candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    constraint_path=Path(__file__).with_name("contact.restraints"),
    # 'default' setup
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    device=torch.device("cuda:0"),
    use_esm_embeddings=True,
)
