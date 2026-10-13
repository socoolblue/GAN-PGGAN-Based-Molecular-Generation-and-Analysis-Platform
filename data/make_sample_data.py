"""샘플 데이터 생성 스크립트.

수계(aqueous) 전해질 / 리튬염 / 용매·첨가제 관련 공개 분자들을 SMILES 로 모아
`sample_electrolytes.csv` 를 만든다. 실제 연구 데이터가 준비되면 같은 형식
(smiles, name, category, target) 의 CSV 로 교체하면 된다.

* target 열 : 측정값(예: 이온전도도, 전기화학 안정창 등). 없으면 비워 둔다.
"""
import csv
from pathlib import Path

from rdkit import Chem

RAW = [
    # ---- 리튬염 음이온 (anion) ----
    ("FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F", "TFSI", "anion"),
    ("FS(=O)(=O)[N-]S(=O)(=O)F", "FSI", "anion"),
    ("FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)F", "FTFSI", "anion"),
    ("FC(F)(F)C(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)C(F)(F)F", "BETI", "anion"),
    ("FC(F)(F)S(=O)(=O)[O-]", "OTf (triflate)", "anion"),
    ("F[P-](F)(F)(F)(F)F", "PF6", "anion"),
    ("F[B-](F)(F)F", "BF4", "anion"),
    ("[O-]Cl(=O)(=O)=O", "ClO4", "anion"),
    ("[O-][N+](=O)[O-]", "NO3", "anion"),
    ("O=C1O[B-]2(OC1=O)OC(=O)C(=O)O2", "BOB", "anion"),
    ("O=C1O[B-](F)(F)OC1=O", "DFOB", "anion"),
    ("FC(F)(F)C(=O)[O-]", "TFA", "anion"),
    ("CS(=O)(=O)[O-]", "mesylate", "anion"),
    ("[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "nonaflate", "anion"),
    ("N#C[N-]C#N", "DCA (dicyanamide)", "anion"),
    ("N#CC(C#N)=[N-]", "TCM-like", "anion"),
    ("FC(F)(F)S(=O)(=O)[N-]C#N", "TFSAM", "anion"),
    ("FC(F)(F)C(=O)[N-]S(=O)(=O)C(F)(F)F", "TSAC", "anion"),
    ("CC(=O)[O-]", "acetate", "anion"),
    ("OC(=O)C(O)C(O)C(=O)[O-]", "tartrate", "anion"),
    ("[O-]C(=O)C(F)(F)C(F)(F)C(=O)[O-]", "tetrafluorosuccinate", "anion"),
    ("FC(F)(F)C(F)(F)C(F)(F)C(=O)[O-]", "heptafluorobutyrate", "anion"),
    ("[O-]S(=O)(=O)CC(F)(F)F", "trifluoroethanesulfonate", "anion"),
    ("FC(F)(F)S(=O)(=O)[C-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F", "TFSM", "anion"),
    ("O=S1(=O)[N-]S(=O)(=O)C(F)(F)C(F)(F)1", "cyclic-TFSI (CTFSI)", "anion"),
    ("FC(F)(F)C(F)(F)C(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)F", "BHFSI", "anion"),
    # ---- 이온성 액체 양이온 / 리튬 ----
    ("CCn1cc[n+](C)c1", "EMIM", "cation"),
    ("CCCCn1cc[n+](C)c1", "BMIM", "cation"),
    ("CCCC[N+]1(C)CCCC1", "Pyr14", "cation"),
    ("CCC[N+]1(C)CCCC1", "Pyr13", "cation"),
    ("CCCC[N+]1(C)CCCCC1", "Pip14", "cation"),
    ("CC[N+](C)(C)CCOC", "DEME-like", "cation"),
    ("C[N+](C)(C)C", "TMA", "cation"),
    ("CC[N+](CC)(CC)CC", "TEA", "cation"),
    ("COCC[N+]1(C)CCCC1", "Pyr1,2O1", "cation"),
    ("[Li+]", "Li+", "cation"),
    # ---- 용매 / 첨가제 / 고분자 단위 ----
    ("O", "water", "solvent"),
    ("O=C1OCCO1", "EC", "solvent"),
    ("CC1COC(=O)O1", "PC", "solvent"),
    ("COC(=O)OC", "DMC", "solvent"),
    ("CCOC(=O)OC", "EMC", "solvent"),
    ("CCOC(=O)OCC", "DEC", "solvent"),
    ("O=C1OCC(F)O1", "FEC", "additive"),
    ("O=C1OC=CO1", "VC", "additive"),
    ("COCCOC", "DME", "solvent"),
    ("COCCOCCOC", "diglyme", "solvent"),
    ("COCCOCCOCCOC", "triglyme", "solvent"),
    ("COCCOCCOCCOCCOC", "tetraglyme", "solvent"),
    ("O=S1(=O)CCCC1", "sulfolane", "solvent"),
    ("CC#N", "acetonitrile", "solvent"),
    ("CS(C)=O", "DMSO", "solvent"),
    ("CN(C)C=O", "DMF", "solvent"),
    ("OCCO", "ethylene glycol", "solvent"),
    ("OCC(O)CO", "glycerol", "solvent"),
    ("OCCOCCO", "diethylene glycol", "solvent"),
    ("OCCOCCOCCO", "triethylene glycol", "solvent"),
    ("C1CCOC1", "THF", "solvent"),
    ("CC1CCOC1", "2-MeTHF", "solvent"),
    ("C1COCO1", "1,3-dioxolane", "solvent"),
    ("CC(=O)OCC", "ethyl acetate", "solvent"),
    ("CCOC(=O)CF", "fluoroethyl acetate-like", "solvent"),
    ("FC(F)(F)COC(F)(F)C(F)F", "HFE (TTE)", "solvent"),
    ("FC(F)(F)C(F)(F)COCC(F)(F)C(F)(F)F", "HFE-2", "solvent"),
    ("O=P(OC)(OC)OC", "TMP", "additive"),
    ("O=P(OCC)(OCC)OCC", "TEP", "additive"),
    ("CC(=O)N(C)C", "DMA", "solvent"),
    ("CN1CCCC1=O", "NMP", "solvent"),
    ("CC(C)=O", "acetone", "solvent"),
    ("CO", "methanol", "solvent"),
    ("CCO", "ethanol", "solvent"),
    ("O=C(N)N", "urea", "additive"),
    ("CN(C)C(N)=O", "DMU", "additive"),
    ("OC(=O)CO", "glycolic acid", "additive"),
    ("NCCO", "ethanolamine", "additive"),
    ("OCCN(CCO)CCO", "triethanolamine", "additive"),
    ("C=CC(=O)OCCOC", "methoxyethyl acrylate (PEGA unit)", "polymer"),
    ("C=CC(=O)OCCOCCOC", "PEG-acrylate unit", "polymer"),
    ("COCCOCCOCCOCCOCCOC", "pentaglyme", "polymer"),
    ("C=CC(=O)N", "acrylamide (hydrogel unit)", "polymer"),
    ("C=CC(=O)O", "acrylic acid (hydrogel unit)", "polymer"),
    ("OCC(O)CO.O", "glycerol hydrate-like", "polymer"),
    ("C=CO", "vinyl alcohol (PVA unit)", "polymer"),
    ("CC(C)(C)CC(=O)O", "neo-acid", "additive"),
    ("OS(=O)(=O)O", "sulfuric acid", "additive"),
    ("OP(=O)(O)O", "phosphoric acid", "additive"),
    ("OB(O)O", "boric acid", "additive"),
    ("O=C(O)C(F)(F)F", "TFA acid", "additive"),
    ("FC(F)(F)S(=O)(=O)O", "triflic acid", "additive"),
    ("CS(=O)(=O)O", "MSA", "additive"),
    ("OCC(F)(F)F", "trifluoroethanol", "solvent"),
    ("FC(F)C(F)(F)CO", "tetrafluoropropanol", "solvent"),
    ("OCCS(=O)(=O)O", "isethionic acid", "additive"),
    ("NCCS(=O)(=O)O", "taurine", "additive"),
    ("OCCN1CCOCC1", "hydroxyethyl morpholine", "additive"),
    ("O=C1CCCCN1", "valerolactam", "solvent"),
    ("O=C1CCCO1", "GBL", "solvent"),
    ("O=C1CCCCO1", "delta-valerolactone", "solvent"),
    ("CC(O)C(=O)O", "lactic acid", "additive"),
    ("OC(=O)CCC(=O)O", "succinic acid", "additive"),
    ("OC(=O)C(O)C(O)C(O)C(O)CO", "gluconic acid", "additive"),
    ("OCC1OC(O)C(O)C(O)C1O", "glucose", "additive"),
    ("OC(=O)CS(=O)(=O)O", "sulfoacetic acid", "additive"),
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)O", "PFBS acid", "additive"),
    ("CCOC(=O)C(F)(F)F", "ethyl trifluoroacetate", "solvent"),
    ("COC(=O)C(F)(F)F", "methyl trifluoroacetate", "solvent"),
    ("FC(F)(F)C(=O)OCC(F)(F)F", "TFE-TFA", "solvent"),
    ("CC(C)(C)OC(=O)N", "Boc-amine", "additive"),
]


def main(out: Path):
    rows, seen = [], set()
    for smi, name, cat in RAW:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print("invalid, skipped:", smi, name)
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        rows.append({"smiles": can, "name": name, "category": cat, "target": ""})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["smiles", "name", "category", "target"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} molecules -> {out}")


if __name__ == "__main__":
    main(Path(__file__).with_name("sample_electrolytes.csv"))
