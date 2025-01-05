## Guaranteeing Safety in Medication Recommendation: Is It Possible?

### Our proposed model: **GPSRec**

### Folder Specification
Taking MIMIC-III as an example, the processing method of MIMIC-IV is almost the same.
> - data/: we use the same data set and processing methods as moleRec2023
>> - processing.py: data preprocessing file.  
>> - input/ (extracted from external resources)
>>> - PRESCRIPTIONS.csv: the prescription file from MIMIC-III raw dataset  
>>> - DIAGNOSES_ICD.csv: the diagnosis file from MIMIC-III raw dataset  
>>> - PROCEDURES_ICD.csv: the procedure file from MIMIC-III raw dataset  
>>> - RXCUI2atc4.csv: this is a NDC-RXCUI-ATC4 mapping file, and we only need the RXCUI to ATC4 mapping.
>>> - drug-atc.csv: this is a CID-ATC file, which gives the mapping from CID code to detailed ATC code (we will use the prefix of the ATC code latter for aggregation).
>>> - ndc2RXCUI.txt: NDC to RXCUI mapping file.
>>> - drugbank_drugs_info.csv: drug information table, which is used to map drug name to drug SMILES string.  
>>> - drug-DDI.csv: this a large file, containing the drug DDI information, coded by CID.
>> - output/mimic-iii/
>>> - atc3toSMILES.pkl: drug ID (we use ATC-3 level code to represent drug ID) to drug SMILES string dict
>>> - ddi_A_final.pkl: ddi adjacency matrix
>>> - ehr_adj_final.pkl: if two drugs appear in one set, then they are connected
>>> - records_final.pkl: The final diagnosis-procedure-medication EHR records of each patient, used for train/val/test split
>>> - voc_final.pkl: diag/prod/med index to code dictionary

> - pretrain/: The pre-training model
>> - pretrained_model.model

> - saved/: The trained model
>> - best_model.model

> - src/: source code and training results
>> - main.py: run the main framework and process of the model
>> - model.py: model details and implementation
>> - train.py: training process
>> - util.py: small functions and toolkits
>> - layers.py: the implementation of the multi-gat
>> - graph.py: visualization

## Step 1: Package Dependency  
- Create an environment and install necessary packages
- We conduct our experiments on an Ubuntu 20.04 work-station equipped with 12 CPUs, 40GB memory, and an
NVIDIA RTX 3080 GPU with 10GB of VRAM.
```requirements
dill==0.3.4
matplotlib==3.7.2
numpy==1.23.0
pandas==2.0.3
rdkit_pypi==2022.9.5
scikit_learn==1.2.2
torch==2.0.0+cu118
```
- If you still need another package, please proceed with the installation as described above

## Step 2: Data Processing  
- Due to medical resource protection policies, we do not have the right to share relevant datasets. And if you want to reprocess the data yourself, please download the dataset from the official website and store it in a folder according to the format in the folder specification
- run processing.py
- > The processed data has been provided in this code, and this process can be selectively skipped.
```angular2html
cd data
python processing.py
```

## Step 3: Run the code  
```angular2html
cd src
python main.py
```