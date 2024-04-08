"""
This generates synthetic medical transcripts with GPT-4, using a random real one as an example.
"""

import random
import uuid
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Iterable
import json
import pandas as pd

from loguru import logger


load_dotenv()


class SyntheticMedicalData(BaseModel):
    id: str
    example_transcript: str
    transcript: str
    medications: List[str]
    patient_name: str


gpt4_turbo = AzureChatOpenAI(deployment_name="gpt-4-2")

output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that geterates synthetic data.",
        ),
        (
            "user",
            "Here is an example of a transcript of a patient's visit to the doctor:\n\n{transcript}\n\nPlease generate a completely different synthetic transcript containing at least {num_words} words, representing a patient's visit to the doctor. The patient's name is {patient_name}, and they are taking the medications {medications}. Make sure the transcript references the patient name and the medications (and dosages) they are taking. Return ONLY the synthetic transcript.",
        ),
    ]
)


MEDICATION_COMBINATIONS = [
    ["Ibuprofen"],  # Pain relief, inflammation
    ["Paracetamol", "Codeine"],  # Pain relief
    ["Aspirin", "Clopidogrel"],  # Blood thinning
    ["Lisinopril"],  # Hypertension
    ["Atorvastatin"],  # High cholesterol
    ["Metformin"],  # Type 2 Diabetes
    ["Amlodipine", "Lisinopril"],  # Hypertension
    ["Simvastatin", "Ezetimibe"],  # High cholesterol
    ["Fluticasone", "Salmeterol"],  # Asthma, COPD
    ["Tiotropium"],  # COPD
    ["Levothyroxine"],  # Hypothyroidism
    ["Insulin Glargine", "Metformin"],  # Type 1 and 2 Diabetes
    ["Omeprazole"],  # GERD, Ulcers
    ["Prednisone"],  # Inflammatory conditions
    ["Hydrochlorothiazide", "Lisinopril"],  # Hypertension
    ["Amoxicillin"],  # Bacterial infections
    ["Cephalexin"],  # Bacterial infections
    ["Warfarin", "Aspirin"],  # Blood thinning
    ["Nitroglycerin"],  # Angina
    ["Sildenafil"],  # Erectile dysfunction
    ["Tadalafil", "Finasteride"],  # Erectile dysfunction, BPH
    ["Sertraline"],  # Depression
    ["Venlafaxine", "Mirtazapine"],  # Depression
    ["Lamotrigine", "Valproate"],  # Bipolar disorder
    ["Risperidone"],  # Schizophrenia, Bipolar disorder
    ["Olanzapine", "Fluoxetine"],  # Depression, Bipolar disorder
    ["Clarithromycin", "Amoxicillin", "Omeprazole"],  # H. pylori infection
    ["Furosemide"],  # Heart failure, Edema
    ["Digoxin", "Furosemide", "Spironolactone"],  # Heart failure
    ["Albuterol"],  # Asthma
    ["Budesonide", "Formoterol"],  # Asthma, COPD
    ["Montelukast"],  # Asthma, Allergic rhinitis
    ["Cetirizine"],  # Allergic rhinitis
    ["Naproxen"],  # Pain relief, inflammation
    ["Gabapentin"],  # Neuropathic pain
    ["Pregabalin", "Duloxetine"],  # Neuropathic pain, Fibromyalgia
    ["Trimethoprim/Sulfamethoxazole"],  # Bacterial infections
    ["Azithromycin"],  # Bacterial infections
    ["Doxycycline"],  # Bacterial infections, Acne
    ["Methotrexate", "Folic Acid"],  # Rheumatoid arthritis, Psoriasis
    ["Adalimumab"],  # Rheumatoid arthritis, IBD, Psoriasis
    ["Infliximab", "Methotrexate"],  # Rheumatoid arthritis, IBD
    ["Insulin Aspart", "Insulin Glargine"],  # Diabetes mellitus
    ["Gliclazide", "Metformin"],  # Type 2 Diabetes
    ["Losartan", "Hydrochlorothiazide"],  # Hypertension
    ["Enalapril", "Furosemide"],  # Heart failure, Hypertension
    ["Amiodarone"],  # Arrhythmias
    ["Dabigatran", "Aspirin"],  # Blood thinning, Stroke prevention
    ["Rivastigmine"],  # Alzheimer's disease
    ["Memantine", "Donepezil"],  # Alzheimer's disease
]

PATIENT_NAMES = [
    "Ava Smith",
    "Mohammed Al Farsi",
    "Liam Johnson",
    "Olivia Williams",
    "Juan Martinez",
    "Sophia Chen",
    "Noah Patel",
    "Isabella Garcia",
    "Mia Wang",
    "Ethan Anderson",
    "Aria Kaur",
    "Lucas Rodriguez",
    "Amelia Hernandez",
    "Mason Lee",
    "Harper Gonzalez",
    "Charlotte Brown",
    "Alexander Davis",
    "Ella Kim",
    "James Wilson",
    "Avery Lopez",
    "Benjamin Perez",
    "Lily White",
    "Jacob Taylor",
    "Madison Clark",
    "Michael Lewis",
    "Chloe Walker",
    "Daniel Hill",
    "Emily Jones",
    "Henry Scott",
    "Zoe Green",
    "Jackson Adams",
    "Mila Robinson",
    "Sebastian Young",
    "Luna King",
    "Jack Wright",
    "Layla Moore",
    "Oliver Harris",
    "Scarlett Martin",
    "Elijah Thompson",
    "Sofia Sanchez",
    "Gabriel Ramirez",
    "Isaac Campbell",
    "Alice Mitchell",
    "Logan Carter",
    "Riley Roberts",
    "Carter Phillips",
    "Emma Turner",
    "Aiden Barnes",
    "Grace Murphy",
    "Nathan Stewart",
]


noop_chain = prompt | (lambda x: "N/A")
gpt4_turbo_chain = prompt | gpt4_turbo | output_parser
gpt4_turbo_chain = gpt4_turbo_chain.with_fallbacks([noop_chain])


if __name__ == "__main__":

    # old_df = pd.read_csv("responses/synthetic_results.csv")
    raw_dataset = json.loads(open("data/medical.json").read())
    batch_size = 1
    num_results = 50

    final_results = []

    # Set the random seed
    random.seed(42)

    # Iterate over the dataset in batches
    for i in range(0, num_results, batch_size):
        transcripts = random.sample(MEDICATION_COMBINATIONS, batch_size)

        # Skip if the batch has already been processed and put in final_results
        if all(
            b in [record.medications for record in final_results] for b in transcripts
        ):
            continue

        logger.info(f"Processing batch {i} to {i+batch_size}")

        sample_transcripts = random.sample(raw_dataset, batch_size)
        sample_names = random.sample(PATIENT_NAMES, batch_size)

        batch = [
            {
                "medications": meds,
                "num_words": len(st["transcript"].split()),
                "patient_name": name,
                "transcript": st["transcript"],
            }
            for st, meds, name in zip(sample_transcripts, transcripts, sample_names)
        ]

        # Run the chains on the batch
        results = gpt4_turbo_chain.batch(
            [
                {
                    "transcript": b["transcript"],
                    "medications": b["medications"],
                    "num_words": b["num_words"],
                    "patient_name": b["patient_name"],
                }
                for b in batch
            ]
        )

        data_points = []
        for result, example in zip(results, batch):
            # Create an example dict
            data_point = SyntheticMedicalData(
                id=str(uuid.uuid4()),
                example_transcript=example["transcript"],
                transcript=result,
                medications=example["medications"],
                patient_name=example["patient_name"],
            )

            # logger.info(example)
            logger.info(result)
            logger.info(
                f"The generated transcript has {len(result.split())} words. It was supposed to have {example['num_words']} words."
            )
            data_points.append(data_point)

        final_results.extend(data_points)

        # Write to json
        with open("data/synthetic.json", "w") as f:
            json.dump([r.dict() for r in final_results], f, indent=2)
