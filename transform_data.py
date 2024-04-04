from dotenv import load_dotenv
import json

from rich import print

load_dotenv()

ds = json.loads(open("data/MedicalDataset.json").read())
records = ds["train"] + ds["test"]

print(len(records))
print(records[0])

new_records = []
for record in records:
    document = record["document"]
    summary, transcript = document.split("DIALOGUE:")
    summary = summary.replace(
        "\n\n\n---------------------------------------------", ""
    ).strip()
    transcript = transcript.strip()
    new_records.append(
        {
            "id": record["id"],
            "summary": summary,
            "transcript": transcript,
        }
    )

# Write to file
with open("data/medical.json", "w") as f:
    json.dump(new_records, f, indent=2)
