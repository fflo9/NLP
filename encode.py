from datasets import load_dataset
import pandas as pd

#load medmentions dataset
dataset = load_dataset(
    "bigbio/medmentions",
    "medmentions_st21pv_bigbio_kb",
    trust_remote_code=True
)
#define mapping 
label_map= {
    "T098":"person",
    "T097":"person",
    "T091":"organization",
    "T092":"organization",
    "T082":"location",
    "T033":"sympthom",
    "T017":"sympthom",
    "T022":"sympthom",
    "T031":"sympthom",
    "T037":"history",
    "T201":"history",
    "T038":"history",
    "T005":"history",
    "T007":"history",
    "T103":"history",
    "T058":"action",
    "T062":"action",
    "T074":"action",
    "T204":"other",
    "T170":"other",
    "T168":"other"
}
#function to map from type to label
def map_entity(ent):
    original_type = ent["type"]
    new_type = label_map[original_type]
    return {
        **ent,  # keep all existing key-value pairs
        "label": new_type
    }
#function to encode everything in the dataset
def encode(item):
    item["entities"] = [map_entity(ent) for ent in item["entities"]]
    return item
#apply encode to dataset
dataset = dataset.map(encode)

for split in ["train", "validation", "test"]:
    df = pd.DataFrame(dataset[split])
    df.to_csv(f"{split}.csv", index=False)
