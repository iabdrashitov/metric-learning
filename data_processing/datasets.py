import pandas as pd
from torch.utils.data import Dataset


def get_training_dataset(label_dir, image_dir):

    with open(label_dir, 'r') as file:
        images = []
        class_ids = []
        species = []
        breed_ids = []

        for line in file.readlines():
            line = line.strip("\n")
            image = line.split(" ")[0]
            class_id = line.split(" ")[1]
            specie = line.split(" ")[2]
            breed_id = line.split(" ")[3]

            images.append(image)
            class_ids.append(class_id)
            species.append(specie)
            breed_ids.append(breed_id)

    labels_df = pd.DataFrame()
    labels_df["image"] = pd.Series(images).astype(str)
    labels_df["class_id"] = pd.Series(class_ids).astype(int)
    labels_df["specie"] = pd.Series(species).astype(int)
    labels_df["breed_id"] = pd.Series(breed_ids).astype(int)

    cats_mapping = {}
    dogs_mapping = {}
    dogs = labels_df[labels_df["specie"] == 1]
    cats = labels_df[labels_df["specie"] == 2]
    breed_id_offset = max(dogs["breed_id"].unique())

    for cat in cats["breed_id"].unique():
        cats_mapping[cat] = cat + breed_id_offset

    cats = cats.assign(breed_id=cats["breed_id"].map(cats_mapping))

    df = pd.concat([dogs, cats])
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.drop(columns=["class_id", "specie"])
    df["image"] = df["image"].apply(lambda x: image_dir + "/" + x + ".jpg")
    return df


class TrainDataSet(Dataset):

    def __init__(self, list_image_urls, list_labels):
        self.list_image_urls = list_image_urls
        self.list_labels = list_labels

    def __len__(self):
        return len(self.list_image_urls)

    def __getitem__(self, index):
        y = self.list_labels[index]
        X = self.list_image_urls[index]
        return X, y







