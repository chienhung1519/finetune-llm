from datasets import load_dataset

class DatasetModule:

    def __init__(self, dataset_path, dataset_text_field="text", max_seq_length=512):
        self.dataset = self.load_dataset(dataset_path)
        self.dataset = self.process(self.dataset, dataset_text_field, max_seq_length)

    def load_dataset(self, dataset_path):
        return load_dataset(dataset_path)
    
    def process(self, dataset, dataset_text_field="text", max_seq_length=512):
        dataset = dataset.map(
            lambda x: self.tokenizer(x[dataset_text_field], truncation=True, padding="max_length", max_length=max_seq_length),
            batched=True,
        )
        return dataset