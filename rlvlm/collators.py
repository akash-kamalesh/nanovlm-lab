import torch


class BaseCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def prepare_batch(self, batch, max_length=None):
        # 1) Handle empty
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # 2) Drop None rows
        batch = [s for s in batch if s is not None]
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # batch is a list of dicts, each containing "input_ids", "attention_mask", "labels", "images"
        # let's convert it to a dict of lists of tensors
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        # Truncate sequences that exceed max_length
        if max_length is not None:
            batch = self._truncate_samples(batch, max_length)

        if len(batch["input_ids"]) == 0:
            return batch

        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        self._pad_batch(batch, max_len) #  dictionaries in Python are mutable and passed by reference

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }

    def _truncate_samples(self, batch, max_length):
        """Truncate sequences to max_length instead of discarding them."""
        truncated = [
            (ids[:max_length], label[:max_length], attn[:max_length], img)
            for ids, label, attn, img in zip(batch["input_ids"], batch["labels"], batch["attention_mask"], batch["images"])
        ]
        if not truncated:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}
        batch_token_ids, batch_labels, batch_attentions, batch_images = zip(*truncated)
        return {"input_ids": list(batch_token_ids), "labels": list(batch_labels), "attention_mask": list(batch_attentions), "images": list(batch_images)}


class VQACollator(BaseCollator):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)

    def _pad_batch(self, batch, max_length):  # Reimplementing to use -100 as the pad value for labels, so that it's ignored by the loss
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def __call__(self, batch):
        batch = self.prepare_batch(batch, max_length=self.max_length)
        return batch
