# Then convert to RGB
def convert_to_rgb(example):
    image = example["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["image"] = image
    return example

def convert_fn(batch):
    texts_list = []

    for q, ans in zip(batch["problem"], batch["solution"]):

        user_text = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
            "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think><answer> answer here </answer>"
        )

        assistant_text = f"Answer: {ans}"

        # Single-element list, with a single dictionary containing both messages
        texts_list.append([
            {
                "user": user_text+". "+q,
                "assistant": assistant_text
            }
        ])

    return {"texts": texts_list}

def extract_prompt(example):
    # safe extraction in case texts is missing or empty
    if example.get('texts') and len(example['texts']) > 0:
        text_item = example['texts'][0]
        # text_item is a list, get the first dict from it
        if isinstance(text_item, list) and len(text_item) > 0:
            return {'prompt': text_item[0].get('user', '')}
    return {'prompt': ''}
