from PIL import Image

def ensure_rgb(example):
    # Convert the image to RGB if it's not already
    image = example['image']
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        example['image'] = image
    return example
