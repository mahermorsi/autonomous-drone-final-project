from PIL import Image


def convert_non_black_to_white(image_path, output_path='masked_image.png'):
    img = Image.open(image_path)
    # Convert to grayscale
    img = img.convert('L')
    img = img.point(lambda p: 255 if p > 7 else 0)
    img.save(output_path)
    return output_path
