from PIL import Image


def filter_colors(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to RGB mode (in case it's in a different mode)
    image = image.convert("RGB")

    # Get the width and height of the image
    width, height = image.size

    # Iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            r, g, b = image.getpixel((x, y))

            # Check if the pixel is white (or close to white)
            if r > 5 and g > 5 and b > 5:
                # Set the pixel to white
                image.putpixel((x, y), (255, 255, 255))
            # else:
            #     # Set the pixel to black
            #     image.putpixel((x, y), (0, 0, 0))

    # Save the filtered image
    filtered_image_path = "masked_image.png"
    image.save(filtered_image_path)
    return filtered_image_path

# Provide the path to your input image
#input_image_path = "objects.jpeg"

# Call the function to filter the colors
#filter_colors(input_image_path)


