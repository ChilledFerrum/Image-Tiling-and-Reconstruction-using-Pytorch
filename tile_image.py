import torch
import cv2 as cv
import time
from torchvision import transforms
from PIL import Image


def load_image():
    img_path = "Image.jpg"
    return transforms.ToTensor()(Image.open(img_path))


def tile_image(img, tile_size=640):
    start = time.time()
    tiles_list = tile_image_list(img, tile_size)
    end = time.time()
    print(f"Image Tiling Time ({end - start}) Seconds")
    tensor = image_to_tensor(tiles_list)
    return tensor


def image_to_tensor(tiles_list):
    num_rows = len(tiles_list)
    num_cols = len(tiles_list[0])
    channels, height, width = tiles_list[0][0].shape

    tiled_tensor = torch.empty(num_rows, num_cols, channels, height, width)
    for row in range(0, num_rows):
        for col in range(0, num_cols):
            tiled_tensor[row, col] = tiles_list[row][col]
    return tiled_tensor


def tile_image_list(img, tile_size=640):
    channels, height, width = img.shape
    if height <= tile_size and width <= tile_size:
        return [img]

    tiled_images = []
    for y in range(0, height, tile_size):
        tiles_row = []
        tile_y = min(y + tile_size, height)

        for x in range(0, width, tile_size):

            tile_x = min(x + tile_size, width)

            if y + tile_size > height:
                tile_y = height
            if x + tile_size > width:
                tile_x = width
            tile = torch.zeros(channels, tile_size, tile_size)

            tile[:, :tile_y - y, :tile_x - x].copy_(img[:, y:tile_y, x:tile_x])

            tiles_row.append(tile)

            # UNCOMMENT THIS TO VIEW THE TILES
            # cv_img = cv.cvtColor((tile.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'), cv.COLOR_RGB2BGR)
            # cv.imshow('TILE', cv_img)
            # cv.waitKey(0)
        tiled_images.append(tiles_row)

    return tiled_images


def reconstruct(tensor):
    num_rows, num_cols, channels, height, width = tensor.shape
    complete_image = torch.zeros(channels, num_rows * height, num_cols * width)
    for row in range(num_rows):
        for col in range(num_cols):
            current_tile = tensor[row, col]

            start_row = row * height
            end_row = start_row + height
            start_col = col * width
            end_col = start_col + width

            complete_image[:, start_row:end_row, start_col:end_col] = current_tile

    return complete_image


if __name__ == "__main__":
    print("STARTING...")

    img = load_image()  # Tensor image shape:(Channels, height, Width)
    start_start = time.time()

    print("Image Tiling - ")
    start = time.time()
    tensor = tile_image(img, tile_size=640)  # Returns Tensor with Tiled Images
    end = time.time()
    print(f"Image Tiling - Done {end-start}")

    print("Image Reconstruction - ")
    start = time.time()
    tensor = reconstruct(tensor)
    end = time.time()
    print(f"Image Reconstruction - Done {end-start}")

    print(f"Code Finished Images Tiled & Reconstructed at time {end - start_start}")

    print("Viewing final Result...")
    cv_img = cv.cvtColor((tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'), cv.COLOR_RGB2BGR)
    cv.imshow('TILE', cv_img)
    cv.waitKey(0)


















# UwU you didn't expect this message did you
