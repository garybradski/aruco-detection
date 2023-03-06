import argparse
import os

import cv2
import numpy as np

depth_scaling = 20  # only because original deapth went 0-10
depth_threshold_far = int(
    3.85 * depth_scaling
)  # Threshold to use on scaled far depth image
depth_threshold_near = int(depth_scaling)  # Threshold to use on near depth image
depth_image = None
x1 = 1
y1 = 1
x2 = 4
y2 = 4
do_median = False


def update_depth_threshold_far(*args):  # Callback to set depth threshold by trackbar
    global depth_threshold_far
    # apply the thresholding
    depth_threshold_far = cv2.getTrackbarPos("Depth_Far", "Depth")


def update_depth_threshold_near(*args):  # Callback to set depth threshold by trackbar
    global depth_threshold_near
    # apply the thresholding
    depth_threshold_near = cv2.getTrackbarPos("Depth_Near", "Depth")


def depth_pixel(
    event, x, y, flags, param
):  # Callback to print out pixels on the depth image
    global depth_image, x1, y1, x2, y2, do_median
    Z = depth_image[y, x]
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
        print("Down value (x,y) at ({}, {}) : [{}]".format(x, y, Z))
    if event == cv2.EVENT_LBUTTONUP:
        x2 = x
        y2 = y
        do_median = True
        print("Up value (x,y) at ({}, {}) : [{}]".format(x, y, Z))


def display_images(directory, waitkey=0):
    global depth_threshold_far, depth_image, depth_threshold_far, x1, y1, x2, y2, do_median
    colorBad = (0, 0, 255)
    colorGood = (0, 255, 0)
    colorBinTooLow = (0, 0, 100)
    colorBinFilling = (100, 0, 100)
    colorBinFilled = (0, 200, 0)
    # Verify directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist")
        exit()

    # Create list of image filenames
    png_files = [
        f
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(".png")
    ]

    # Define a function to extract the last 4 characters of a filename
    def extract_sort_key(filename):
        return filename[-8:-4] + filename[0:3]

    png_files = sorted(png_files, key=extract_sort_key)

    # Handle Windows
    cv2.namedWindow("All")

    cv2.namedWindow("Depth")
    cv2.setMouseCallback("Depth", depth_pixel)

    # create a trackbar to control the threshold value
    cv2.createTrackbar("Depth_Far", "Depth", 0, 200, update_depth_threshold_far)
    cv2.createTrackbar("Depth_Near", "Depth", 0, 200, update_depth_threshold_near)

    # set the trackbar to its initial position
    depth_threshold_far = 77
    depth_threshold_near = 20
    print("far, near {} {}".format(depth_threshold_far, depth_threshold_near))
    cv2.setTrackbarPos("Depth_Far", "Depth", depth_threshold_far)
    cv2.setTrackbarPos("Depth_Near", "Depth", depth_threshold_near)

    resize = True
    # Load and display images five at a time
    for i in range(0, len(png_files), 5):
        # Load images
        images = [
            cv2.imread(os.path.join(directory, png_files[j]))
            for j in range(i, min(i + 5, len(png_files)))
        ]
        # Scale up the depth image (images[1]) from 0-10 to 0-250 for easier viewing
        images[1] *= depth_scaling  # Scale this image up for viewing

        w, h, c = images[0].shape
        rgb_reshaped = cv2.resize(images[3], (h * 2, w * 2))
        # Create canvas to display images on
        canvas = cv2.hconcat(
            [
                cv2.vconcat(
                    [
                        cv2.hconcat([images[2], images[4]]),
                        cv2.hconcat([images[0], images[1]]),
                    ]
                ),
                rgb_reshaped,
            ]
        )
        # canvas = cv2.vconcat([cv2.hconcat([images[2],images[4]]), cv2.hconcat([images[0],images[1]])])

        # Display loop while getting input
        key = ord("d")
        while key != ord("q") or key != ord(" "):
            _, depth_image = cv2.threshold(
                images[1][:, :, 0], depth_threshold_far, 255, cv2.THRESH_TOZERO_INV
            )
            depth_image[depth_image < depth_threshold_near] = 0

            if do_median:
                if x1 != x2 and y1 != y2:
                    subregion = depth_image[y1:y2, x1:x2]
                    # calculate the median of the subregion
                    median = np.median(subregion)
                    print("median = ", median)
                do_median = False

            #  Decide if good
            depth3 = cv2.merge((depth_image, depth_image, depth_image))
            FILL_MAX = 20
            FILL_FULL = 30
            med_bin = int(np.median(depth_image[132:349, 195:429]))
            med_left = int(np.median(depth_image[325:387, 113:305]))
            med_right = int(np.median(depth_image[326:386, 350:509]))
            # Amiga not pulled in left side
            if med_left == 0:
                cv2.rectangle(depth3, (113, 325), (305, 387), colorBad, cv2.FILLED)
            # Amiga not pulled in right side
            if med_right == 0:
                cv2.rectangle(depth3, (350, 326), (509, 386), colorBad, cv2.FILLED)
            # Amiga in!
            if (
                med_left > 0
                and med_left > FILL_MAX
                and med_right > 0
                and med_right > FILL_MAX
            ):
                cv2.rectangle(depth3, (113, 325), (305, 387), colorGood, cv2.FILLED)
                cv2.rectangle(depth3, (350, 326), (509, 386), colorGood, cv2.FILLED)
                # Amiga parked, but bin too low
                if med_bin == 0:
                    cv2.rectangle(
                        depth3, (195, 132), (429, 349), colorBinTooLow, cv2.FILLED
                    )
                # Amiga filling
                elif med_bin > FILL_FULL:
                    cv2.rectangle(
                        depth3, (195, 132), (429, 349), colorBinFilling, cv2.FILLED
                    )
                # Amiga full
                else:  # med_bin >= FILL_FULL:
                    cv2.rectangle(
                        depth3, (195, 132), (429, 349), colorBinFilled, cv2.FILLED
                    )

            # Display images
            cv2.imshow("All", canvas)
            cv2.imshow("Depth", depth3)

            key = cv2.waitKey(waitkey)
            if key == ord("q") or key == ord(" "):
                break
            if resize:
                cv2.resizeWindow("All", 1280, 800)  # Width height
                resize = False
        if key == ord("q"):
            break

    # Clean up
    cv2.destroyAllWindows()


def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Reads in a directory of .png images and displays them five at a time"
    )
    parser.add_argument("directory", type=str, help="Path to directory of .png images")
    parser.add_argument(
        "--waitkey",
        type=int,
        default=0,
        help="Number of milliseconds to wait for key press after displaying each set of five images",
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Call function to display images
    display_images(args.directory, args.waitkey)


if __name__ == "__main__":
    main()
