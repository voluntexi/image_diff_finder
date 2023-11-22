import cv2
import numpy as np
def split_image(img):
    height, width, _ = img.shape
    center_line = width // 2
    img1 = img[:, :center_line, :]
    img2 = img[:, center_line:, :]

    return img1, img2

def merge_overlapping_rectangles(rectangles):
    flags = np.zeros(len(rectangles))

    # Iterate through rectangles
    for i in range(len(rectangles)):
        # Skip rectangles that are already merged or contained within another
        if flags[i] == 1:
            continue

        # Iterate through remaining rectangles
        for j in range(1, len(rectangles)):
            if j == i:
                continue
            # Skip rectangles that are already merged or contained within another
            if flags[j] == 1:
                continue

            # Check for overlap
            if (
                    rectangles[i][0] < rectangles[j][0] + rectangles[j][2] and
                    rectangles[i][0] + rectangles[i][2] > rectangles[j][0] and
                    rectangles[i][1] < rectangles[j][1] + rectangles[j][3] and
                    rectangles[i][1] + rectangles[i][3] > rectangles[j][1]
            ):
                # Merge rectangles
                rectangles[i] = (
                    min(rectangles[i][0], rectangles[j][0]),
                    min(rectangles[i][1], rectangles[j][1]),
                    max(rectangles[i][0] + rectangles[i][2], rectangles[j][0] + rectangles[j][2]) - min(
                        rectangles[i][0], rectangles[j][0]),
                    max(rectangles[i][1] + rectangles[i][3], rectangles[j][1] + rectangles[j][3]) - min(
                        rectangles[i][1], rectangles[j][1])
                )

                # Mark the merged rectangle
                flags[j] = 1

            # Check if rectangle i is completely contained within rectangle j
            elif (
                    rectangles[i][0] >= rectangles[j][0] and
                    rectangles[i][1] >= rectangles[j][1] and
                    rectangles[i][0] + rectangles[i][2] <= rectangles[j][0] + rectangles[j][2] and
                    rectangles[i][1] + rectangles[i][3] <= rectangles[j][1] + rectangles[j][3]
            ):
                # Mark the contained rectangle
                flags[i] = 1

    # Return the merged rectangles
    return [rect for i, rect in enumerate(rectangles) if flags[i] == 0]

def matchAB(imgA, imgB):
    global max, min

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    height, width = grayA.shape
    window_size = 600
    result_window = np.zeros((height, width), dtype=imgA.dtype)
    for start_y in range(0, height-window_size, height//10):
        for start_x in range(0, width-window_size, width//5):
            window = grayA[start_y:start_y + window_size, start_x:start_x + window_size]
            match = cv2.matchTemplate(grayB, window, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(match)
            matched_window = grayB[max_loc[1]:max_loc[1] + window_size, max_loc[0]:max_loc[0] + window_size]
            result = cv2.absdiff(window, matched_window)
            result_window[start_y:start_y + window_size, start_x:start_x + window_size] = result
        print(start_y)
    _, result_window_bin = cv2.threshold(result_window, 45, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(result_window_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgC = imgA.copy()
    rectangles = []  # Store rectangles as (x, y, w, h)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            min_val = np.nanmin(contour, 0)
            max_val = np.nanmax(contour, 0)
            loc1 = (min_val[0][0], min_val[0][1])
            loc2 = (max_val[0][0], max_val[0][1])

            width = max_val[0][0] - min_val[0][0]
            height = max_val[0][1] - min_val[0][1]
            if width and height:
                rectangles.append((loc1[0], loc1[1], loc2[0] - loc1[0], loc2[1] - loc1[1]))
                # aspect_ratio = width / height
                # # if 0.75 <= aspect_ratio <= 1.5:
                # cv2.rectangle(imgC, loc1, loc2, 255, 4)
                #
    rectangles.sort(key=lambda x: x[0], reverse=False)
    merged_rectangles = merge_overlapping_rectangles(rectangles)
    for i, rect in enumerate(merged_rectangles):
        x, y, w, h = rect
        cv2.rectangle(imgC, (x, y), (x + w, y + h), 255, 4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_position = (x + 10, y + 30)
        cv2.putText(imgC, str(i + 1), text_position, font, font_scale, 255, font_thickness, cv2.LINE_AA)

    # cv2.imwrite("./img/result.jpg", imgC)
    return imgC
def start(img):
    img1, img2 = split_image(img)
    return matchAB(img1, img2)
