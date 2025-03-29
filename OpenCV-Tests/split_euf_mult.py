import cv2 as cv
import numpy as np

haystack_img = cv.imread("data/splitted_pages-jpg/output-001.jpg", cv.IMREAD_UNCHANGED)
haystack_width = haystack_img.shape[1]

needle_img = cv.imread("data/Questao.jpg", cv.IMREAD_UNCHANGED)
needle_img = cv.cvtColor(needle_img, cv.COLOR_BGRA2BGR)



result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)


threshold = 0.7
locations = np.array(np.where(result >= threshold))

# Filtering for when it gets the same square multiple times
height = needle_img.shape[0]
diffs = np.diff(locations[0])
indices = np.where(diffs > height)[0]
selected_indices = np.concatenate(([0], indices+1))
locations = locations[:, selected_indices]

locations = list(zip(*locations[::-1]))

if locations:
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]
    line_color = (0, 255, 0)
    line_type = cv.LINE_4

    for loc in locations:
        top_left = loc
        bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

        cv.rectangle(haystack_img, top_left, bottom_right, line_color, line_type)
    
    output_path = "output/matched_result.jpg"  # Change the path as needed
    cv.imwrite(output_path, haystack_img)
    # cv.waitKey()