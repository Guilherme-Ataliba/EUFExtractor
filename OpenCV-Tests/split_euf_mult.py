import cv2 as cv
import numpy as np

def searchImg(haystack_img, needle_img, threshold):
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

    return locations


haystack_img = cv.imread("data/splitted_pages-jpg/output-003.jpg", cv.IMREAD_UNCHANGED)
haystack_width = haystack_img.shape[1]
haystack_height = haystack_img.shape[0]

needle_img = cv.imread("data/Black_square.jpg", cv.IMREAD_UNCHANGED)
needle_img = cv.cvtColor(needle_img, cv.COLOR_BGRA2BGR)
needle_height = needle_img.shape[0]
needle_width = needle_img.shape[1]

questao_img = cv.imread("data/Questao.jpg", cv.IMREAD_UNCHANGED)
questao_img = cv.cvtColor(questao_img, cv.COLOR_BGRA2BGR)

locations_black_square = np.array(searchImg(haystack_img, needle_img, 0.7))
locations_questao = np.array(searchImg(haystack_img, questao_img, 0.7))

locations = locations_black_square + locations_questao

regions = []

for loc_square in locations_black_square:
    for loc_quest in locations_questao:
        if loc_quest[1] > loc_square[1]:
            regions.append([loc_square, loc_quest])
            break

# Adding from the last question to the end of the page
regions.append([locations_black_square[-1], np.array([haystack_width, haystack_height])])

regions

line_color = (0, 255, 0)
line_type = cv.LINE_4

for reg_start, reg_end in regions:
    top_left = (reg_start[0] - needle_width//2, reg_start[1] - needle_height)
    bottom_right = (reg_start[0] + haystack_width, reg_end[1] - needle_height//2)

    cv.rectangle(haystack_img, top_left, bottom_right, line_color, line_type)
    
output_path = "output/matched_result.jpg"  # Change the path as needed
cv.imwrite(output_path, haystack_img)