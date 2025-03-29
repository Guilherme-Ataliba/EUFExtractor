import cv2 as cv
import numpy as np

page_img = cv.imread("data/splitted_pages-jpg/output-001.jpg", cv.IMREAD_UNCHANGED)
page_width = page_img.shape[1]

square_img = cv.imread("data/Items.jpg", cv.IMREAD_UNCHANGED)
square_img = cv.cvtColor(square_img, cv.COLOR_BGRA2BGR)

if page_img is None:
    print("Error: Could not load page_img.")
if square_img is None:
    print("Error: Could not load square_img.")

print(f"page_img shape: {page_img.shape}, dtype: {page_img.dtype}")
print(f"square_img shape: {square_img.shape}, dtype: {square_img.dtype}")


result = cv.matchTemplate(page_img, square_img, cv.TM_CCOEFF)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

quest_w = square_img.shape[1]
quest_h = square_img.shape[0]

max_loc_trans = np.array(max_loc) - np.array([-40, 20])
top_left = max_loc_trans

bottom_right = (top_left[0] + quest_w, top_left[1] + quest_h)
bottom_right_trans = (top_left[0] + quest_w + page_width-600, top_left[1] + quest_h + 120)

# Extract the region of interest (ROI) from the original image
roi = page_img[top_left[1]:bottom_right_trans[1], top_left[0]:bottom_right_trans[0]]
# Save the extracted rectangle as a new image
output_roi_path = "output/extracted_rectangle.jpg"  # Change the path as needed
cv.imwrite(output_roi_path, roi)
print(f"Extracted rectangle saved at: {output_roi_path}")


cv.rectangle(page_img, top_left, bottom_right_trans, color=(0, 255, 0),
             thickness=2, lineType=cv.LINE_4)

# Save the image with a rectangle
output_path = "output/matched_result.jpg"  # Change the path as needed
cv.imwrite(output_path, page_img)
print(f"Image saved at: {output_path}")


# cv.imshow("Result", page_img)
# cv.waitKey()