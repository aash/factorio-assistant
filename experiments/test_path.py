import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a binary blob (for example purposes)
# blob_img = np.zeros((500, 500), dtype=np.uint8)
# cv2.circle(blob_img, (250, 250), 100, 255, -1)  # Creating a circular blob
blob_img1 = cv2.imread('c1.png')
blob_img1 = cv2.cvtColor(blob_img1, cv2.COLOR_BGR2GRAY)

def find_and_stitch_polygons(blob_img, erosion_size, area_threshold):
    
    stitched_polygons = []
    
    cblob = blob_img

    while True:
        contours, _ = cv2.findContours(cblob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            print('stop, no contours found')
            break
        
        current_contour = contours[0]
        # Step 2: Approximate the current contour to a polygon
        epsilon = 0.02 * cv2.arcLength(current_contour, True)  # Epsilon can be adjusted
        approx_polygon = cv2.approxPolyDP(current_contour, epsilon, True)
        
        # Add the polygon to the list
        stitched_polygons.append(approx_polygon)

        # Calculate the area and break if it's below the threshold
        area = cv2.contourArea(current_contour)
        print(area)
        if area < area_threshold:
            break
        
        # Step 3: Erode the original image to create a smaller blob
        eroded_blob = cv2.erode(blob_img, np.ones((erosion_size, erosion_size), np.uint8))
        
        # Step 4: Find contours again after erosion
        contours, _ = cv2.findContours(eroded_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check again if any contours are found
        if not contours:
            break
        
        # Update current contour
        current_contour = contours[0]
    
    return stitched_polygons

# Parameters for erosion and area threshold
erosion_size = 50  # Size of erosion
area_threshold = 200  # Minimum area threshold for stopping

# Find and stitch polygons
stitched_polygons = find_and_stitch_polygons(blob_img1, erosion_size, area_threshold)

# Prepare to visualize the results
output_image = cv2.cvtColor(blob_img1, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization

# Draw all stitched polygons
if stitched_polygons:
    for polygon in stitched_polygons:
        cv2.polylines(output_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)

# Show the resulting image with Matplotlib
plt.imshow(output_image)
plt.title("Stitched Polygons on Blob")
plt.axis('off')  # Turn off axis labels
plt.show()