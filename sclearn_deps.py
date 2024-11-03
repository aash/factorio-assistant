import numpy as np
from sklearn.cluster import KMeans

def get_dominant_colors(image, k=5):
    # Convert the image to RGB (OpenCV loads images in BGR by default)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image into a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Apply KMeans to find the top k colors
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    
    # The cluster centers are our dominant colors.
    colors = kmeans.cluster_centers_

    return colors, kmeans.labels_

def recreate_image(centroids, labels, w, h):
    '''Recreate the (compressed) image from the cluster centers & labels'''
    d = centroids.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = centroids[labels[label_idx]]
            label_idx += 1
    return image




def get_reduced_img(img1, k = 6):
    dominant_colors, labels = get_dominant_colors(img1, k)

    # Recreate the image using the dominant colors
    w, h, _ = img1.shape
    new_image = recreate_image(dominant_colors, labels, w, h)

    # Convert the image back to BGR
    # new_image_bgr = cv2.cvtColor(new_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return new_image.astype(np.uint8)
