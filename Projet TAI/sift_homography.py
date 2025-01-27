import cv2
import numpy as np
import os

# Step 1: Load Dataset Images
def load_dataset_images(folder_path):
    images = []
    image_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                images.append(img)
                image_names.append(filename)
    return images, image_names

# Step 2: SIFT Feature Extraction
def sift(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        print("No descriptors found for an image.")
    return keypoints, descriptors

# Step 3: Descriptor Matching using FLANN
def match_descriptors(desc1, desc2):
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        print("Not enough descriptors to match.")
        return []

    # FLANN parameters
    index_params = dict(algorithm=1, trees=7)
    search_params = dict(checks=70)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform matching
    matches = flann.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)  # Ensure descriptors are float32

    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
    return good_matches

# Step 4: Filter Matches using Homography
def filter_matches_with_homography(matches, keypoints1, keypoints2):
    if len(matches) < 4:  # Minimum matches required for homography
        return []
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return []
    matches_mask = mask.ravel().tolist()
    return [m for i, m in enumerate(matches) if matches_mask[i]]

# Step 5: Find the Most Similar Image
def find_most_similar_image(query_image, dataset_images, image_names):
    query_keypoints, query_descriptors = sift(query_image)
    if query_descriptors is None:
        print("No descriptors found in query image.")
        return None, None, None, None, None

    max_matches = 0
    best_image = None
    best_image_name = None
    best_keypoints = None
    best_matches = None

    for dataset_image, image_name in zip(dataset_images, image_names):
        dataset_keypoints, dataset_descriptors = sift(dataset_image)
        if dataset_descriptors is None:
            continue  # Skip images with no descriptors

        matches = match_descriptors(query_descriptors, dataset_descriptors)
        filtered_matches = filter_matches_with_homography(matches, query_keypoints, dataset_keypoints)
        num_good_matches = len(filtered_matches)

        if num_good_matches > max_matches:
            max_matches = num_good_matches
            best_image = dataset_image
            best_image_name = image_name
            best_keypoints = dataset_keypoints
            best_matches = filtered_matches

    return best_image, best_image_name, best_keypoints, best_matches, query_keypoints

# Step 6: Visualize the Best Match 
def visualize_matches(query_image, query_keypoints, best_image, best_keypoints, matches):
    img_matches = cv2.drawMatches(query_image, query_keypoints, best_image, best_keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Best Match", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main Execution
if __name__ == "__main__":
    dataset_folder = "test2"  
    query_image_path = "query.jpeg"  

    dataset_images, image_names = load_dataset_images(dataset_folder)
    query_image = cv2.imread(query_image_path)

    if query_image is None:
        print("Query image not found.")
    else:
        # Find the most similar image and retrieve all required data
        best_image, best_image_name, best_keypoints, best_matches, query_keypoints = find_most_similar_image(
            query_image, dataset_images, image_names
        )

        if best_image is not None:
            print(f"The most similar image is: {best_image_name}")
            visualize_matches(query_image, query_keypoints, best_image, best_keypoints, best_matches)
        else:
            print("No matching image found.")
