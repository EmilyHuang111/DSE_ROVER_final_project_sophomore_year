# Importing necessary packages and modules
import pandas as pd
import numpy as np
import cv2
import math



def find_contours(image, og, extra_pix=0):
   # Find and process contours on the image, drawing midline between two largest contours
   # Find the contours on the image
   contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   contours_list = list(contours)
   # Sort the list of contours by the contour area
   contours_list.sort(key=cv2.contourArea)


   if len(contours_list) > 1:
       # Get the 2 largest contours
       c1 = contours_list[-1]
       c2 = contours_list[-2]


       # Fit polylines to each contour
       outline1 = cv2.approxPolyDP(c1, 4, True)
       cv2.drawContours(image, [outline1], -1, (0, 255, 255), 15)


       outline2 = cv2.approxPolyDP(c2, 4, True)
       cv2.drawContours(image, [outline2], -1, (0, 255, 255), 15)


       # Draw a midline by averaging coordinates of corresponding points
       midline = []
       for pt1, pt2 in zip(outline1[:int(len(outline1) / 1.8)], outline2[:int(len(outline2) / 1.8)]):
           mid_x = int((pt1[0][0] + pt2[0][0]) / 2) + extra_pix
           mid_y = int((pt1[0][1] + pt2[0][1]) / 2)
           midline.append([[mid_x, mid_y]])


       midline = np.array(midline, dtype=np.int32)


       # Draw the midline on the original image
       cv2.polylines(og, [midline], False, (0, 255, 0), 15)
       return midline
   cv2.imshow("mid", og)




def apply_color_mask(image, lower, upper):
   # Apply color mask to the image and return the masked image
   mask = cv2.inRange(image, lower, upper)
   masked = cv2.bitwise_and(image, image, mask=mask)
   return masked




def apply_filtering(image):
   # Apply filtering to the image to detect edges
   mask2 = apply_color_mask(image, (50, 50, 60), (250, 250, 250))
   image = image - mask2
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gauss = cv2.GaussianBlur(gray, (15, 15), 0)
   gauss = cv2.medianBlur(gauss, 15)
   edges = cv2.Canny(gauss, 30, 60)
   cv2.imshow("Filtered Image", edges)
   return edges




def unwarp_image(img, mask_vert, screen_vert):
   # Unwarp an image given a set of vertices
   matrix = cv2.getPerspectiveTransform(screen_vert, mask_vert)
   result = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
   return result




def warp_image(image, mask_vert, screen_vert):
   # Warp an image given a set of vertices
   matrix = cv2.getPerspectiveTransform(mask_vert, screen_vert)
   result = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
   return result




def process_image(image):
   height, width = image.shape[:2]  # Get the height and width of the image


   # Define vertices for the trapezoidal mask
   p1 = [round(width * 0.1), round(height * 1)]
   p2 = [round(width * 0.22), round(height * 0.28)]
   p3 = [round(width * 0.79), round(height * 0.28)]
   p4 = [round(width * 0.90), round(height * 1)]


   # Create a trapezoidal mask around the road
   mask_vertices = np.int32([p1, p2, p3, p4])


   # Define screen vertices for perspective transform
   screen_verts = np.float32([[0, height], [0, 0], [width, 0], [width, height]])


   # Warp the frame to fit this trapezoidal mask to get a bird's-eye view of the road
   warped_image = warp_image(image, np.float32(mask_vertices), screen_verts)


   # Apply filtering to the warped image
   filtered = apply_filtering(warped_image)


   # Crop the filtered image into left and right halves
   crop_l = filtered[0:height, 0:width // 2]
   cv2.imshow("cropl", crop_l)
   crop_r = filtered[0:height, width // 2:width]
   cv2.imshow("cropr", crop_r)


   # Find contours in the left and right halves
   left_contours = find_contours(crop_l, warped_image)
   right_contours = find_contours(crop_r, warped_image, width // 2)


   middle = []  # List to store the midpoints between left and right contours
   scale_factors = []  # List to store scale factors for midpoints
   max_x = width // 2
   max_y = 0


   if left_contours is not None and right_contours is not None:
       for x in range(len(left_contours)):
           try:
               scale_factor = int((left_contours[x][0][0] + right_contours[x][0][0]) / 2) / int(
                   (left_contours[x][0][0]))
               middle.append([int((left_contours[x][0][0] + right_contours[x][0][0]) / 2),
                              int((left_contours[x][0][1] + right_contours[x][0][1]) / 2)])
               scale_factors.append(scale_factor)
           except:
               if scale_factors:
                   avg_scale = sum(scale_factors) / len(scale_factors)
                   middle.append([int(left_contours[x][0][0] * avg_scale), int(left_contours[x][0][1] * avg_scale)])
               else:
                   break


       for point in middle:
           if point[1] > max_y:
               max_x = point[0]


       middle = np.array(middle, dtype=np.int32)
       # Draw the midline on the warped image
       cv2.polylines(warped_image, [middle], False, (0, 255, 255), 15)


   # Unwarp the warped image back to the original perspective
   unwarped_image = unwarp_image(warped_image, np.float32(mask_vertices), screen_verts)
   cv2.imshow("unwarped", unwarped_image)


   # Blend the unwarped image with the original image
   finished = cv2.addWeighted(image, 0.5, unwarped_image, 0.5, 0.0)
   cv2.imshow("finished", finished)


   return max_x




# Open video capture from the specified file
vid = cv2.VideoCapture("IMG_2988.MOV")


while True:
   ret, frame = vid.read()  # Read a frame from the video
   if frame is None:
       break  # Exit the loop if no frame is read



   mid_x = process_image(frame)  # Process the frame to find the midline


   # Determine the direction based on the position of the midline
   width = frame.shape[1]
   if mid_x < (width // 2 -10):
       print("turn left")
	   left()
   elif mid_x > (width // 2 +10):
       print("RIGHT")
	   right()
   else:
       print("forward")
	   forward()


   if cv2.waitKey(15) & 0xFF == ord('q'):
       break  # Exit the loop if 'q' key is pressed


vid.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows



