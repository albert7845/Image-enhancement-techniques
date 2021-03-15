import cv2
import numpy as np

img=cv2.imread(r'image1.pgm')
img=cv2.GaussianBlur(img, (3, 3), 0)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("img_gray", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

cv2.imshow("img_bin", img_bin)
cv2.waitKey(0)
cv2.destroyAllWindows()

output = cv2.connectedComponents(img_bin, connectivity=8, ltype=cv2.CV_32S)
print(ret)

num_labels = output[0]
print(num_labels)
labels = output[1]

# form colors
colors = []
for i in range(num_labels):
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    colors.append((b, g, r))
colors[0] = (0, 0, 0)

img_bin2=img_bin
cv2.imwrite("bin.png", img_bin)
cv2.bitwise_not(img_bin,img_bin2)
cv2.imwrite("bin2.png", img_bin2)



# make connect-8 graph

h2_size, w2_size = img_bin2.shape
img_conn = np.zeros((h2_size, w2_size, 3), dtype=np.uint8)
num_conn = 1
for row in range(h2_size):
    for col in range(w2_size):
        # if the point is white point begin our search
      if ((img_bin2[row,col] == 0) and (img_conn[row,col,0]== 0 ) and (img_conn[row,col,1]== 0 ) and (img_conn[row,col,2]== 0 )):
        color=[np.random.randint(1, 256),np.random.randint(1, 256),np.random.randint(1, 256)]
        list = [[row, col]]
        second_moments = []
        # add the only point to our list as a start
        while len(list) > 0:

            point = list.pop()
            second_moments.append([point[0], point[1]])
            # do a connect-8 search
            if (img_bin2[    point[0] - 1, point[1] - 1] == 0) and (img_conn[point[0] - 1, point[1] - 1, 0] == 0):
                list.append([point[0] - 1, point[1] - 1])
            if (img_bin2[    point[0],     point[1] - 1] == 0) and (img_conn[point[0],     point[1] - 1, 0] == 0):
                list.append([point[0],     point[1] - 1])
            if (point[0]+1<h2_size) and (img_bin2[    point[0] + 1, point[1] - 1] == 0) and (img_conn[point[0] + 1, point[1] - 1, 0] == 0):
                list.append([point[0] + 1, point[1] - 1])
            if (img_bin2[    point[0] - 1, point[1]] == 0)     and (img_conn[point[0] - 1, point[1], 0] == 0):
                list.append([point[0] - 1, point[1]])
            if (point[0]+1<h2_size) and (img_bin2[    point[0] + 1, point[1]] == 0)     and (img_conn[point[0] + 1, point[1], 0] == 0):
                list.append([point[0] + 1, point[1]])
            if (point[1]+1<w2_size) and (img_bin2[    point[0] - 1, point[1] + 1] == 0) and (img_conn[point[0] - 1, point[1] + 1, 0] == 0):
                list.append([point[0] - 1, point[1] + 1])
            if (point[1]+1<w2_size) and(img_bin2[    point[0],     point[1] + 1] == 0) and (img_conn[point[0],     point[1] + 1, 0] == 0):
                list.append([point[0],     point[1] + 1])
            if (point[0]+1<h2_size) and (point[1]+1<w2_size) and(img_bin2[    point[0] + 1, point[1] + 1] == 0) and (img_conn[point[0] + 1, point[1] + 1, 0] == 0):
                list.append([point[0] + 1, point[1] + 1])
            #draw the colors
            img_conn[point[0], point[1], 0] = color[0]
            img_conn[point[0], point[1], 1] = color[1]
            img_conn[point[0], point[1], 2] = color[2]

        #calculate center point
        second_moments_mean=[0,0]
        second_moments_mean=np.mean(second_moments,axis=0)
        #print(second_moments_mean)
        sum_urc=0
        sum_ucc = 0
        sum_urr = 0
        for point in second_moments :
          sum_urc=sum_urc+np.absolute((point[0]-second_moments_mean[0])*(point[1]-second_moments_mean[1]))
          sum_ucc=sum_ucc+np.absolute((point[1]-second_moments_mean[1])*(point[1]-second_moments_mean[1]))
          sum_urr=sum_urr+np.absolute((point[0]-second_moments_mean[0])*(point[0]-second_moments_mean[0]))

        print('sum_urc for label ',num_conn,"is",sum_urc/len (second_moments))
        print('sum_ucc for label ',num_conn,"is",sum_ucc/len (second_moments))
        print('sum_urr for label ',num_conn,"is",sum_urr/len (second_moments))
        num_conn = num_conn +1
cv2.imwrite("conn.png", img_conn)

# draw connect-8 graph
h, w = img_gray.shape
image = np.zeros((h, w, 3), dtype=np.uint8)
for row in range(h):
    for col in range(w):
        image[row, col] = colors[labels[row, col]]

cv2.imshow("colored labels", image)
cv2.imwrite("labels.png", image)
print("num_labels : ", num_labels - 1)

print("num_conn : ", num_conn - 1)
cv2.waitKey(0)
cv2.destroyAllWindows()

def canny_demo(image):
    t = 80
    canny_output = cv2.Canny(image, t, t * 2)
    cv2.imshow("canny_output", canny_output)
    cv2.imwrite("canny_output.png", canny_output)
    return canny_output

def compute_circularity(area, arclen):
    return pow(arclen, 2) / area

cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
cv2.imshow("input", img)
binary = canny_demo(img)
k = np.ones((3, 3), dtype=np.uint8)

# find the contour of the image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    area = cv2.contourArea(contours[c]) # calculate the area
    arclen = cv2.arcLength(contours[c], True) #calculate the arclength ， True means closing area
    if (area!=0 and area>10):
        circularity = compute_circularity(area, arclen) #calculate C1
        print('Object{}:'.format(c+1))
        print('Circularity:'+str(circularity)+' Area:'+str(area))
        rect = cv2.minAreaRect(contours[c])
        cx, cy = rect[0]
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,0,255),2)
        cv2.circle(img, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)



# show the result of contour search
cv2.imshow("contours_analysis", img)
cv2.imwrite("contours_analysis.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



