import cv2


# path = "/media/server/HDD1/zpf/otherNet/2022-CVPR-SegMaR-4/62-74/"
# img = cv2.imread(path + "animal-74-700.png")
# cv2.imshow("111",img)
# cv2.waitKey(0)
# new = cv2.resize(img,(933,678))
# # h w
# print(img.shape)
# print(new.shape)
# cv2.imwrite(path + "animal-74-678.png",new)


path = "/media/server/HDD1/zpf/otherNet/2022-CVPR-SegMaR-4/62-74/"
img = cv2.imread(path + "animal-62-1000.png")
cv2.imshow("111",img)
cv2.waitKey(0)
new = cv2.resize(img,(998,628))
# h w
print(img.shape)
print(new.shape)
cv2.imwrite(path + "animal-62-998.png",new)