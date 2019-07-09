import cv2
path="/home/student/VireoFood172/118/13NV313W0043521.jpg"
image=cv2.imread(path)
print(image)
image = cv2.resize(image, (256, 256))

