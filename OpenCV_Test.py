import os

import cv2



cap = cv2.VideoCapture("/dev/video2")

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

PATH1 = os.path.join(os.getcwd(), 'animal/canigatti')

print(os.listdir(PATH1))
print(PATH1)

while(True):
    ret, frame = cap.read()
    #print(frame)


    #plotImages(capture_images[:1])

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    stroke = 2
    cv2.putText(frame, "test", (100, 100), font, 1, color, stroke, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()