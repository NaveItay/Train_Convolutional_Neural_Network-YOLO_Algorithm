import cv2
cap = cv2.VideoCapture('DJI_0858.MP4')
ret, current_frame = cap.read()
ret1, original = cap.read()
previous_frame = current_frame

# Saves every save_count frame
save_count = 1

# Start names from '0.jpg'
image_name = 0

while(cap.isOpened()):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
    sourceImg = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    cv2.imshow('frame_diff', frame_diff)
    cv2.imshow('current_frame', current_frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("Save frame " + str(image_name) + ".jpg")
        # Save frame to image file
        cv2.imwrite(str(image_name) + '.jpg', current_frame)
        # Next image name
        image_name += 1
        #
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()
    ret1, original = cap.read()

cap.release()
cv2.destroyAllWindows()
