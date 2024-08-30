import cv2
import pyautogui as robot

def initialize_cascades():
    """Load the Haar cascades for face and eye detection."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return face_cascade, eye_cascade

def detect_faces(gray_image, face_cascade):
    """Detect faces in a grayscale image."""
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    return faces

def detec_eyes(gray_image, eye_cascade):
    """Detect eyes in a grayscale image of face (approximtely half of the face)."""
    eyes = eye_cascade.detectMultiScale(gray_image, minSize=(30,30), scaleFactor=1.1, minNeighbors=5)
    return eyes

def change_mouse(final_result, x_start, y_start, x_end, y_end):
    """Change mouse position by moving the face"""
    
    white = (255, 255, 255)
    red = (0, 0, 255)
    
    color = white
    
    mouse_x = robot.position().x
    mouse_y = robot.position().y
    
    width = final_result.shape[1]//4
    height = final_result.shape[0]//4
    
    
    if x_start < width+50:
        color= red
        
        mouse_x = mouse_x - abs(x_start - width-50)
        robot.moveTo(mouse_x, mouse_y, 0)
        
    if x_end > 3*width-50:
        color=red
        
        mouse_x = mouse_x + abs(x_end-(3*width+50))
        robot.moveTo(mouse_x, mouse_y, 0)
        
    if y_start < height-50:
        color=red
        
        mouse_y = mouse_y - abs(y_start-height+50)
        robot.moveTo(mouse_x, mouse_y, 0)
    
    if y_end > 3*height+50:
        color=red
        
        mouse_y = mouse_y + abs(y_end-3*height-50)
        robot.moveTo(mouse_x, mouse_y, 0)
        
        
    #draw rectangle in the middle of the picture
    final_result=cv2.rectangle(final_result, (width+50,height-50), (3*width-50,3*height+50), color, 2)
    
    return final_result
    
    
def main():
    """Main function to capture video, detect faces and eyes, and control the mouse."""
    face_cascade, eye_cascade = initialize_cascades()
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detect_faces(gray, face_cascade)
        final_result=frame.copy()
        
        
        #if a face been detected, detect eye in the face and draw rectangle around it
        if len(face) > 0:
            x_start=face[0][0]
            y_start=face[0][1]
            x_end=x_start+face[0][2]
            y_end=y_start+face[0][3]
            
            final_result=cv2.rectangle(final_result, (x_start,y_start), (x_end,y_end), (0,255,0), 3)

            # gray_face=gray[y:y2, x:x2]
            gray_face=gray[y_start:y_end-face[0][3]//3, x_start:x_end]
            eyes = detec_eyes(gray_face, eye_cascade)
            
            
            for (xe, ye, w, h) in eyes:
                cv2.rectangle(final_result, (xe+x_start, ye+y_start), (xe+x_start+w, ye+y_start+h), (255, 0, 0), 3)
                
            final_result = change_mouse(final_result, x_start, y_start, x_end, y_end)
            
        cv2.imshow('image', final_result)
    
        if cv2.waitKey(1)==ord("q"):
            loop=False
            cv2.destroyAllWindows()
            cam.release()
            break
    
        
if __name__ == "__main__":
    main()