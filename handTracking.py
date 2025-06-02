import cv2
import mediapipe as mp
import time
import pyautogui
import math

#warming up the camera
video = cv2.VideoCapture(0)
time.sleep(3)

#mandatory for hand detection
mpHands = mp.solutions.hands

#attain the required params
hands = mpHands.Hands(max_num_hands = 1)

#drawing setup
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = video.read()
    
    

    #flip frame(inverted camera)
    img = cv2.flip(img, 1)
    
    #new game text to open new game tab
    cv2.putText(img, str("New Game"), (950,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)

    #title
    cv2.putText(img, str("Dino Game"), (550,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 3)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #process the landmarks(id, x and y coordinates)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handslm in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handslm.landmark):
                h, w, c = img.shape
                
                #coordinate for each landmark 1-20
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

                #draw purple circle on tip of thumb and index finger
                if id == 4 or id == 8:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            
            #draw landmarks with connections
            mpDraw.draw_landmarks(img, handslm, mpHands.HAND_CONNECTIONS)
            
                
            if lmList:
                #get coordinates of thumb and index finger tip
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                #print(lmList[8])

                #start dino game
                if x2 >= 1000 and y2 <= 95:
                    #load dino game on new tab if hands are captured 
                    pyautogui.hotkey('command', 't')
                    #time.sleep(1)
                    pyautogui.write('https://chrome-dino-game.github.io/')
                    pyautogui.press('enter')
                    #time.sleep(3)
                 
                distance = math.hypot((x2 - x1), (y2 - y1))

                #jumping and running instructions
                if distance < 40:
                    pyautogui.press('space')
                    cv2.putText(img, str("Jumping"), (150,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)
                else:
                    cv2.putText(img, str("Running"), (150,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 3)

    #show image
    cv2.imshow("image", img)
    #close window if ESC key is pressed
    if cv2.waitKey(10) == 27:
        break
