import cv2
import mediapipe as mp
import pyautogui
import tkinter
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import time
driver=None
cap = None
entry1=None
entry2=None
button1=None
button2=None

def euclid(x, y): return (x**2+y**2)**(0.5)

def chrome(idx, babystart, babyend):
    if idx==6:
        pyautogui.hotkey('ctrl', 't')
    elif idx==7 or idx==8:
        if babystart.x-babyend.x>0: pyautogui.hotkey('ctrl','tab')
        elif babystart.x-babyend.x<0: pyautogui.hotkey('ctrl', 'shift', 'tab')
    else: return
    time.sleep(1)

def youtube(idx, thumb, joint, index, end):
    if idx==0 and thumb.y-joint.y<0:
        driver.find_element(By.XPATH,"//*[@id='segmented-like-button']/ytd-toggle-button-renderer/yt-button-shape/button/yt-touch-feedback-shape/div/div[2]").click()
        print("like successed")
                        
    elif idx==1 and thumb.y-joint.y>0:
        driver.find_element(By.XPATH,"//*[@id='segmented-dislike-button']/ytd-toggle-button-renderer/yt-button-shape/button/yt-touch-feedback-shape/div/div[2]").click()
        print("hate successed")
                        

    elif idx==2:
        try: driver.find_element(By.XPATH,"//*[@id='subscribe-button']/ytd-subscribe-button-renderer/yt-smartimation/yt-button-shape/button/yt-touch-feedback-shape/div/div[2]").click()
        except: 
            pass
        print("subscribe successed")

    elif idx==3:
        pyautogui.press('k')
                    
    elif idx==4 or idx==5:
        if index.x-end.x>0: pyautogui.press('right')
        elif index.x-end.x<0: pyautogui.press('left')

    elif idx==9:
        pyautogui.hotkey('shift', 'n')
    
    elif idx==11:
        pyautogui.press('f')
    
    elif idx==12:
        diff1 = euclid(thumb.x-index.x, thumb.y-index.y)
        diff2 = euclid(thumb.x-joint.x, thumb.y-joint.y)

        try: 
            ratio = diff1/diff2
            if ratio>=1:
                for i in range(int((ratio-1)*30)): pyautogui.press('volumeup')
            elif ratio<1:
                for i in range(int((1-ratio)*30)): pyautogui.press('volumedown')
        except: pass
                    
    time.sleep(1)

def play(): 
    global cap
    global hands
    global driver
    driver=uc.Chrome()
    cap = cv2.VideoCapture(0)
    driversetting()
    while cv2.waitKey(1)!=ord('q'):
        ret, img = cap.read()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])
                thumb = res.landmark[4]
                index = res.landmark[8]
                joint = res.landmark[2]
                end = res.landmark[5]
                babystart = res.landmark[20]
                babyend = res.landmark[17]
                
                chrome(idx, babystart, babyend)
                if 'youtube' in driver.current_url:
                    youtube(idx, thumb, joint, index, end)
        cv2.imshow('Game', img)
        

def shutdown():
    global cap
    global driver
    cap.release()
    cv2.destroyAllWindows()
    driver.quit()
    driver=None
    cap=None

def driversetting():
    driver.implicitly_wait(7)
    driver.get("https://www.youtube.com/")
    driver.implicitly_wait(7)
    driver.find_element(By.XPATH, "//*[@id='buttons']/ytd-button-renderer/yt-button-shape/a/yt-touch-feedback-shape/div/div[2]").click()
    driver.find_element(By.XPATH, "//*[@id='identifierId']").send_keys(str(entry1.get()))
    driver.find_element(By.XPATH, "//*[@id='identifierNext']/div/button/span").click()
    driver.implicitly_wait(5)
    driver.find_element(By.XPATH, "//*[@id='password']/div[1]/div/div[1]/input").send_keys(str(entry2.get()))
    driver.find_element(By.XPATH, "//*[@id='passwordNext']/div/button/span").click()

def windowsetting():
    global entry1
    global entry2
    global button1
    global button2
    window=tkinter.Tk()

    window.title("Like&Subscribe")
    window.geometry("640x400+100+100")
    window.resizable(False, False)

    label=tkinter.Label(window, text="실행할 때, 열려 있는 크롬 창은 모두 꺼주세요.", width=50, height=3, relief="solid")
    label.pack()

    button1 = tkinter.Button(window, text='실행', overrelief="solid", width=15, command=play, repeatdelay=1000, repeatinterval=100)
    button2 = tkinter.Button(window, text='종료', overrelief="solid", width=15, command=shutdown, repeatdelay=1000, repeatinterval=100)
    label2=tkinter.Label(window, text="구글 아이디", width=50, height=3, relief="solid")
    label2.pack()
    entry1 = tkinter.Entry(window)
    entry1.pack()
    label3=tkinter.Label(window, text="구글 비번", width=50, height=3, relief="solid")
    label3.pack()
    entry2 = tkinter.Entry(window)
    entry2.pack()
    
    button1.pack()
    button2.pack()

    window.mainloop()

if __name__=="__main__":
    max_num_hands = 1
    gesture = {
        0:'like', 1:'hate', 2:'V', 3:'resume_paper', 4:'afterfive', 5:'beforefive', 6:'plus', 7:'left', 8:'right', 9:'nextvideo', 10:'full', 11:'rock'
    }

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    # Gesture recognition model
    file = np.genfromtxt('data/gesture_train_fy.csv', delimiter=',')
    print(file)
    angle = file[:,:-1].astype(np.float32)
    print(angle)
    label = file[:, -1].astype(np.float32)
    print(label)
    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)
    windowsetting()
