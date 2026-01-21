import cv2
import numpy as np
import pyautogui
import time

CUP_IMAGE = 'photo.png'
MIN_MATCHES = 15 
PAUSE = 1.5         

def main():
    orb = cv2.ORB_create(nfeatures=2000)

    img_template = cv2.imread(CUP_IMAGE, 0)
    if img_template is None:
        print(f"Ошибка: Не найден файл {CUP_IMAGE}")
        return
    
    kp1, des1 = orb.detectAndCompute(img_template, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    cap = cv2.VideoCapture(0)
    last_scroll = 0

    print("Программа видит всю область камеры. Покажи кружку в любом углу!")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp2, des2 = orb.detectAndCompute(gray_frame, None)

        if des2 is not None and des1 is not None:
            matches = bf.match(des1, des2)
            good_matches = [m for m in matches if m.distance < 45]
            count = len(good_matches)
            color = (0, 255, 0) if count >= MIN_MATCHES else (0, 0, 255)
            
            for m in good_matches[:20]: 
                pt = kp2[m.trainIdx].pt
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)

            cv2.putText(frame, f"Matches: {count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if count >= MIN_MATCHES:
                if time.time() - last_scroll > PAUSE:
                    pyautogui.press('down')
                    print(f"Кружка обнаружена! Точек: {count}")
                    last_scroll = time.time()

        cv2.imshow('Wide Area Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()