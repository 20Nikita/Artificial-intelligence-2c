import cv2 as cv
import numpy as np
import warp
if __name__ == '__main__':
    cv.namedWindow( "result" )
    cap = cv.VideoCapture(0)
    flag, img = cap.read()

    hsv_min = np.array((0, 0, 0), np.uint8)
    hsv_max = np.array((255,255,60), np.uint8)

    color_blue = (255,0,0)
    color_red = (0,0,128)
    # открыть изображение для деформации
    fromim = np.array(cv.imread('data/1.jpeg'))
    x,y = np.meshgrid(range(5),range(6))
    x = (fromim.shape[1]/4) * x.flatten()
    y = (fromim.shape[0]/5) * y.flatten()
    x = np.array([0, fromim.shape[1],0, fromim.shape[1]])
    y = np.array([0,0, fromim.shape[0], fromim.shape[0]])
    # triangulate
    tri = warp.triangulate_points(np.stack((x, y), axis=1)).simplices

    while True:
        flag, img = cap.read()
        img = cv.flip(img,1)
        try:
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV )
            thresh = cv.inRange(hsv, hsv_min, hsv_max)
            contours0, hierarchy = cv.findContours( thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            for cnt in contours0:
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.array([box[0],box[3],box[1],box[2]])
                box = np.int0(box)
                t = box.T
                t[0] =np.clip(t[0], 0, img.shape[1])
                t[1] =np.clip(t[1], 0, img.shape[0])
                box = t.T
                area = int(rect[1][0]*rect[1][1])
                if area > 500:
                    cv.drawContours(img,[box],0,color_blue,2)
                    print(box)
                    # конвертировать точки в hom. координаты
                    fp = np.vstack((y,x,np.ones((1,len(x)))))
                    box =np.vstack((box[:,1],box[:,0],np.ones((1,len(box)))))
                    # warp triangles
                    img = warp.pw_affine(fromim.copy(),img,fp,box,tri)

            cv.imshow('result', img)
        except:
            cap.release()
            raise
        ch = cv.waitKey(5)
        if ch == 27:
            break

    cap.release()
    cv.destroyAllWindows()