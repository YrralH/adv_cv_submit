import cv2 as cv

def hj_resize_to(img, a, b, colors=[1.0, 1.0, 1.0]) :
#created in 2020-10-13 by hj
#ususally padding after resizing
    x = img.shape[0]
    y = img.shape[1]
    
    if x == 0 :
        flag = False
    else :
        if y == 0 :
            flag = True
        else :
            if a/x < b/y :
                flag = True
            else :
                flag = False

    if flag : 
        new_x = int (a) # which means new_x = x*(a/x)
        new_y = int (y * (a/x))
        #print('new_x_y', new_x, new_y)
        resize_img = cv.resize(img, (new_y, new_x), interpolation=cv.INTER_CUBIC)
        
        padding_y_1 = (b - new_y) // 2
        padding_y_2 = (b - new_y) - padding_y_1
        
        padding_img = cv.copyMakeBorder(resize_img, 0, 0, padding_y_1, padding_y_2, cv.BORDER_CONSTANT, value=colors)
        pare_img_valid_begin = (0, padding_y_1)
    else :
        new_y = int (b) # which means new_y = y*(b/x)
        new_x = int (x * (b/y))
        #print('new_x_y', new_x, new_y)
        resize_img = cv.resize(img, (new_y, new_x), interpolation=cv.INTER_CUBIC)
        
        padding_x_1 = (a - new_x) // 2
        padding_x_2 = (a - new_x) - padding_x_1
        
        padding_img = cv.copyMakeBorder(resize_img, padding_x_1, padding_x_2, 0, 0, cv.BORDER_CONSTANT, value=colors)
        pare_img_valid_begin = (padding_x_1, 0)
        
    return (padding_img, pare_img_valid_begin)

