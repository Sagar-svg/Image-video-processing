# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt


# %%
def get_histogram(img, _print):
    
    gray=img
    #create the array which will be storing the no. of pixel of each intensity
    img_hist = np.zeros([256])
    
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            ind = gray[i][j]
            img_hist[ind] += 1
    
    norm_hist = img_hist/(gray.shape[0]*gray.shape[1])
    
    if(_print):
        plot_histogram(norm_hist, "normalized histogram", "n_h_image")
    
    return (norm_hist, gray)
        
        
def plot_histogram(hist, title, name):
    plt.figure()
    plt.title(title)
    plt.plot(hist, color='#ef476f')
    plt.bar(np.arange(len(hist)), hist, color='#b7b7a4')
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    
    
    
def hist_eq(img, hist, _plot):
    
    en_img = np.zeros_like(img)
    cdf = np.zeros_like(hist)
    
    for i in range(len(hist)):
        cdf[i] = int(255 * np.sum(hist[0:i]))
    for x_pixel in range(img.shape[0]):
        for y_pixel in range(img.shape[1]):
            pixel_val = int(img[x_pixel, y_pixel])
            en_img[x_pixel, y_pixel] = cdf[pixel_val]   
    en_img = en_img.astype('uint8')
    if(_plot):
        plt.figure()
        plt.imshow(en_img,cmap='gray')
        plot_histogram(cdf, "equilized histogram", "e_h_image")
        
        
    return cdf, en_img



def hist_matching(in_hist, trgt_hist, in_img, trgt_img):
    modi_hist = np.zeros_like(in_hist)
    en_img = np.zeros_like(in_img)

    for i in range(len(modi_hist)):
        modi_hist[i] = find_value_target(val = in_hist[i],target_arr = trgt_hist)
    for x_pixel in range(in_img.shape[0]):
        for y_pixel in range(in_img.shape[1]):
            pixel_val = int(in_img[x_pixel, y_pixel])
            en_img[x_pixel, y_pixel] = modi_hist[pixel_val]   
    en_img = en_img.astype('uint8')
    in_img = in_img.astype('uint8')
    plot_histogram(modi_hist, 'Modified Histogram', 'm_h_img')
    plt.figure()
    plt.imshow(en_img, cmap = 'gray')
    


    return modi_hist, en_img



# %%
img1 = cv2.imread(r'C:\Users\SAGAR THASAL\Documents\IVP_lab\images\grayscale.jpg',0)
img2 = cv2.imread(r'C:\Users\SAGAR THASAL\Documents\IVP_lab\images\input_contrast.jpg',0)
cv2.imshow("input image", img1)

cv2.waitKey(0) 
cv2.destroyAllWindows()
cv2.imshow("traget_image", img2)
cv2.waitKey(0) 
cv2.destroyAllWindows()


# %%
def find_value_target(val, target_arr):
    key = np.where(target_arr == val)[0]

    if len(key) == 0:
        key = find_value_target(val-1, target_arr)
        
    else:
        return key[0]
# %%
hist1, img1 = get_histogram(img1,False)
hist2, img2 = get_histogram(img2, False)


# %%
eq_hist1, en_img1 =  hist_eq(img1, hist1, False)
eq_hist2, en_img2 =  hist_eq(img2, hist2, False)



# %%


# %%
modi_hist, new_img = hist_matching(eq_hist2,eq_hist1,img2, img1)




# %%
key = np.where(5 == eq_hist2)[0]
print(len(key))
print(key[0])
# %%
