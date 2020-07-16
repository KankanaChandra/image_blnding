import cv2
import numpy as np

kcc=cv2.imread('appl.png')
kcc=cv2.resize(kcc,(224,224))
kc= cv2.imread('24.png')
kc=cv2.resize(kc,(224,224))


print(kcc.shape)
print(kc.shape)

kcc_kc=np.hstack((kc[:, :112], kcc[:, 112:]))
kcc_copy=kcc.copy()
gp_kcc=[kcc_copy]

for i in range(6):
    kcc_copy = cv2.pyrDown(kcc_copy)
    gp_kcc.append(kcc_copy)

kc_copy=kc.copy()
gp_kc=[kc_copy]

for i in range(6):
    kc_copy = cv2.pyrDown(kc_copy)
    gp_kc.append(kc_copy)


kcc_copy=gp_kcc[5]
lp_kcc = [kcc_copy]

for i in range(5,0,-1):
    gaussian_expanded= cv2.pyrUp(gp_kcc[i])
    laplacian=cv2.subtract(gp_kcc[i-1], gaussian_expanded)
    lp_kcc.append(laplacian)

kc_copy=gp_kc[5]
lp_kc = [kc_copy]

for i in range(5,0,-1):
    gaussian_expanded= cv2.pyrUp(gp_kc[i])
    laplacian=cv2.subtract(gp_kc[i-1], gaussian_expanded)
    lp_kc.append(laplacian)

#now add the two
kcc_kc_pyramid=[]
n=0
for kcc_lap, kc_lap in zip(lp_kcc, lp_kc):
    n+=1
    cols, rows, ch= kcc_lap.shape
    laplacian = np.hstack((kc_lap[:, 0:int(cols/2)], kcc_lap[:, int(cols/2):]))
    kcc_kc_pyramid.append(laplacian)
kcc_kc_reconstruct= kcc_kc_pyramid[0]
for i in range(1,6):
    kcc_kc_reconstruct=cv2.pyrUp(kcc_kc_reconstruct)
    kcc_kc_reconstruct=cv2.add(kcc_kc_pyramid[i], kcc_kc_reconstruct)
cv2.imshow('kcc',kcc)
cv2.imshow('kc',kc)
cv2.imshow("blended",kcc_kc)
cv2.imshow("kcc_kc", kcc_kc_reconstruct)
cv2.waitKey(0)
cv2.destroyAllWindows()