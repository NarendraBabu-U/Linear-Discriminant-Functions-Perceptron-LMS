import matplotlib.pyplot as plt
import numpy as np
import time

# -- PLOTTING DATA POINTS--
plt.scatter([1,7,8,9,4,8],[6,2,9,9,8,5],marker='o',label='Class w1')
plt.scatter([2,3,2,7,1,5],[1,3,4,1,3,2],marker='x',label='Class w2')

# -- NORMALIZATION--
y = np.array([[1,1,6],[1,7,2],[1,8,9],[1,9,9],[1,4,8],[1,8,5],[-1,-2,-1],[-1,-3,-3],[-1,-2,-4],[-1,-7,-1],[-1,-1,-3],[-1,-5,-2]])
nsamp = y.shape[0]

# -- TEST DATA --
tests = np.array([[1,6,5],[1,4,4],[1,3,9],[-1,-2,-2],[-1,-4,-1],[-1,-1,-5]])
print "Test samples:\n",tests

# -- Single sample perceptron --
a = np.array([0,2.83,4.17])  #initial weights
c = 0
k = 0

while(c!=nsamp):
    c = 0
    for k in range(nsamp):
        if (np.dot(a,y[k])<=0):
            a = a+y[k]
        else:
            c+=1

plt.plot([0,float(-a[0])/a[1]],[float(-a[0])/a[2],0],color='r',label="Single sample perceptron")
#plt.arrow(0,0,a[1],a[2],fc = 'r',ec='r',head_width =0.15,head_length=0.2)
# for test samples
mis = 0
for l in range(6):
    if (np.dot(a,tests[l])<0):
        mis = mis+1
print "No.of misclassified using ssp",mis


# -- SSP WITH MARGIN --
c1 = 0
k1 = 0
a1 =np.array([0,2.83,4.17])

while(c1!=nsamp):
    c1 = 0
    for k1 in range(12):
        if(np.dot(a1,y[k1])<=100):
            a1 = a1 + y[k1]
        else:
            c1 += 1
# for test samples
mis = 0
for l in range(6):
    if(np.dot(a1,tests[l])<0):
        mis = mis+1
print "No.of misclassified using ssp with margin",mis
plt.plot([0,float(-a1[0])/a1[1]],[float(-a1[0])/a1[2],0],color='g',label="SSP with margin")
#plt.arrow(0,0,a1[1],a1[2],fc = 'g',ec='g',head_width =0.15,head_length=0.2)

# -- RELAXATION WITH MARGIN --
c2 = 0
k2 = 0
a2 = np.array([1,1,1])
while(c2!=nsamp):
    c2 = 0
    for k2 in range(nsamp):
        if(np.dot(a2,y[k2])<1000):
            delta = 2 * y[k2] * (float(1000-np.dot(a2,y[k2]))/(np.dot(y[k2],y[k2]))) # eta should be in between 1.8 and 3.2
            a2 = a2 + delta
        else:
            c2+=1
#for samples
mis = 0
for l in range(6):
    if (np.dot(a2,tests[l])<0):
        mis = mis+1
print "No.of misclassified using relaxation with margin",mis

plt.plot([0,float(-a2[0])/a2[1]],[float(-a2[0])/a2[2],0],color='b',label="Relaxation with margin")
#plt.arrow(0,0,float(a2[1]),float(a2[2]),fc = 'b',ec='b',head_width =0.15,head_length=0.2)


# -- LMS: Least Mean Square error --

a3 = np.array([1,1,1])
err1 = np.array([0,0,0])
err2 = np.array([50,50,50])
c3 = 0
k3 = 0
while(1):
    err1 = np.array([0,0,0])
    for k3 in range(nsamp):
        delta = 0.001 * (1-np.dot(a3,y[k3])) * y[k3]
        a3 = a3 + delta
        err1 = err1 + (np.absolute(delta))
    c3 = c3+1
    if((err2-err1).all()<=0):
        break
    err2 = err1

print "NO.of iterations (on whole batch) took by LMS: ",c3
#for testing
mis = 0
for l in range(6):
    if(np.dot(a3,tests[l])<0):
        mis = mis+1
        print "No.of misclassified using LMS",mis
plt.plot([0,float(-a3[0])/a3[1]],[float(-a3[0])/a3[2],0],color='y',label="LMS")
#plt.arrow(0,0,a3[1],a3[2],fc = 'y',ec='y',head_width =0.5,head_length=0.2)

# -- LMS USING PSEUDO INVERSE --
ym = np.matrix(y)
bm = np.matrix([1,1,1,1,1,1,1,1,1,1,1,1])
am = np.matrix([1,1,1])
jm = ym.transpose() * ym
km = np.linalg.inv(jm)
lm = km * ym.transpose()
am = lm *( bm.transpose())
am = am.transpose()
am = np.array(am)
plt.plot([0,float(-am[0][0])/am[0][1]],[float(-am[0][0])/am[0][2],0],color='c',label="LMS Using pseudoinverse")
#plt.arrow(0,0,am[0][1],am[0][2],fc = 'c',ec='c',head_width =0.15,head_length=0.2)


# -- DISPLAYING THE PLOTS --
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
