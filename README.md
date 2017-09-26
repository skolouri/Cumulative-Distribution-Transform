# Cumulative Distribution Transform


Cumulative distribution transform (CDT) as described in:

[(1) Park SR, Kolouri S, Kundu S, Rohde GK. The cumulative distribution transform and linear pattern classification. Applied and Computational Harmonic Analysis. 2017 Feb 22.](http://www.sciencedirect.com/science/article/pii/S1063520317300076)

is a nonlinear and invertible transformation for nonnegative one-dimensional signals that guarantees certain linear separation theorems. CDT rises from the rich mathematical foundations of optimal mass transportation, and therefore has a unique geometric interpretation. Unlike the current data extensive nonlinear models, including deep neural networks and their variations, CDT provides a well-defined invertible nonlinear transformation that could be used alongside linear modeling techniques, including principal component analysis, linear discriminant analysis, and support vector machines (SVM), and does not require extensive training data.

 The corresponding iPython Notebook file for this post could be find [here](https://github.com/skolouri/Cumulative-Distribution-Transform/blob/master/CDT_Demo.ipynb). The demo is tested with:

1. numpy '1.13.1'
2. sklearn '0.18.1'
3. scipy '0.19.1'

Here we first walk you through the formulation of CDT and then demonstrate its application on various demos.

## Formulation

Consider two nonnegative one-dimensional signals <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/> defined on <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/82289f06c71b94420b624654501ad06e.svg?invert_in_darkmode" align=middle width=68.09187pt height=22.56408pt/>. Without the loss of generality assume that these signals are normalized so that they could be treated as probability density functions (PDFs). Considering <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> to be a pre-determined 'template'/'reference' PDF, and following the definition of the **optimal mass transportation**  for one-dimensional distributions, one can define the optimal transport map, <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/a4fe1ef6edd893e70831c6cf216f5ec3.svg?invert_in_darkmode" align=middle width=76.982895pt height=22.74591pt/> using,

\[ \int_{inf(Y)}^{f(x)} I(\tau) d\tau=\int_{inf(X)}^{x}I_0(\tau)d\tau\]

which uniquely associates <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/7211ec64117b386a4d281f03e816f84c.svg?invert_in_darkmode" align=middle width=76.982895pt height=22.74591pt/> to the given density <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/>.

### Forward Transform
We use this relationship to define the ** Cumulative Distribution Transform (CDT)** of <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/> (denoted as <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/009c5c6e872cda936558aa1391e2980a.svg?invert_in_darkmode" align=middle width=74.39223pt height=31.0563pt/>), with respect to the reference <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/>:

\[\hat{I}(x) = \left(  f(x) - x \right) \sqrt{I_0(x)}.\]

For one-dimensional PDFs the transport map <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> is uniquely defined, and can be calculated from:

\[f(x)=J^{-1}(J_0(x)).\]

where <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/d5678db6a0e35236c7037b64736ccf19.svg?invert_in_darkmode" align=middle width=103.26822pt height=24.56553pt/> and <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/066ab7381e7d650225f1730bd6a691a7.svg?invert_in_darkmode" align=middle width=95.777715pt height=24.56553pt/> are the corresponding cumulative distribution functions (CDFs) for <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/3ee98a0ddf705fc4e453f42e3e2563c6.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/>, that is: <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/002d1c8d6f7bd6452b21706524c5b673.svg?invert_in_darkmode" align=middle width=163.540245pt height=28.2282pt/>, <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/88ba2440e8b83e1f8a6db6af4599ea63.svg?invert_in_darkmode" align=middle width=149.80515pt height=28.2282pt/>. For continuous positive PDFs <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/>, <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> is a continuous and monotonically increasing function. If <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> is differentiable, we can rewrite the above equation as:

\[I_0(x) = f^{\prime}(x) I(f(x)).\]

### Inverse Transform

The Inverse-CDT of <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/bb1509c53ed9e94118cb98cd9436ad7f.svg?invert_in_darkmode" align=middle width=10.163505pt height=31.0563pt/> is defined as:
\[I(y) = \frac{d }{dy}J_0(f_1^{-1}(y)) = (f_1^{-1})^{\prime} I_0(f^{-1}(y))\]

where <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/86035f674998337e99cf3bd753ab438f.svg?invert_in_darkmode" align=middle width=94.64004pt height=26.70657pt/> refers to the inverse of <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> (i.e. <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/37634a69c6e56ca1ec7efe3f61465c07.svg?invert_in_darkmode" align=middle width=103.288185pt height=26.70657pt/>), and where <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/db2e04fa41ca5d29f4ff03898514dda5.svg?invert_in_darkmode" align=middle width=181.347045pt height=31.0563pt/>. The equation above holds for points where <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/33799619e6a8adb0933941909e268d50.svg?invert_in_darkmode" align=middle width=15.60933pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> are differentiable. By the construction above, <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> will be differentiable except for points where <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/d906cd9791e4b48a3b848558acda5899.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> are discontinuous. Now we are ready to delve into some exciting applications of CDT.

## CDT Demo

Throughout the experiments in this tutorial we assume that: 1) <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/fd4304ee3054290b895e68316c23bfb1.svg?invert_in_darkmode" align=middle width=49.874715pt height=22.38192pt/>, and 2) the template PDF is the uniform distribution on <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.85297pt height=22.38192pt/>. Lets start by showing the nonlinear nature of CDT.

### Nonlinearity

Let <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/d906cd9791e4b48a3b848558acda5899.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/9eff113852463b85a970d2d65d52280c.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> be two one-dimensional Gaussian distributions, where:
\[I_i(x)= \frac{1}{\sqrt{2\pi\sigma_i^2}}exp({-\frac{|x-\mu_i|^2}{2\sigma_i^2}})\]
and let <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/a10ff06f53725bca54c49936f91f5fa5.svg?invert_in_darkmode" align=middle width=11.832645pt height=31.0563pt/> denotes the corresponding CDTs. The signals and their CDT could for instance be calculated from the code below:

```python
import numpy as np
import transportBasedTransforms.cdt as CDT
#Define the template (Reference) PDF
N=250
I0= (1.0/N)*np.ones(N)
cdt=CDT.CDT(template=I0)
# Define the signals I_i and calculate their CDT
mu=np.array([25,200]) #Means
sigma=np.array([5,10])#stds
I=np.zeros((2,N))
Ihat=np.zeros((2,N))
for i in range(2):
    I[i,:]=1/(sigma[i]*np.sqrt(2*np.pi))*np.exp(-((x-mu[i])**2)/(2*sigma[i]**2))
    Ihat[i,:]=cdt.transform(I[i,:])
```
which results in,

![<img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/45de514e4bd2f5ba36f09fff6b549760.svg?invert_in_darkmode" align=middle width=11.832645pt height=22.38192pt/>s and their corresponding CDT](Figures/figure1.png)

Now to demonstrate the nonlinear nature of CDT, we choose the simplest linear operator, which is averaging the two signals. We average the signals in the signal space, <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/8a44d333fd9307aaca485044a927c088.svg?invert_in_darkmode" align=middle width=119.29764pt height=24.56553pt/>, and in the CDT space, <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/e7671f5d3f9854cf81eff88849eebd67.svg?invert_in_darkmode" align=middle width=172.956795pt height=31.0563pt/>, and compare the results below.

```python
I3=I.mean(axis=0)
I4=cdt.itransform(Ihat.mean(axis=0)
```
which results in,

![Averaging in the signal domain versus in the CDT domain](Figures/figure2.png)

It can be clearly seen that CDT provides a nonlinear averaging for these signals. Note that we don't have a specific model for Gaussians and while CDT is unaware of the parametric formulation of the signals it can still provide a meaningful average. Next we will discuss the linear separability characteristic of CDT.

### Linear separability

Park et al. (1) showed that CDT can turn certain not linearly separable classes of one-dimensional signals into linearly separable ones. Here we run a toy example to demonstrate this characteristic. We start by defining three classes of signals, where Class <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.041505pt height=22.74591pt/>, for <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/1cc5019c20e25f4af278b10609f0574a.svg?invert_in_darkmode" align=middle width=84.613815pt height=24.56553pt/>, consists of translated versions of a <img src="https://rawgit.com/skolouri/Cumulative-Distribution-Transform/master/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.041505pt height=22.74591pt/>-modal Gaussian distribution. Here we generate these signal classes and their corresponding CDTs.

```python
K=3 # Number of classes
L=500 # Number of datapoints per class  
I=np.zeros((K,L,N))
Ihat=np.zeros((K,L,N))
kmodal_shift=[]
kmodal_shift.append(np.array([0]))
kmodal_shift.append(np.array([-15,15]))
kmodal_shift.append(np.array([-30,0,30]))
sigma=5
for k in range(K):
    for i,mu in enumerate(np.linspace(50,200,L)):
        for j in range(k+1):
            I[k,i,:]+=1/((k+1)*sigma*np.sqrt(2*np.pi))*np.exp(-((x-mu-kmodal_shift[k][j])**2)/(2*sigma**2))
        Ihat[k,i,:]=cdt.transform(I[k,i,:])
```

This leads to the following signals:

![Sample signals from the three classes.](Figures/figure3_a.png)
![CDT of the sample signals from the three classes.](Figures/figure3_b.png)

Next we run a simple linear classification on these signals in the original space and in the CDT space.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=2) #Get the classifier object
X=np.reshape(I,(K*L,N))      #Combine the signals into a features vector X
Xhat=np.reshape(Ihat,(K*L,N))     #Combine the transformed signals into a features vector Xhat
data=[X,Xhat]
label=np.concatenate((np.zeros(L,),np.ones(L,),-1*np.ones(L,))) # Define the labels as -1,0,1 for the three classes
dataLDA=[[],[]]
for i in range(2):
    dataLDA[i]=lda.fit_transform(data[i],label)
```

Below we visualize the two-dimensional discriminant subspace calculated by the linear discriminant analysis (LDA).

![Visualization of the LDA subspace calculated from the original space and the CDT space.](Figures/figure4.png)

It can be clearly seen that while the classes are not linearly separable in the original space (Note the one-dimensional nonlinear manifold structure of each class), the CDT representations of the signals is linearly separable.
