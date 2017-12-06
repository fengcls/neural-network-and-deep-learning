from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import theano
import theano.tensor as T

# from theano.tensor.nnet.neighbours import images2neibs

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean image
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('output/hw1b_im{0}.png'.format(im_num))
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of a image

    imname: string
        name of file where image will be saved.
    '''
    # raise NotImplementedError
    
    f, axarr = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            axarr[i,j].imshow(D[:,i*4+j].reshape(sz,sz),cmap = cm.Greys_r)
            axarr[i,j].get_xaxis().set_visible(False)
            axarr[i,j].get_yaxis().set_visible(False)
    
    f.savefig(imname)
    plt.close(f)


def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and c as
    the coefficient vector
    Parameters
    -------------------
        c: np.ndarray
            a l x 1 vector  representing the coefficients of the image.
            l represents the dimension of the PCA space used for reconstruction

        D: np.ndarray
            an N x l matrix representing first l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in the image)

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to the reconstructed image

        ax: the axis on which the image will be plotted
    '''
    # raise NotImplementedError
    I=np.dot(D,c).T
    
    sz,stmp = D.shape
    sz = np.int(np.sqrt(sz))
    
    ax.imshow(I.reshape([sz,sz])+X_mn,cmap = cm.Greys_r)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

if __name__ == '__main__':
    '''
    Read all images(grayscale) from jaffe folder and collapse each image
    to get an numpy array Ims with size (no_images, height*width).
    Make sure to sort the filenames before reading the images
    '''
    # didn't use walk
    from glob import glob
    import os
    
    files_list = glob(os.path.join('jaffe/', '*.tiff'))
    im_ind=0
    im = Image.open(files_list[0])
    Ims = np.float32(np.zeros([len(files_list),im.height*im.width]))
    for a_file in sorted(files_list):
        im = Image.open(a_file).convert("L")
        Ims[im_ind,:]=np.array(im).reshape([1,im.height*im.width])       
        im_ind = im_ind+1

    Ims = Ims.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''
    
    imhw = im.height*im.width
    
    # cost mut be a scalar,convert (100,1) to (100,); d1[:,0]

    # define symbols used in theano, attach a T
    dT = theano.shared(np.random.randn(imhw),name="dT")

    DT = T.matrix("DT")
    lamT = T.vector("lamT")
    XT = T.matrix("XT")
    
    XdT = T.dot(XT,dT)
    DdT = T.dot(DT.T,dT)
    
    cost = -T.dot(XdT.T,XdT)+T.sum((DdT**2)*lamT)
    
    # gradient
    # g_d = T.grad(cost,[d]) [] cause can't multiply sequence by non-int of type 'float'
    g_dT = T.grad(cost,dT)

    tr_step_max = 200
    
    # Compile
    update_dT = (dT-0.01*g_dT)
    
    train = theano.function(
              inputs=[XT,DT,lamT],
              outputs=[cost,dT],
              updates=[(dT, update_dT/update_dT.norm(2))])
    # error The updates parameter must be an OrderedDict/dict or a list of lists/tuples with 2 elements
    #       updates=((d, d - 0.1 * g_d)) should be [] when using one parameter
    
    num_eig = 16
    D = np.zeros([imhw,num_eig])
    lam = np.zeros(num_eig)
    
    for i in range(num_eig):
        ii = 1
        costval=np.zeros(tr_step_max)
        # terminate when the cost function doesn't change much; and also set a upper limit of possible iterations
        while ((np.abs(costval[ii]-costval[ii-1])>1) or (ii<3)) and (ii<tr_step_max-1):
            ii = ii+1
            costval[ii],dval = train(X,np.float32(D),np.float32(lam))
            print(costval[ii])         
        print(ii)
        D[:,i]=dT.get_value()
        lam[i] = np.dot(np.dot(X,D[:,i]).T,np.dot(X,D[:,i]))
    
    # calculate the coefficient
    c = np.dot(D.T,X.T)
    
    # plot multi
    for i in range(0,200,10):
        plot_mul(c, D, i, X_mn.reshape((256, 256)),
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])
    
    # plot eigenvector
    plot_top_16(D, 256, 'output/hw1b_top16_256.png')

