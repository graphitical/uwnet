#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int im_w = l.width;
    int im_h = l.height;
    // printf("im_w:%d\tim_h:%d\n",im_w,im_h);
    int outw = (im_w-1)/l.stride + 1;
    int outh = (im_h-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);
    

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int idx = 0;
    int start = -(l.size%2);
    int d, k, c, i, j, m, n;
    float v, t;
    for (d = 0; d < in.rows; ++d) {
        for (k = 0; k < l.channels; ++k) {
            for (c = 0; c < outw*outh; ++c) {
                v = -1e3;
                for (i = 0; i < l.size; ++i) {
                    for (j = 0; j < l.size; ++j) {
                        m = ((c*l.stride)/im_w)*l.stride + i + start;
                        n = (c*l.stride)%im_w + j + start;
                        if(m < 0) m=0; if (n < 0) n=0;
                        if(m > im_h) m = im_h; if (n > im_w) n = im_w;
                        t = in.data[n + im_w*(m + im_h*k)];
                        if (t > v) v = t;
                    }
                }
                out.data[idx++] = v;
            }
        }
    }
    assert(idx == out.cols * out.rows);
    // printf("rows:%d",out.rows);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    // int outw = (l.width-1)/l.stride + 1;
    // int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int d, k, i, j, m, n, a, b, r, c;
    int idx = 0;
    float t, v;
    int start = -(l.size%2);
    for (d = 0; d < dy.rows; ++d) {
        for (k = 0; k < l.channels; ++k) {
            for (i = 0; i < l.height; i=i+l.stride) {
                for (j = 0; j < l.width; j=j+l.stride) {
                    v = -1e3;
                    for (m = start; m < l.size + start; ++m) {
                        for (n = start; n < l.size + start; ++n) {
                            c = j + n;
                            r = i + m;
                            if (c < 0) c = 0;
                            if (r < 0) r = 0;
                            if (c > l.width) c = l.width;
                            if (r > l.height) r = l.height;
                            t = in.data[c + l.width*(r + l.height*k)];     
                            if (t > v) {
                                v = t;
                                a = r;
                                b = c;
                            }                       
                        }
                    }

                    dx.data[b + l.width*(a + l.height*k)] += dy.data[idx++];
                }
            }
        }
    }

    assert(idx == dy.cols * dy.rows);
    // printf("rows:%d",dy.rows);
    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

