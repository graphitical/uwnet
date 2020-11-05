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
    int outw = (im_w-1)/l.stride + 1;
    int outh = (im_h-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);
    

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int idx = 0;
    int start = -(l.size%2);
    int d, k, c, i, j, m, n;
    float v, t;
    for (d = 0; d<in.rows; ++d) {
        for (k = 0; k<l.channels; ++k) {
            for (c = 0; c<outw*outh; ++c) {
                v = -FLT_MIN;
                t = -FLT_MIN;
                for (i = 0; i<l.size; ++i) {
                    for (j = 0; j<l.size; ++j) {
                        m = ((c*l.stride)/im_w)*l.stride + i + start;
                        n = (c*l.stride)%im_w + j + start;
                        if (m<0 || n<0 || m>=im_h || n>=im_w) continue;
                        t = in.data[n + im_w*m + im_w*im_h*k + d*im_w*im_h*l.channels];
                        if (v < t) v = t;
                    }
                }
                out.data[idx++] = v;
            }
        }
    }
    assert(idx == out.cols*out.rows);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int im_w = l.width;
    int im_h = l.height;

    int d, k, c, i, j, m, n, a, b;
    float v, t;
    int idx = 0;
    int start = -(l.size%2);
    for (d = 0; d<in.rows; ++d) {
        for (k = 0; k<l.channels; ++k) {
            for (c = 0; c<outw*outh; ++c) {
                v = -FLT_MIN;
                t = -FLT_MIN;
                for (i = 0; i<l.size; ++i) {
                    for (j = 0; j<l.size; ++j) {
                        m = ((c*l.stride)/im_w)*l.stride + i + start;
                        n = (c*l.stride)%im_w + j + start; 
                        if (m<0 || n<0 || m>=im_h || n>=im_w) continue;
                        t = in.data[n + im_w*(m + im_h*k) + d*in.cols];
                        if (v < t) {
                            v = t;
                            a = m;
                            b = n;
                        }
                    }
                }
                dx.data[b + im_w*(a + im_h*k) + d*in.cols] += dy.data[idx++];
            }
        }
    }

    assert(idx == dy.cols*dy.rows);
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

