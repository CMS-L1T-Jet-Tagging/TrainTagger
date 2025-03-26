//Numpy array shape [10]
//Min -0.046875000000
//Max 0.156250000000
//Number of zeros 0

#ifndef B25_H_
#define B25_H_

#ifndef __SYNTHESIS__
bias25_t b25[10];
#else
bias25_t b25[10] = {0.078125, 0.156250, 0.140625, -0.046875, 0.062500, 0.093750, -0.015625, 0.093750, 0.031250, 0.015625};

#endif

#endif
