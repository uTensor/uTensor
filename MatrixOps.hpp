#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

//Useful links
//http://www.plantation-productions.com/Webster/www.artofasm.com/Linux/HTML/Arraysa2.html

//Assume Tensor index order:  rows, columns | the first 2 dimensions
typedef TensorBase<unsigned char> Mat

void printMat(Mat mat) {
    const uint32_t ROWS = mat.getShape()[0];
    const uint32_t COLS = mat.getShape()[1];
    unsigned char* pData = mat.getPointer({});

    for(int i_r = 0; i_r < ROWS; i_r++) {
        for(int i_c = 0; i_c < COLS; i_c++) {
            printf("%f ", pData[i_r * COLS + i_c]);
        }
        printf("\r\n");
    }
}

void initMatConst(Mat mat, unsigned char value) {
    const uint32_t ROWS = mat.getShape()[0];
    const uint32_t COLS = mat.getShape()[1];
    unsigned char* pData = mat.getPointer({});

    for(int i_r = 0; i_r < ROWS; i_r++) {
        for(int i_c = 0; i_c < COLS; i_c++) {
            pData[i_r * COLS + i_c] = value;
        }
    }
}

void multMat(Mat A, Mat B, Mat C) {
    const uint32_t A_ROWS = A.getShape()[0];
    const uint32_t A_COLS = A.getShape()[1];
    const uint32_t B_ROWS = B.getShape()[0];
    const uint32_t B_COLS = B.getShape()[1];
    const uint32_t C_ROWS = C.getShape()[0];
    const uint32_t C_COLS = C.getShape()[1];
    unsigned char* A_Data = A.getPointer({});
    unsigned char* B_Data = B.getPointer({});
    unsigned char* C_Data = C.getPointer({});

    if(A_COLS != B_ROWS) {
        printf("A and B matrices dimension mismatch\r\n");
        return;
    }

    if(C_ROWS != A_ROWS || C_COLS != B_COLS) {
        printf("output matrix dimension mismatch\r\n");
        return;
    }

    for(int r = 0; r < A_ROWS; r++) {
        for(int c = 0; c < B_COLS; c++) {

            uint32_t acc = 0;

            for(int i = 0; i < B_ROWS; i++) {
                acc += A_Data[i + r * A_COLS] * B_Data[c + i * B_ROWS];
            }

            C_Data[c + r * C_ROWS] = acc;
        }
    }
}

#endif
