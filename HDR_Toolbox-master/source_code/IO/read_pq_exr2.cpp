/*==========================================================
 * read_exr_pq.cpp
 * 与 read_limited_to_pq_exr.cpp 完全一致的 PQ → Linear 读取代码
 *========================================================*/

#include "mex.h"
#include <vector>
#include <string>
#include <cmath>

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

// ---------------- PQ → Linear 转换函数 ----------------
inline float pq2lin(float V)
{
    const double Lmax = 10000.0;
    const double m1 = 0.1593017578125;
    const double m2 = 78.84375;
    const double c1 = 0.8359375;
    const double c2 = 18.8515625;
    const double c3 = 18.6875;

    if (V < 0.0f) V = 0.0f;
    if (V > 1.0f) V = 1.0f;

    double Vm = pow(V, 1.0 / m2);
    double num = std::max(Vm - c1, 0.0);
    double den = c2 - c3 * Vm;
    if (den <= 0.0) return 0.0f;

    double L = Lmax * pow(num / den, 1.0 / m1);
    return static_cast<float>(L);
}

/* ---------------- 主函数入口 ---------------- */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("HDRToolbox:read_exr_pq:nrhs", "One input filename required.");
    }

    // 读取文件名
    char *nameFile;
    mwSize buflen = mxGetN(prhs[0]) * sizeof(mxChar) + 1;
    nameFile = (char *)mxMalloc(buflen);
    mxGetString(prhs[0], nameFile, buflen);

    // 初始化 EXR
    EXRImage image;
    InitEXRImage(&image);

    const char *err = nullptr;
    int ret = ParseMultiChannelEXRHeaderFromFile(&image, nameFile, &err);
    if (ret != 0) {
        mexErrMsgIdAndTxt("HDRToolbox:read_exr_pq:parse", err);
        return;
    }

    int width = image.width;
    int height = image.height;
    int channels = image.num_channels;

    // 创建输出
    mwSize dims[3] = {(mwSize)height, (mwSize)width, (mwSize)channels};
    plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    double *outMatrix = mxGetPr(plhs[0]);

    // half → float
    for (int i = 0; i < channels; i++) {
        if (image.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
            image.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        }
    }

    // 加载 EXR
    ret = LoadMultiChannelEXRFromFile(&image, nameFile, &err);
    if (ret != 0) {
        mexErrMsgIdAndTxt("HDRToolbox:read_exr_pq:load", err);
        return;
    }

    float **images = (float **)image.images;

    int nPixels = width * height;
    int nPixels2 = nPixels * 2;

    if (channels == 1) {
        nPixels = 0;
        nPixels2 = 0;
    } else if (channels == 2) {
        nPixels2 = 0;
    }

    // ------- 遍历像素 & PQ → Linear -------
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {

            int index = i * height + j;        // MATLAB输出顺序
            int indexOut = j * width + i;     // EXR原始顺序 (row-major)

            // OpenEXR 默认 B G R
            float R = images[2][indexOut];
            float G = images[1][indexOut];
            float B = images[0][indexOut];

            // PQ → Linear
            R = pq2lin(R);
            G = pq2lin(G);
            B = pq2lin(B);

            outMatrix[index] = R;
            if (channels > 1)
                outMatrix[index + nPixels] = G;
            if (channels > 2)
                outMatrix[index + nPixels2] = B;
        }
    }

    FreeEXRImage(&image);
}
