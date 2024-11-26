#include <stdio.h>
#include <fstream>

// coded by Jormun 20241117

#define N0 10
#define N1 10
#define S 6
#define P 8
float pi = 3.1415926, timer = 0, dt = 0.01, end = 10; // ma re

float legendre(float ps, int k)
{
    if (k == -1)
        return 0.0;
    else if (k == 0)
        return 1.0;
    else
        return ((2 * k - 1) * ps * legendre(ps, k - 1) - (k - 1) * legendre(ps, k - 2)) / k;
}

float degendre(float ps, int k)
{
    if (k == -1)
        return 0.0;
    else if (k == 0)
        return 0.0;
    else
        return ((2 * k - 1) * (ps * degendre(ps, k - 1) + legendre(ps, k - 1)) - (k - 1) * degendre(ps, k - 2)) / k;
}

struct cubit
{
    cubit *idx[4] = {nullptr, nullptr, nullptr, nullptr};
    int flg[4] = {0, 0, 0, 0};

    float spect[3][4][S]; // r d
    float value[4][P][P], value_x[4][P][P], value_y[4][P][P];
    float base[S][P][P], base_x[S][P][P], base_y[S][P][P];
    float test[S][P][P], test_x[S][P][P], test_y[S][P][P];
    float q[4][2], n[4][2], pos[2][P][P]; // quadrant normal positon

    __device__ void clearance(float un[4], unsigned int p0, unsigned int p1, int flgidx)
    {
        if (flg[flgidx] == 0 && flgidx == 0)
            p0 = p0, p1 = 0;
        else if (flg[flgidx] == 0 && flgidx == 1)
            p0 = P - 1, p1 = p1;
        else if (flg[flgidx] == 0 && flgidx == 2)
            p0 = p0, p1 = P - 1;
        else if (flg[flgidx] == 0 && flgidx == 3)
            p0 = 0, p1 = p1;
        else if (flg[flgidx] == 1)
            ;
        else
            ; // support
        if (flg[flgidx] == 0)
            for (int d = 0; d < 4; ++d)
                un[d] = idx[flgidx]->value[d][p0][p1];
    }

    __device__ void convection_x(float cx[4], unsigned int p0, unsigned int p1)
    {
        float u[4], pres, alpha, nn, cxn[4], un[4], presn, alphan;
        u[0] = value[0][p0][p1], u[1] = value[1][p0][p1], u[2] = value[2][p0][p1], u[3] = value[3][p0][p1];
        pres = 0.4 * u[3] - 0.2 * u[1] * u[1] / u[0] - 0.2 * u[2] * u[2] / u[0],
        cx[0] = u[1], cx[1] = u[1] * u[1] / u[0] + pres, cx[2] = u[1] * u[2] / u[0], cx[3] = u[1] / u[0] * (u[3] + pres);
        if (p1 == P - 1)
            clearance(un, p0, p1, 0), nn = n[0][0];
        else if (p0 == 0)
            clearance(un, p0, p1, 1), nn = n[1][0];
        else if (p1 == 0)
            clearance(un, p0, p1, 2), nn = n[2][0];
        else if (p0 == P - 1)
            clearance(un, p0, p1, 3), nn = n[3][0];
        else
            return;
        presn = 0.4 * un[3] - 0.2 * un[1] * un[1] / un[0] - 0.2 * un[2] * un[2] / un[0],
        cxn[0] = un[1], cxn[1] = un[1] * un[1] / un[0] + presn, cxn[2] = un[1] * un[2] / un[0], cxn[3] = un[1] / un[0] * (un[3] + presn);
        alpha = sqrt((u[1] * u[1] + u[2] * u[2]) / (u[0] * u[0])) + sqrt(1.4 * pres / u[0]), alphan = sqrt((un[1] * un[1] + un[2] * un[2]) / (un[0] * un[0])) + sqrt(1.4 * presn / un[0]),
        alpha = max(alpha, alphan);
        for (int d = 0; d < 4; ++d)
            cx[d] = 0.5 * (cx[d] + cxn[d]) - 0.5 * alpha * nn * (u[d] - un[d]);
    }

    __device__ void convection_y(float cy[4], unsigned int p0, unsigned int p1)
    {
        float u[4], pres, alpha, nn, cyn[4], un[4], presn, alphan;
        u[0] = value[0][p0][p1], u[1] = value[1][p0][p1], u[2] = value[2][p0][p1], u[3] = value[3][p0][p1];
        pres = 0.4 * u[3] - 0.2 * u[1] * u[1] / u[0] - 0.2 * u[2] * u[2] / u[0],
        cy[0] = u[2], cy[1] = u[1] * u[2] / u[0], cy[2] = u[2] * u[2] / u[0] + pres, cy[3] = u[2] / u[0] * (u[3] + pres);
        if (p1 == P - 1)
            clearance(un, p0, p1, 0), nn = n[0][1];
        else if (p0 == 0)
            clearance(un, p0, p1, 1), nn = n[1][1];
        else if (p1 == 0)
            clearance(un, p0, p1, 2), nn = n[2][1];
        else if (p0 == P - 1)
            clearance(un, p0, p1, 3), nn = n[3][1];
        else
            return;
        presn = 0.4 * un[3] - 0.2 * un[1] * un[1] / un[0] - 0.2 * un[2] * un[2] / un[0],
        cyn[0] = un[2], cyn[1] = un[1] * un[2] / un[0], cyn[2] = un[2] * un[2] / un[0] + presn, cyn[3] = un[2] / un[0] * (un[3] + presn);
        alpha = sqrt((u[1] * u[1] + u[2] * u[2]) / (u[0] * u[0])) + sqrt(1.4 * pres / u[0]), alphan = sqrt((un[1] * un[1] + un[2] * un[2]) / (un[0] * un[0])) + sqrt(1.4 * presn / un[0]),
        alpha = max(alpha, alphan);
        for (int d = 0; d < 4; ++d)
            cy[d] = 0.5 * (cy[d] + cyn[d]) - 0.5 * alpha * nn * (u[d] - un[d]);
    }

    void caculation()
    {
        float l[4], ps[P], ws[P]; // position weight standard
        if (P == 4)
            ps[0] = -1.0, ps[1] = -0.57735026918963, ps[2] = +0.57735026918963, ps[3] = +1.0,
            ws[0] = 0.0, ws[1] = 1.0, ws[2] = 1.0, ws[3] = 0.0;
        else if (P == 8)
            ps[0] = -1.0, ps[1] = -0.9324695142031521, ps[2] = -0.6612093864662645, ps[3] = -0.2386191860831969, ps[4] = 0.2386191860831969, ps[5] = 0.6612093864662645, ps[6] = 0.9324695142031521, ps[7] = +1.0,
            ws[0] = 0.0, ws[1] = 0.1713244923791704, ws[2] = 0.3607615730481386, ws[3] = 0.4679139345726910, ws[4] = 0.4679139345726910, ws[5] = 0.3607615730481386, ws[6] = 0.1713244923791704, ws[7] = 0.0;
        else
            return;
        n[0][0] = q[0][1] - q[1][1], n[0][1] = q[1][0] - q[0][0], n[1][0] = q[1][1] - q[2][1], n[1][1] = q[2][0] - q[1][0],
        n[2][0] = q[2][1] - q[3][1], n[2][1] = q[3][0] - q[2][0], n[3][0] = q[3][1] - q[0][1], n[3][1] = q[0][0] - q[3][0];
        for (int i = 0; i < 4; ++i)
            l[i] = sqrt(n[i][0] * n[i][0] + n[i][1] * n[i][1]), n[i][0] /= l[i], n[i][1] /= l[i];

        float x0 = (q[0][0] - q[1][0] + q[2][0] - q[3][0]) / 4, x1 = (q[0][0] - q[1][0] - q[2][0] + q[3][0]) / 4, x2 = (q[0][0] + q[1][0] - q[2][0] - q[3][0]) / 4,
              y0 = (q[0][1] - q[1][1] + q[2][1] - q[3][1]) / 4, y1 = (q[0][1] - q[1][1] - q[2][1] + q[3][1]) / 4, y2 = (q[0][1] + q[1][1] - q[2][1] - q[3][1]) / 4,
              a = x1 * y0 - x0 * y1, b = x0 * y2 - x2 * y0, c = x1 * y2 - x2 * y1, x_xs, x_ys, y_xs, y_ys, jaco, j2, mass;

        int K = 1, pyramid = 0;
        while ((pyramid += K) < S)
            ++K;
        --K;
        for (int k = 0, s = 0; k <= K; ++k)
            for (int s0 = 0, s1; s1 = k - s0, s0 <= k; ++s, ++s0)
                for (int p0 = 0; p0 < P; ++p0)
                    for (int p1 = 0; p1 < P; ++p1)
                        x_xs = x0 * ps[p1] + x1, x_ys = x0 * ps[p0] + x2, y_xs = y0 * ps[p1] + y1, y_ys = y0 * ps[p0] + y2,
                        jaco = a * ps[p0] + b * ps[p1] + c, j2 = jaco * jaco, mass = (2.0 * s0 + 1) * (2.0 * s1 + 1) / 4,
                        base[s][p0][p1] = legendre(ps[p0], s0) * legendre(ps[p1], s1) * mass,
                        test[s][p0][p1] = legendre(ps[p0], s0) * legendre(ps[p1], s1) * ws[p0] * ws[p1],
                        base_x[s][p0][p1] = (degendre(ps[p0], s0) * legendre(ps[p1], s1) * y_ys + legendre(ps[p0], s0) * degendre(ps[p1], s1) * (-y_xs)) / jaco * mass,
                        base_y[s][p0][p1] = (degendre(ps[p0], s0) * legendre(ps[p1], s1) * (-x_ys) + legendre(ps[p0], s0) * degendre(ps[p1], s1) * x_xs) / jaco * mass,
                        test_x[s][p0][p1] = (degendre(ps[p0], s0) * legendre(ps[p1], s1) * jaco - legendre(ps[p0], s0) * legendre(ps[p1], s1) * a) / j2 * y_ys * ws[p0] * ws[p1] +
                                            (legendre(ps[p0], s0) * degendre(ps[p1], s1) * jaco - legendre(ps[p0], s0) * legendre(ps[p1], s1) * b) / j2 * (-y_xs) * ws[p0] * ws[p1],
                        test_y[s][p0][p1] = (degendre(ps[p0], s0) * legendre(ps[p1], s1) * jaco - legendre(ps[p0], s0) * legendre(ps[p1], s1) * a) / j2 * (-x_ys) * ws[p0] * ws[p1] +
                                            (legendre(ps[p0], s0) * degendre(ps[p1], s1) * jaco - legendre(ps[p0], s0) * legendre(ps[p1], s1) * b) / j2 * x_xs * ws[p0] * ws[p1];
        for (int k = 0, s = 0; k <= K; ++k)
            for (int s0 = 0, s1; s1 = k - s0, s0 <= k; ++s, ++s0)
                for (int p = 0; p < P; ++p)
                    jaco = a * ps[p] + b * ps[P - 1] + c,
                    test_x[s][p][P - 1] = legendre(ps[p], s0) * legendre(ps[P - 1], s1) / jaco * l[0] * n[0][0] * ws[p] / 2,
                    test_y[s][p][P - 1] = legendre(ps[p], s0) * legendre(ps[P - 1], s1) / jaco * l[0] * n[0][1] * ws[p] / 2,
                    jaco = a * ps[0] + b * ps[p] + c,
                    test_x[s][0][p] = legendre(ps[0], s0) * legendre(ps[p], s1) / jaco * l[1] * n[1][0] * ws[p] / 2,
                    test_y[s][0][p] = legendre(ps[0], s0) * legendre(ps[p], s1) / jaco * l[1] * n[1][1] * ws[p] / 2,
                    jaco = a * ps[p] + b * ps[0] + c,
                    test_x[s][p][0] = legendre(ps[p], s0) * legendre(ps[0], s1) / jaco * l[2] * n[2][0] * ws[p] / 2,
                    test_y[s][p][0] = legendre(ps[p], s0) * legendre(ps[0], s1) / jaco * l[2] * n[2][1] * ws[p] / 2,
                    jaco = a * ps[P - 1] + b * ps[p] + c,
                    test_x[s][P - 1][p] = legendre(ps[P - 1], s0) * legendre(ps[p], s1) / jaco * l[3] * n[3][0] * ws[p] / 2,
                    test_y[s][P - 1][p] = legendre(ps[P - 1], s0) * legendre(ps[p], s1) / jaco * l[3] * n[3][1] * ws[p] / 2;

        float x, y, xs, ys, xc = 5.0, yc = 5.0, xn, yn, rn, rho, u, v, temp, pres; // eddy value
        for (int p0 = 0; p0 < P; ++p0)
            for (int p1 = 0; p1 < P; ++p1)
                xs = ps[p0], ys = ps[p1],
                x = (q[0][0] * (1 + xs) * (1 + ys) + q[1][0] * (1 - xs) * (1 + ys) + q[2][0] * (1 - xs) * (1 - ys) + q[3][0] * (1 + xs) * (1 - ys)) / 4.0,
                y = (q[0][1] * (1 + xs) * (1 + ys) + q[1][1] * (1 - xs) * (1 + ys) + q[2][1] * (1 - xs) * (1 - ys) + q[3][1] * (1 + xs) * (1 - ys)) / 4.0,
                xn = x - xc, yn = y - yc, rn = xn * xn + yn * yn, u = 1.0 + 2.5 / pi * exp(0.5 * (1.0 - rn)) * (-yn), v = 1.0 + 2.5 / pi * exp(0.5 * (1.0 - rn)) * (+xn),
                temp = 1.0 - 10.0 / (11.2 * pi * pi) * exp(1.0 - rn), rho = pow(temp, 2.5), pres = pow(rho, 1.4), pos[0][p0][p1] = x, pos[1][p0][p1] = y,
                value[0][p0][p1] = rho, value[1][p0][p1] = rho * u, value[2][p0][p1] = rho * v, value[3][p0][p1] = 2.5 * pres + 0.5 * rho * (u * u + v * v);

        for (int d = 0; d < 4; ++d)
            for (int s = 0; s < S; ++s)
                spect[0][d][s] = 0;
        for (int d = 0; d < 4; ++d)
            for (int s = 0; s < S; ++s)
                for (int p0 = 0; p0 < P; ++p0)
                    for (int p1 = 0; p1 < P; ++p1)
                        spect[0][d][s] += value[d][p0][p1] * test[s][p0][p1];
        // for (int d = 0; d < 4; ++d)
        //     for (int p0 = 0; p0 < P; ++p0)
        //         for (int p1 = 0; p1 < P; ++p1)
        //             value[d][p0][p1] = 0, value_x[d][p0][p1] = 0, value_y[d][p0][p1] = 0;
        // for (int d = 0; d < 4; ++d)
        //     for (int p0 = 0; p0 < P; ++p0)
        //         for (int p1 = 0; p1 < P; ++p1)
        //             for (int s = 0; s < S; ++s)
        //                 value[d][p0][p1] += spect[0][d][s] * base[s][p0][p1], value_x[d][p0][p1] += spect[0][d][s] * base_x[s][p0][p1], value_y[d][p0][p1] += spect[0][d][s] * base_y[s][p0][p1];
    }
};

struct block
{
    cubit cbt[N0][N1];
    float coord[N0 + 1][N1 + 1][2];
    float layer[N0 * P][N1 * P][7];

    void map(float a0, float b0, float a1, float b1)
    {
        float h0 = (b0 - a0) / N0, h1 = (b1 - a1) / N1;
        for (int n0 = 0; n0 <= N0; ++n0)
            for (int n1 = 0; n1 <= N1; ++n1)
                coord[n0][n1][0] = a0 + n0 * h0, coord[n0][n1][1] = a1 + n1 * h1;
    }
    void map(std::string file)
    {
        float trash;
        std::ifstream glasses(file, std::ios::in);
        glasses >> trash >> trash;
        for (int n0 = 0; n0 <= N0; ++n0)
            for (int n1 = 0; n1 <= N1; ++n1)
                glasses >> coord[n0][n1][0] >> coord[n0][n1][1] >> trash;
    }

    void reality() /// ma re
    {
        for (int n0 = 0; n0 < N0; ++n0)
            for (int n1 = 0; n1 < N1; cbt[n0][n1].caculation(), ++n1)
                for (int s = 0; s < 2; ++s)
                    cbt[n0][n1].q[0][s] = coord[n0 + 1][n1 + 1][s], cbt[n0][n1].q[1][s] = coord[n0][n1 + 1][s], cbt[n0][n1].q[2][s] = coord[n0][n1][s], cbt[n0][n1].q[3][s] = coord[n0 + 1][n1][s];
    }

    void paraview(int interval)
    {
        static int timing = -1;
        ++timing;
        if (timing % interval)
            return;
        for (int n0 = 0; n0 < N0; ++n0)
            for (int n1 = 0; n1 < N1; ++n1)
                for (int p0 = 0; p0 < P; ++p0)
                    for (int p1 = 0; p1 < P; ++p1)
                        layer[n0 * P + p0][n1 * P + p1][0] = cbt[n0][n1].pos[0][p0][p1], layer[n0 * P + p0][n1 * P + p1][1] = cbt[n0][n1].pos[1][p0][p1], layer[n0 * P + p0][n1 * P + p1][2] = 0,
                                                    layer[n0 * P + p0][n1 * P + p1][3] = cbt[n0][n1].value[0][p0][p1], layer[n0 * P + p0][n1 * P + p1][4] = cbt[n0][n1].value[1][p0][p1],
                                                    layer[n0 * P + p0][n1 * P + p1][5] = cbt[n0][n1].value[2][p0][p1], layer[n0 * P + p0][n1 * P + p1][6] = cbt[n0][n1].value[3][p0][p1];
        std::string file, title = "x,y,z,u0,u1,u2,u3";
        if (timing > 9999)
            file = "view_9999.csv";
        else if (timing > 999)
            file = "view_" + std::to_string(timing) + ".csv";
        else if (timing > 99)
            file = "view_0" + std::to_string(timing) + ".csv";
        else if (timing > 9)
            file = "view_00" + std::to_string(timing) + ".csv";
        else
            file = "view_000" + std::to_string(timing) + ".csv";
        std::ofstream pen(file, std::ios::out | std::ios::trunc);
        pen << title << std::endl;
        for (int p1 = 0; p1 < N1 * P; ++p1)
            for (int p0 = 0; p0 < N0 * P; ++p0)
                pen << layer[p0][p1][0] << "," << layer[p0][p1][1] << "," << layer[p0][p1][2] << ","
                    << layer[p0][p1][3] << "," << layer[p0][p1][4] << "," << layer[p0][p1][5] << "," << layer[p0][p1][6] << std::endl;
    }
};

__global__ void spect_to_value(int r, block *blk)
{
    cubit &cbt = blk->cbt[blockIdx.x][blockIdx.y];
    for (int d = 0; d < 4; ++d)
        cbt.value[d][threadIdx.x][threadIdx.y] = 0, cbt.value_x[d][threadIdx.x][threadIdx.y] = 0, cbt.value_y[d][threadIdx.x][threadIdx.y] = 0;
    for (int d = 0; d < 4; ++d)
        for (int s = 0; s < S; ++s)
            cbt.value[d][threadIdx.x][threadIdx.y] += cbt.spect[r][d][s] * cbt.base[s][threadIdx.x][threadIdx.y],
                cbt.value_x[d][threadIdx.x][threadIdx.y] += cbt.spect[r][d][s] * cbt.base_x[s][threadIdx.x][threadIdx.y],
                cbt.value_y[d][threadIdx.x][threadIdx.y] += cbt.spect[r][d][s] * cbt.base_y[s][threadIdx.x][threadIdx.y];
}

__global__ void value_to_spect(int r, block *blk)
{
    cubit &cbt = blk->cbt[blockIdx.x][blockIdx.y];
    if (threadIdx.x == 0 && threadIdx.y == 0)
        for (int d = 0; d < 4; ++d)
            for (int s = 0; s < S; ++s)
                cbt.spect[r][d][s] = 0;

    float cx[4], cy[4];
    cbt.convection_x(cx, threadIdx.x, threadIdx.y), cbt.convection_y(cy, threadIdx.x, threadIdx.y);

    __shared__ float cx_s[4][P][P], cy_s[4][P][P];
    for (int d = 0; d < 4; ++d)
        cx_s[d][threadIdx.x][threadIdx.y] = cx[d], cy_s[d][threadIdx.x][threadIdx.y] = cy[d];
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
        for (int d = 0; d < 4; ++d)
            for (int s = 0; s < S; ++s)
                for (int p0 = 0; p0 < P; ++p0)
                    for (int p1 = 0; p1 < P; ++p1)
                        cbt.spect[r][d][s] += cx_s[d][p0][p1] * cbt.test_x[s][p0][p1], cbt.spect[r][d][s] += cy_s[d][p0][p1] * cbt.test_y[s][p0][p1];
}

__global__ void rk(int r, float dt, block *blk)
{
    cubit &cbt = blk->cbt[blockIdx.x][blockIdx.y];
    if (r == 1)
        for (int d = 0; d < 4; ++d)
            for (int s = 0; s < S; ++s)
                cbt.spect[1][d][s] = cbt.spect[0][d][s] + cbt.spect[1][d][s] * dt;
    if (r == 2)
        for (int d = 0; d < 4; ++d)
            for (int s = 0; s < S; ++s)
                cbt.spect[2][d][s] = (3.0 / 4.0) * cbt.spect[0][d][s] + (1.0 / 4.0) * cbt.spect[1][d][s] + (1.0 / 4.0) * cbt.spect[2][d][s] * dt;
    if (r == 3)
        for (int d = 0; d < 4; ++d)
            for (int s = 0; s < S; ++s)
                cbt.spect[0][d][s] = (1.0 / 3.0) * cbt.spect[0][d][s] + (2.0 / 3.0) * cbt.spect[2][d][s] + (2.0 / 3.0) * cbt.spect[1][d][s] * dt;
}

__global__ void construction(block *blk)
{
    cubit &cbt = blk->cbt[blockIdx.x][blockIdx.y];
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        cbt.idx[0] = &(blk->cbt[blockIdx.x][blockIdx.y + 1]), cbt.idx[1] = &(blk->cbt[blockIdx.x - 1][blockIdx.y]),
        cbt.idx[2] = &(blk->cbt[blockIdx.x][blockIdx.y - 1]), cbt.idx[3] = &(blk->cbt[blockIdx.x + 1][blockIdx.y]);
        if (blockIdx.x == 0)
            cbt.idx[1] = &(blk->cbt[N0 - 1][blockIdx.y]);
        if (blockIdx.x == N0 - 1)
            cbt.idx[3] = &(blk->cbt[0][blockIdx.y]);
        if (blockIdx.y == 0)
            cbt.idx[2] = &(blk->cbt[blockIdx.x][N1 - 1]);
        if (blockIdx.y == N1 - 1)
            cbt.idx[0] = &(blk->cbt[blockIdx.x][0]);
    }
}

int main()
{
    dim3 B(N0, N1), G(P, P);
    block *blk_h = new block(), *blk_d;
    blk_h->map(0, 10, 0, 10), blk_h->reality(), blk_h->paraview(1), cudaMalloc((void **)&blk_d, sizeof(*blk_h)), cudaMemcpy(blk_d, blk_h, sizeof(*blk_h), cudaMemcpyHostToDevice);
    construction<<<B, G>>>(blk_d), cudaDeviceSynchronize();
    while (timer < end)
        timer = timer + dt, value_to_spect<<<B, G>>>(1, blk_d), cudaDeviceSynchronize(),
        rk<<<B, G>>>(1, dt, blk_d), cudaDeviceSynchronize(), spect_to_value<<<B, G>>>(1, blk_d), cudaDeviceSynchronize(), value_to_spect<<<B, G>>>(2, blk_d), cudaDeviceSynchronize(),
        rk<<<B, G>>>(2, dt, blk_d), cudaDeviceSynchronize(), spect_to_value<<<B, G>>>(2, blk_d), cudaDeviceSynchronize(), value_to_spect<<<B, G>>>(1, blk_d), cudaDeviceSynchronize(),
        rk<<<B, G>>>(3, dt, blk_d), cudaDeviceSynchronize(), spect_to_value<<<B, G>>>(0, blk_d), cudaMemcpy(blk_h, blk_d, sizeof(*blk_d), cudaMemcpyDeviceToHost),
        printf("%f\n", timer), blk_h->paraview(100);
}
