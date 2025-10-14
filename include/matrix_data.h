// Auto-generated from matrix_data.txt
#pragma once
#ifdef INCLUDE_MATRIX_DATA

/* Compile-time constants */
constexpr const int A_m = 5, A_n = 4;
constexpr const int B_m = 4, B_n = 7;
constexpr const int WIDTH = 4, HEIGHT = 3, offset = 3;

/* Test matrices used for demonstration and validation */
static const int A_host[5][4] = {
    {0, 1, -2, 7},
    {4, 0, 3, -9},
    {-2, 6, 0, 2},
    {4, -1, 12, 0},
    {-1, 2, 3, 4}
};

static const int B_host[4][7] = {
    {-1, 1, -2, 0, 4, 6, -1},
    {3, 0, 3, 1, 7, -3, 9},
    {2, 12, 3, 1, -1, 23, 0},
    {8, -5, -1, 4, 12, 22, -3}
};
#endif
