#include <stdio.h>
#include <stdint.h>

int main() {
    union {
        int32_t i;
        float f;
    } un;

    un.i = 0x413FA000 | 0x80000000;
    //un.i = 0x453FA2DC;

    printf("integer: %d\n" , un.i);
    printf("float:   %.50f\n" , un.f);
    printf("long double (quadruple) size = %ld\n bits", sizeof(long double) * 8);

    double x = 73.66452775641295147580;

    printf("\n\n%.20f\n%.20lf\n", x, (float)x);
    return 0;
}