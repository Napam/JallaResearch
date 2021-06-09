#include "stdio.h"
#include "stdint.h"
#include "inttypes.h"
#include <sys/time.h>
#include "unistd.h"

long long timeInMilliseconds(void) {
    struct timeval tv;

    gettimeofday(&tv,NULL);
    return (((long long)tv.tv_sec)*1000)+(tv.tv_usec/1000);
}

int main (int argc, char **argv)
{
    printf("HACK!\n");
    long long t;
    uintmax_t i = 0;

    t = timeInMilliseconds();
    while (1) {
        if ((timeInMilliseconds() - t) > (long long)(100)) {
            printf("\33[2K\r%ld", i);
            fflush(stdout);
            t = timeInMilliseconds();
        }
        i++;
    }
    return 0;
}
