#include <stdio.h>
#include <stdlib.h>
#include "dv4l.h"

int main(int argc, char* argv[])
{
    dv4l_t camera;

    dv4l_fourcc_t pixelformat_input  = {.u = 0};
    dv4l_fourcc_t pixelformat_output = {.u = V4L2_PIX_FMT_BGR24};


    int W = -1;
    int H = -1;

    if(!dv4l_init(&camera,
                  "/dev/video0",

                  W,H,

                  // fps
                  1,

                  // stream
                  true,

                  pixelformat_input,
                  pixelformat_output,

                  // controls
                  NULL, 0))
        return 1;


    W = camera.format.fmt.pix.width;
    H = camera.format.fmt.pix.height;

    char* buf = malloc(W*H*3);
    if(buf == NULL)
        return 1;

    for(int i=0; i<5; i++)
    {
        if(!dv4l_getframe(&camera, buf, NULL))
            return 1;

        char filename[128];
        sprintf(filename, "/tmp/frame%d.ppm", i);

        FILE* fp = fopen(filename, "w");
        fprintf(fp, "P6\n%d %d\n255\n", W, H);
        fwrite(buf, 1, W*H*3, fp);
        fclose(fp);
    }

    return 0;
}
