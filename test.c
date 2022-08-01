#include <stdio.h>
#include <stdlib.h>
#include "dv4l.h"

int main(int argc, char* argv[])
{
    dv4l_t camera;

    dv4l_pixelformat_choice_t format_choice =
        {.choice = BEST_GRAYSCALE_PIXELFORMAT};


    const int W = 640;
    const int H = 480;

    if(!dv4l_init(&camera,
                  "/dev/video0",

                  W,H,

                  // fps
                  10,

                  // stream
                  true,

                  format_choice,

                  // controls
                  NULL, 0))
        return 1;


    char* buf = malloc(W*H);
    if(buf == NULL)
        return 1;

    for(int i=0; i<5; i++)
    {
        if(!dv4l_getframe(camera, buf, NULL))
            return 1;

        char filename[128];
        sprintf(filename, "/tmp/frame%d.pgm", i);

        FILE* fp = fopen(filename, "w");
        fprintf(fp, "P2\n%d %d\n255\n", W, H);
        fwrite(buf, 1, W*H, fp);
        fclose(fp);
    }

    return 0;
}
