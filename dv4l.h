#pragma once

#warning MOVE THESE TO THE C FILE

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>


#define NUM_STREAMING_BUFFERS_MAX 16

typedef struct
{
    int                        fd;
    v4l2_format                format;
    bool                       streaming  : 1;
    bool                       want_color : 1;
    // used if streaming
    int   mmapped_buf_length[NUM_STREAMING_BUFFERS_MAX];
    void* mmapped_buf       [NUM_STREAMING_BUFFERS_MAX];

    // used if NOT streaming.
    int            allocated_buf_length;
    unsigned char* allocated_buf;

    // If the data needs decoding (not just unpacking), this library does that.
    // Most commonly this is done with MJPEG. Much of the time this isn't
    // needed, and all of these will be NULL
    AVCodec*        av_codec;
    AVCodecContext* av_codec_context;
    AVFrame*        av_frame;
    AVPacket*       av_packet;

    // Used to interpret the pixel data coming out of the camera. Often it's in
    // some sort of packed YUV format, and this library converts it to a plain
    // matrix of data. Optionally, this library can also scale and crop, but I'm
    // not using it for that currently. This is needed at all times. Even if the
    // camera already spits out unpacked data I still use the swscale library
    // for simplicity. It's just a memcpy() in that case
    SwsContext*     sws_context;

} dv4l_t;

typedef struct
{
    uint32_t pixelformat;
    enum {
        USE_REQUESTED_PIXELFORMAT,
        BEST_COLOR_PIXELFORMAT,
        BEST_GRAYSCALE_PIXELFORMAT
    } choice;
} dv4l_pixelformat_choice_t;

bool dv4l_init(// out
               dv4l_t* camera,

               // in
               const char*               device,

               // <0 means "biggest possible"
               int                       width_requested,
               int                       height_requested,

               // <0 means "fastest possible"
               int                       fps_requested,

               // If true: we ask for data to be given to us, according to
               // fps_requested. If false: we read() a frame whenever we like
               bool streaming_requested,

               // We either ask for a specific pixel format from the camera, or
               // we pick the best one we've got
               dv4l_pixelformat_choice_t pixelformat_requested,

               const struct v4l2_control* controls,
               const int Ncontrols);

void dv4l_deinit(dv4l_t* camera);

// Throw out all available frames, so that the next one that comes in is the
// latest one
bool dv4l_flush_frames(dv4l_t* camera)

bool dv4l_getframe(dv4l_t* camera,
                   char* image,
                   uint64_t* timestamp_us);

