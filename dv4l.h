#pragma once

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <linux/videodev2.h>

#include <stdbool.h>

#define NUM_STREAMING_BUFFERS_MAX 16

typedef union
{
    uint32_t u;
    char     s[4];
} dv4l_fourcc_t;

typedef struct
{
    int                fd;
    struct v4l2_format format;
    dv4l_fourcc_t      pixelformat_input;
    dv4l_fourcc_t      pixelformat_output;
    enum AVPixelFormat av_pixelformat_input;
    enum AVPixelFormat av_pixelformat_output;

    bool streaming : 1;

    // used if streaming
    int   mmapped_buf_length[NUM_STREAMING_BUFFERS_MAX];
    void* mmapped_buf       [NUM_STREAMING_BUFFERS_MAX];

    // used if NOT streaming.
    int            allocated_buf_length;
    unsigned char* allocated_buf;

    // If the data needs decoding (not just unpacking), this library does that.
    // Most commonly this is done with MJPEG. Much of the time this isn't
    // needed, and all of these will be NULL
    const AVCodec*  av_codec;
    AVCodecContext* av_codec_context;
    AVFrame*        av_frame;
    AVPacket*       av_packet;

    // Used to interpret the pixel data coming out of the camera. Often it's in
    // some sort of packed YUV format, and this library converts it to a plain
    // matrix of data. Optionally, this library can also scale and crop, but I'm
    // not using it for that currently. This is needed at all times. Even if the
    // camera already spits out unpacked data I still use the swscale library
    // for simplicity. It's just a memcpy() in that case
    struct SwsContext* sws_context;

} dv4l_t;

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

               // The input pixel format is a suggestion. The driver may refuse
               // or it may select something else (a warning will be printed in
               // that case). In the special case of pixelformat_input.u == 0,
               // we pick the first available format
               dv4l_fourcc_t pixelformat_input,
               dv4l_fourcc_t pixelformat_output,

               const struct v4l2_control* controls,
               const int Ncontrols);

void dv4l_deinit(dv4l_t* camera);

// Throw out all available frames, so that the next one that comes in is the
// latest one
bool dv4l_flush_frames(dv4l_t* camera);

bool dv4l_getframe(dv4l_t* camera,

                   // assumed to be large-enough to hold a densely stored image
                   // containing pixelformat_output data
                   char* image,
                   uint64_t* timestamp_us);

