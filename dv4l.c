#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <poll.h>

#include <asm/types.h>
#include <linux/videodev2.h>
#include <libavutil/imgutils.h>

#include "dv4l.h"



#define MSG(fmt, ...) fprintf(stderr, "%s(%d) at %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)


// These macros all need a local cleanup() function to be available to run on
// failure
#define ENSURE_DETAILEDERR(what, fmt, ...) do {                         \
    if(!(what))                                                         \
    {                                                                   \
        MSG("Failed '" #what "'. " fmt , ## __VA_ARGS__);               \
        cleanup();                                                      \
        return false;                                                   \
    } } while(0)
#define ENSURE(what) ENSURE_DETAILEDERR(what, "")

#define ENSURE_IOCTL_DETAILEDERR(fd, request, arg, fmt, ...) do {       \
        ENSURE_DETAILEDERR( ioctl_persistent(fd,request,arg) >= 0,                  \
                "Couldn't " #request ": %s" fmt,                        \
                strerrordesc_np(errno), ## __VA_ARGS__);                \
    } while(0)

#define ENSURE_IOCTL(fd, request, arg) ENSURE_IOCTL_DETAILEDERR(fd,request,arg,"")


static
int ioctl_persistent( int fd, unsigned long request, void* arg)
{
    int r;

    do
    {
        r = ioctl( fd, request, arg);
    } while( r == -1 && errno == EINTR);

    return r;
}

static
enum AVPixelFormat pixelformat_av_from_v4l(dv4l_fourcc_t fmt)
{
    switch(fmt.u)
    {
    /* RGB formats */
    /* 32  BGR-8-8-8-8 */
    case V4L2_PIX_FMT_BGR32:   return AV_PIX_FMT_BGR32;
    /* 32  RGB-8-8-8-8 */
    case V4L2_PIX_FMT_RGB32:   return AV_PIX_FMT_RGB32;
    /* 24  BGR-8-8-8 */
    case V4L2_PIX_FMT_BGR24:   return AV_PIX_FMT_BGR24;
    /* 24  RGB-8-8-8 */
    case V4L2_PIX_FMT_RGB24:   return AV_PIX_FMT_RGB24;
    /* 16  RGB-5-5-5 */
    case V4L2_PIX_FMT_RGB555:  return AV_PIX_FMT_RGB555;
    /* 16  RGB-5-6-5 */
    case V4L2_PIX_FMT_RGB565:  return AV_PIX_FMT_RGB565;

    /* Palette formats */
    /*  8  8-bit palette */
    case V4L2_PIX_FMT_PAL8:    return AV_PIX_FMT_PAL8;

    /* Luminance+Chrominance formats */
    /* 16  YUV 4:2:2 */
    case V4L2_PIX_FMT_YUYV:    return AV_PIX_FMT_YUYV422;
    /* 16  YUV 4:2:2 */
    case V4L2_PIX_FMT_UYVY:    return AV_PIX_FMT_UYVY422;
    /* 16  YVU422 planar */
    case V4L2_PIX_FMT_YUV422P: return AV_PIX_FMT_YUV422P;
    /* 16  YVU411 planar */
    case V4L2_PIX_FMT_YUV411P: return AV_PIX_FMT_YUV411P;
    /* 12  YUV 4:1:1 */
    case V4L2_PIX_FMT_Y41P:    return AV_PIX_FMT_UYYVYY411;
    /* 12  YUV 4:2:0 */
    case V4L2_PIX_FMT_YUV420:  return AV_PIX_FMT_YUV420P;
    /*  9  YUV 4:1:0 */
    case V4L2_PIX_FMT_YUV410:  return AV_PIX_FMT_YUV410P;

    /* two planes -- one Y: one Cr + Cb interleaved */
    /* 12  Y/CbCr 4:2:0 */
    case V4L2_PIX_FMT_NV12:    return AV_PIX_FMT_NV12;
    /* 12  Y/CrCb 4:2:0 */
    case V4L2_PIX_FMT_NV21:    return AV_PIX_FMT_NV21;
    /* 16  Y/CbCr 4:2:2 */
    case V4L2_PIX_FMT_NV16:    return AV_PIX_FMT_YUV422P;

    /* Grey formats */
    /* 16  Greyscale */
    case V4L2_PIX_FMT_Y16:     return AV_PIX_FMT_GRAY16LE;
    /*  8  Greyscale */
    case V4L2_PIX_FMT_GREY:    return AV_PIX_FMT_GRAY8;
    }

    MSG("Selected pixel format \"%.4s\" cannot be decoded by libswscale. Giving up",
        fmt.s);
    return AV_PIX_FMT_NONE;
}

static
bool decoder_init(// out
                  dv4l_t* camera,
                  // in
                  int width, int height)
{
    // for ENSURE()
    void cleanup(void)
    {
    }

    // swscale can't interpret the pixel format directly. Can avcodec do it and
    // THEN feed swscale?
    if(camera->pixelformat_input.u == V4L2_PIX_FMT_JPEG)
    {
        ENSURE(NULL != (camera->av_packet        = av_packet_alloc()));
        ENSURE(NULL != (camera->av_codec         = avcodec_find_decoder(AV_CODEC_ID_MJPEG)));
        ENSURE(NULL != (camera->av_codec_context = avcodec_alloc_context3(camera->av_codec)));
        ENSURE(avcodec_open2(camera->av_codec_context, camera->av_codec, NULL) >= 0);
        ENSURE(NULL != (camera->av_frame         = av_frame_alloc()));

        // I have now set up the decoder and need to set up the scaler. Sadly
        // libavcodec doesn't give me the output pixel format until we've
        // decoded at least one frame so I can't set up the scaler here.
        return true;
    }


    ENSURE_DETAILEDERR(camera->av_pixelformat_input != AV_PIX_FMT_NONE,
                       "I have no AV pixel format and no decoder either. Probably I need some not-yet-implemented decoder. Input fourcc is \"%.4s\"",
                       camera->pixelformat_input.s);

    ENSURE(NULL !=
           (camera->sws_context =
            sws_getContext(// source
                           width, height, camera->av_pixelformat_input,

                           // destination
                           width, height, camera->av_pixelformat_output,

                           // misc stuff
                           SWS_POINT, NULL, NULL, NULL)));

    return true;
}


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
               bool                      streaming_requested,

               // The input pixel format is a suggestion. The driver may refuse
               // or it may select something else (a warning will be printed in
               // that case). In the special case of pixelformat_input.u == 0,
               // we try to pick the best format
               dv4l_fourcc_t pixelformat_input,
               dv4l_fourcc_t pixelformat_output,

               const struct v4l2_control* controls,
               const int Ncontrols)
{
    // for ENSURE()
    void cleanup(void)
    {
        dv4l_deinit(camera);
    }


    *camera = (dv4l_t){ .pixelformat_input  = pixelformat_input,
                        .pixelformat_output = pixelformat_output};

    ENSURE_DETAILEDERR(camera->pixelformat_output.u == V4L2_PIX_FMT_BGR24 ||
                       camera->pixelformat_output.u == V4L2_PIX_FMT_RGB24 ||
                       camera->pixelformat_output.u == V4L2_PIX_FMT_GREY ||
                       camera->pixelformat_output.u == V4L2_PIX_FMT_Y16,
                       "I only support Y8, Y16, BGR and RGB output pixel formats");

    camera->av_pixelformat_output = pixelformat_av_from_v4l(camera->pixelformat_output);
    ENSURE(camera->av_pixelformat_output != AV_PIX_FMT_NONE);

    ENSURE_DETAILEDERR( (camera->fd = open( device, O_RDWR, 0)) >= 0,
            "Couldn't open video device '%s': %s",
            device,
            strerrordesc_np(errno));

    struct v4l2_capability cap;
    ENSURE_IOCTL(camera->fd, VIDIOC_QUERYCAP, &cap);

    ENSURE( cap.capabilities & V4L2_CAP_VIDEO_CAPTURE);

    camera->streaming = streaming_requested;
    if(streaming_requested)
        ENSURE_DETAILEDERR( cap.capabilities & V4L2_CAP_STREAMING,
                "The caller requested streaming, but it's not available. read() is %savailable",
                (cap.capabilities & V4L2_CAP_READWRITE) ? "" : "NOT " );
    else
        ENSURE_DETAILEDERR( cap.capabilities & V4L2_CAP_READWRITE,
                "The caller requested read() access (no streaming), but it's not available. streaming is %savailable",
                (cap.capabilities & V4L2_CAP_STREAMING) ? "" : "NOT " );

    if(camera->pixelformat_input.u == 0)
    {
        // The caller is asking us to pick a pixel format. I take the first one
        struct v4l2_fmtdesc fmtdesc = {.index = 0,
                                       .type  = V4L2_BUF_TYPE_VIDEO_CAPTURE};
        ENSURE_IOCTL( camera->fd, VIDIOC_ENUM_FMT, &fmtdesc);
        camera->pixelformat_input.u = fmtdesc.pixelformat;
    }

    // If asked for the biggest possible images, I ask the driver for an
    // unreasonable upper bound of what I want. The driver will change the
    // passed-in parameters to whatever it is actually capable of
    camera->format =
        (struct v4l2_format)
        {.type    = V4L2_BUF_TYPE_VIDEO_CAPTURE,
         .fmt.pix = {.width       = width_requested  <= 0 ? 100000 : width_requested,
                     .height      = height_requested <= 0 ? 100000 : height_requested,
                     .field       = V4L2_FIELD_NONE} }; // no interlacing

    dv4l_fourcc_t* pixelformat_did_set =
        (dv4l_fourcc_t*)&camera->format.fmt.pix.pixelformat;
    *pixelformat_did_set = camera->pixelformat_input;

    ENSURE_IOCTL(camera->fd, VIDIOC_S_FMT, &camera->format);
    if(pixelformat_did_set->u != camera->pixelformat_input.u)
    {
        MSG("Warning: asked for pixel format \"%.4s\" but V4L2 gave us \"%.4s\" instead. Continuing",
            camera->pixelformat_input.s,
            pixelformat_did_set->s);
    }
    else
    {
        MSG("Info: input pixel format: \"%.4s\"",
            camera->pixelformat_input.s);
    }

    camera->pixelformat_input = *pixelformat_did_set;
    camera->av_pixelformat_input = pixelformat_av_from_v4l(camera->pixelformat_input);
    // "camera->av_pixelformat_input == AV_PIX_FMT_NONE" means we need to decode
    // the video stream.

    ENSURE_DETAILEDERR(camera->format.fmt.pix.field == V4L2_FIELD_NONE,
           "interlacing not yet supported");

    if(controls != NULL)
        for(int i=0; i<Ncontrols; i++)
            ENSURE_IOCTL_DETAILEDERR(camera->fd, VIDIOC_S_CTRL, (struct v4l2_control*)&controls[i],
                                     "Error setting control %d to value %d",
                                     controls[i].id, controls[i].value);

    if( camera->streaming )
    {
        if( fps_requested > 0 )
        {
            struct v4l2_streamparm parm =
                {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
                 .parm.capture.timeperframe = {.numerator   = 1,
                                               .denominator = fps_requested, } };
            ENSURE_IOCTL_DETAILEDERR(camera->fd, VIDIOC_S_PARM, &parm,
                                     "Error setting FPS = %d",
                                     fps_requested);
        }


        struct v4l2_requestbuffers rb =
            {.count  = NUM_STREAMING_BUFFERS_MAX,
             .type   = V4L2_BUF_TYPE_VIDEO_CAPTURE,
             .memory = V4L2_MEMORY_MMAP};
        ENSURE_IOCTL_DETAILEDERR(camera->fd, VIDIOC_REQBUFS, &rb,
                     "Error requesting %d buffers",
                     NUM_STREAMING_BUFFERS_MAX);
        ENSURE_DETAILEDERR( rb.count > 0,
                "Couldn't get any of the buffers I asked for: %s",
                strerrordesc_np(errno));

        for (unsigned int i = 0; i < rb.count; i++)
        {
            struct v4l2_buffer buf = {.index  = i,
                                      .type   = rb.type,
                                      .memory = rb.memory};
            ENSURE_IOCTL(camera->fd, VIDIOC_QUERYBUF, &buf);
            camera->mmapped_buf_length[i] = buf.length;
            camera->mmapped_buf       [i] =
                mmap(0,
                     buf.length, PROT_READ, MAP_SHARED, camera->fd,
                     buf.m.offset);
            ENSURE(camera->mmapped_buf[i] != MAP_FAILED);

            ENSURE_IOCTL(camera->fd, VIDIOC_QBUF, &buf);
        }

        ENSURE_IOCTL(camera->fd, VIDIOC_STREAMON, &(int){V4L2_BUF_TYPE_VIDEO_CAPTURE});
    }
    else
    {
        // When allocating the memory buffer, I leave room for padding. FFMPEG documentation
        // (avcodec.h):
        //  * @warning The input buffer must be \c AV_INPUT_BUFFER_PADDING_SIZE larger than
        //  * the actual read bytes because some optimized bitstream readers read 32 or 64
        //  * bits at once and could read over the end.
        camera->allocated_buf_length =
            camera->format.fmt.pix.sizeimage + AV_INPUT_BUFFER_PADDING_SIZE;
        ENSURE((camera->allocated_buf = malloc(camera->allocated_buf_length)) != NULL);
    }

    ENSURE(decoder_init(camera,
                        camera->format.fmt.pix.width,
                        camera->format.fmt.pix.height));

    return true;
}

void dv4l_deinit(dv4l_t* camera)
{
    if( camera->fd > 0 )
    {
        close( camera->fd );
        camera->fd = -1;
    }

    for(unsigned int i=0;
        i<sizeof(camera->mmapped_buf)/sizeof(camera->mmapped_buf[0]);
        i++)
        if(camera->mmapped_buf[i])
        {
            munmap(camera->mmapped_buf[i], camera->mmapped_buf_length[i]);
            camera->mmapped_buf[i]        = NULL;
            camera->mmapped_buf_length[i] = 0;
        }

    if(camera->allocated_buf)
    {
        free(camera->allocated_buf);
        camera->allocated_buf = NULL;
    }

    // I think I don't need to deallocate av_codec. It's a part of the
    // av_codec_context. I think?

    if(camera->av_codec_context)
    {
        avcodec_close(camera->av_codec_context);
        avcodec_free_context(&camera->av_codec_context);
        camera->av_codec_context = NULL;
    }

    if(camera->av_frame)
    {
        av_frame_free(&camera->av_frame);
        camera->av_frame = NULL;
    }

    if(camera->av_packet)
    {
        av_packet_free(&camera->av_packet);
        camera->av_packet = NULL;
    }

    if(camera->sws_context)
    {
        sws_freeContext(camera->sws_context);
        camera->sws_context = NULL;
    }
}

// Throw out all available frames, so that the next one that comes in is the
// latest one
bool dv4l_flush_frames(dv4l_t* camera)
{
    // for ENSURE()
    void cleanup(void)
    {
    }


    while(1)
    {
        struct pollfd fd =
            {.fd     = camera->fd,
             .events = POLLIN};
        int num_have_data;

        ENSURE_DETAILEDERR(0 > (num_have_data = poll(&fd, 1, 0)),
               "poll() failed reading the camera: %s!",
               strerrordesc_np(errno));

        // if no data is available, get the next frame that comes in
        if( num_have_data == 0 )
            return true;

        // There are frames to read, so I flush the queues
        if( !camera->streaming )
        {
            // I try to read a bunch of data and throw it away. I know it won't
            // block because poll() said so. Then I poll() again until it says
            // there's nothing left
            read(camera->fd, camera->allocated_buf, camera->allocated_buf_length);
        }
        else
        {
            // I DQBUF/QBUF a frame and throw it away. I know it won't block
            // because poll() said so. Then I poll() again until it says there's
            // nothing left
            struct v4l2_buffer v4l2_buf =
                {.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE,
                 .memory = V4L2_MEMORY_MMAP};
            ENSURE_IOCTL(camera->fd, VIDIOC_DQBUF, &v4l2_buf);
            ENSURE_IOCTL(camera->fd, VIDIOC_QBUF,  &v4l2_buf);
        }
    }

    // will not get here
    return true;
}

bool dv4l_getframe(dv4l_t* camera,

                   // assumed to be large-enough to hold a densely stored image
                   // containing pixelformat_output data
                   char* image,
                   uint64_t* timestamp_us)
{
    // for ENSURE()
    struct v4l2_buffer v4l2_buf =
    {
        .type   = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        .memory = V4L2_MEMORY_MMAP
    };


    bool need_to_requeue_buffer = false;

    bool cleanup(void)
    {
        // So that THIS ensure call doesn't loop
        void cleanup(void) { }

        if(need_to_requeue_buffer)
            ENSURE_IOCTL(camera->fd, VIDIOC_QBUF, &v4l2_buf);
        return true;
    }



    int      Nbytes_frame;
    uint8_t* bytes_frame;

    if( !camera->streaming )
    {
        Nbytes_frame = camera->format.fmt.pix.sizeimage;

        ENSURE_DETAILEDERR(0 > read( camera->fd, camera->allocated_buf, Nbytes_frame),
               "camera read() returned errno %s",
               strerrordesc_np(errno));

        bytes_frame = camera->allocated_buf;
        if(timestamp_us != NULL)
            // I don't know when the frame came in. I read it later
            *timestamp_us = 0;
    }
    else
    {
        ENSURE_IOCTL(camera->fd, VIDIOC_DQBUF, &v4l2_buf);
        need_to_requeue_buffer = true;

        bytes_frame = (uint8_t*)camera->mmapped_buf[v4l2_buf.index];
        Nbytes_frame = v4l2_buf.bytesused;

        int     s  = v4l2_buf.timestamp.tv_sec;
        int     us = v4l2_buf.timestamp.tv_usec;
        if(timestamp_us != NULL)
            *timestamp_us = (uint64_t)s*1000000UL + (uint64_t)us;
    }

    // I now have the raw frame data in (bytes_frame, Nbytes_frame). I convert

    const uint8_t * const * scale_source;
    int* scale_stride;

    // may need to point the above pointers here
    int      scale_stride_value[4];
    uint8_t* scale_source_value[4];

    // This is only needed for some formats
    if(camera->av_codec_context)
    {
        ENSURE(0 <= avcodec_send_packet(camera->av_codec_context,
                                        camera->av_packet));

        // Not checking for
        //
        //   AVERROR(EAGAIN) || ret == AVERROR_EOF
        //
        // As the sample suggests. I'm waiting for exactly one frame, so that's
        // what I should get
        ENSURE(0 <= avcodec_receive_frame(camera->av_codec_context,
                                          camera->av_frame));

        scale_source = (const uint8_t*const*)camera->av_frame->data;
        scale_stride = camera->av_frame->linesize;

        // decoder_init() may not have created the sws context yet. I have the
        // output pixel format only after the first frame was read (i.e. now),
        // so I make the sws context now, if I need to
        if(camera->sws_context == NULL)
            ENSURE(NULL !=
                   (camera->sws_context =
                    sws_getContext(// source
                                   camera->format.fmt.pix.width,
                                   camera->format.fmt.pix.height,
                                   camera->av_codec_context->pix_fmt,

                                   // destination
                                   camera->format.fmt.pix.width,
                                   camera->format.fmt.pix.height,
                                   camera->av_pixelformat_output,

                                   // misc stuff
                                   SWS_POINT, NULL, NULL, NULL)));

        // should be done
        ENSURE(AVERROR_EOF == avcodec_receive_frame(camera->av_codec_context,
                                                    camera->av_frame));
    }
    else
    {

        ENSURE(0 < av_image_fill_arrays(scale_source_value, scale_stride_value,
                                        bytes_frame,
                                        camera->av_pixelformat_input,
                                        camera->format.fmt.pix.width,
                                        camera->format.fmt.pix.height,
                                        1) );

        scale_source = (const uint8_t*const*)scale_source_value;
        scale_stride = scale_stride_value;
    }

    int output_stride;
    switch(camera->pixelformat_output.u)
    {
    case V4L2_PIX_FMT_BGR24:
    case V4L2_PIX_FMT_RGB24:
        output_stride = 3*camera->format.fmt.pix.width;
        break;

    case V4L2_PIX_FMT_GREY:
        output_stride = camera->format.fmt.pix.width;
        break;

    case V4L2_PIX_FMT_Y16:
        output_stride = 2*camera->format.fmt.pix.width;
        break;

    default:
        MSG("Output pixel format %.4s is unsupported. dv4l_init() should have made sure we never get here", camera->pixelformat_output.s);
        return false;
    }

    sws_scale(camera->sws_context,
              // source
              scale_source, scale_stride, 0, camera->format.fmt.pix.height,
              // destination buffer, stride
              (uint8_t*const*)&image,
              &output_stride);

    cleanup();

    return true;
}
