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



// pixel formats in order of decreasing desireability. Sorta arbitrary. I
// favor colors, higher bitrates and lower compression
static const
uint32_t pixfmts_color[] =
    {
        /* RGB formats */
        V4L2_PIX_FMT_BGR32,        /* 32  BGR-8-8-8-8   */
        V4L2_PIX_FMT_RGB32,        /* 32  RGB-8-8-8-8   */
        V4L2_PIX_FMT_BGR24,        /* 24  BGR-8-8-8     */
        V4L2_PIX_FMT_RGB24,        /* 24  RGB-8-8-8     */
        V4L2_PIX_FMT_RGB444,       /* 16  xxxxrrrr ggggbbbb */
        V4L2_PIX_FMT_RGB555,       /* 16  RGB-5-5-5     */
        V4L2_PIX_FMT_RGB565,       /* 16  RGB-5-6-5     */
        V4L2_PIX_FMT_RGB555X,      /* 16  RGB-5-5-5 BE  */
        V4L2_PIX_FMT_RGB565X,      /* 16  RGB-5-6-5 BE  */
        V4L2_PIX_FMT_RGB332,       /*  8  RGB-3-3-2     */

        /* Palette formats */
        V4L2_PIX_FMT_PAL8,         /*  8  8-bit palette */

        /* Luminance+Chrominance formats */
        V4L2_PIX_FMT_YUV32,        /* 32  YUV-8-8-8-8   */
        V4L2_PIX_FMT_YUV444,       /* 16  xxxxyyyy uuuuvvvv */
        V4L2_PIX_FMT_YUV555,       /* 16  YUV-5-5-5     */
        V4L2_PIX_FMT_YUV565,       /* 16  YUV-5-6-5     */
        V4L2_PIX_FMT_YUYV,         /* 16  YUV 4:2:2     */
        V4L2_PIX_FMT_YYUV,         /* 16  YUV 4:2:2     */
        V4L2_PIX_FMT_YVYU,         /* 16 YVU 4:2:2 */
        V4L2_PIX_FMT_UYVY,         /* 16  YUV 4:2:2     */
        V4L2_PIX_FMT_VYUY,         /* 16  YUV 4:2:2     */
        V4L2_PIX_FMT_YUV422P,      /* 16  YVU422 planar */
        V4L2_PIX_FMT_YUV411P,      /* 16  YVU411 planar */
        V4L2_PIX_FMT_YVU420,       /* 12  YVU 4:2:0     */
        V4L2_PIX_FMT_Y41P,         /* 12  YUV 4:1:1     */
        V4L2_PIX_FMT_YUV420,       /* 12  YUV 4:2:0     */
        V4L2_PIX_FMT_YVU410,       /*  9  YVU 4:1:0     */
        V4L2_PIX_FMT_YUV410,       /*  9  YUV 4:1:0     */
        V4L2_PIX_FMT_HI240,        /*  8  8-bit color   */
        V4L2_PIX_FMT_HM12,         /*  8  YUV 4:2:0 16x16 macroblocks */

        /* two planes -- one Y, one Cr + Cb interleaved  */
        V4L2_PIX_FMT_NV12,         /* 12  Y/CbCr 4:2:0  */
        V4L2_PIX_FMT_NV21,         /* 12  Y/CrCb 4:2:0  */
        V4L2_PIX_FMT_NV16,         /* 16  Y/CbCr 4:2:2  */
        V4L2_PIX_FMT_NV61,         /* 16  Y/CrCb 4:2:2  */

        /* Bayer formats - see http://www.siliconimaging.com/RGB%20Bayer.htm */
        V4L2_PIX_FMT_SGRBG10,      /* 10bit raw bayer */
        /*
         * 10bit raw bayer, expanded to 16 bits
         * xxxxrrrrrrrrrrxxxxgggggggggg xxxxggggggggggxxxxbbbbbbbbbb...
         */
        V4L2_PIX_FMT_SBGGR16,      /* 16  BGBG.. GRGR.. */
        /* 10bit raw bayer DPCM compressed to 8 bits */
        V4L2_PIX_FMT_SGRBG10DPCM8,
        V4L2_PIX_FMT_SBGGR8,       /*  8  BGBG.. GRGR.. */
        V4L2_PIX_FMT_SGBRG8,       /*  8  GBGB.. RGRG.. */
        V4L2_PIX_FMT_SGRBG8,       /*  8  GRGR.. BGBG.. */

        /* compressed formats */
        V4L2_PIX_FMT_MJPEG,        /* Motion-JPEG   */
        V4L2_PIX_FMT_JPEG,         /* JFIF JPEG     */
        V4L2_PIX_FMT_DV,           /* 1394          */
        V4L2_PIX_FMT_MPEG,         /* MPEG-1/2/4    */

        /*  Vendor-specific formats   */
        V4L2_PIX_FMT_WNVA,         /* Winnov hw compress */
        V4L2_PIX_FMT_SN9C10X,      /* SN9C10x compression */
        V4L2_PIX_FMT_SN9C20X_I420, /* SN9C20x YUV 4:2:0 */
        V4L2_PIX_FMT_PWC1,         /* pwc older webcam */
        V4L2_PIX_FMT_PWC2,         /* pwc newer webcam */
        V4L2_PIX_FMT_ET61X251,     /* ET61X251 compression */
        V4L2_PIX_FMT_SPCA501,      /* YUYV per line */
        V4L2_PIX_FMT_SPCA505,      /* YYUV per line */
        V4L2_PIX_FMT_SPCA508,      /* YUVY per line */
        V4L2_PIX_FMT_SPCA561,      /* compressed GBRG bayer */
        V4L2_PIX_FMT_PAC207,       /* compressed BGGR bayer */
        V4L2_PIX_FMT_MR97310A,     /* compressed BGGR bayer */
        V4L2_PIX_FMT_SQ905C,       /* compressed RGGB bayer */
        V4L2_PIX_FMT_PJPG,         /* Pixart 73xx JPEG */
        V4L2_PIX_FMT_OV511,        /* ov511 JPEG */
        V4L2_PIX_FMT_OV518,        /* ov518 JPEG */
    };

static const
uint32_t pixfmts_gray[] =
    {
        /* Grey formats */
        V4L2_PIX_FMT_Y16,  /* 16  Greyscale     */
        V4L2_PIX_FMT_GREY, /*  8  Greyscale     */
    };

static bool is_pixelformat_color(uint32_t pixfmt)
{
    for(unsigned int i=0; i<sizeof(pixfmts_color) / sizeof(pixfmts_color[0]); i++)
        if(pixfmts_color[i] == pixfmt)
            return true;

    for(unsigned int i=0; i<sizeof(pixfmts_gray) / sizeof(pixfmts_gray[0]); i++)
        if(pixfmts_gray [i] == pixfmt)
            return true;

    MSG("Unknown pixel format %08x. Will return grayscale images just in case",
        pixfmt);
    return false;
}

static
bool pixfmt_select(// out
                   uint32_t* pixfmt,
                   // in
                   int fd,
                   bool want_color)
{
    int pixfmt_cost(uint32_t pixfmt, bool want_color)
    {
        // return the cost if there's a match. We add a cost penalty for color modes
        // when we wanted grayscale and vice-versa
        for(unsigned int i=0; i<sizeof(pixfmts_color) / sizeof(pixfmts_color[0]); i++)
            if(pixfmts_color[i] == pixfmt) return i + ( want_color ? 0 : 10000);

        for(unsigned int i=0; i<sizeof(pixfmts_gray) / sizeof(pixfmts_gray[0]); i++)
            if(pixfmts_gray [i] == pixfmt) return i + (!want_color ? 0 : 10000);

        return -1;
    }




    *pixfmt = 0;

    int pixfmt_best_cost = INT_MAX;

    for( struct v4l2_fmtdesc fmtdesc = {.index = 0,
                                        .type  = V4L2_BUF_TYPE_VIDEO_CAPTURE};
         ;
         fmtdesc.index++ )
    {
        if(ioctl_persistent( fd, VIDIOC_ENUM_FMT, &fmtdesc) < 0)
        {
            if(errno == EINVAL)
                // No more formats left. done. Return the best match I found so
                // far
                return true;

            // Something bad happened
            MSG("Error querying pixel format %d: %s",
                fmtdesc.index,
                strerrordesc_np(errno));
            return false;
        }

        int cost = pixfmt_cost(fmtdesc.pixelformat, want_color);
        if(0 <= cost &&
           cost < pixfmt_best_cost)
        {
            // Best one I've seen so far. Use it.
            pixfmt_best_cost = cost;
            *pixfmt          = fmtdesc.pixelformat;
        }
    }

    MSG("Getting here is a bug");
}


static
bool decoder_init(// out
                  dv4l_t* camera,
                  // in
                  const struct v4l2_pix_format* pixfmt,
                  bool want_color)
{
    // for ENSURE()
    void cleanup(void)
    {
    }


    // swscale can't interpret the pixel format directly. Can avcodec do it and
    // THEN feed swscale?
    if(pixfmt->pixelformat == V4L2_PIX_FMT_JPEG)
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



    switch(pixfmt->pixelformat)
    {
    /* RGB formats */
    /* 32  BGR-8-8-8-8 */
    case V4L2_PIX_FMT_BGR32:
        camera->av_pixel_format = AV_PIX_FMT_BGR32;
        break;
    /* 32  RGB-8-8-8-8 */
    case V4L2_PIX_FMT_RGB32:
        camera->av_pixel_format = AV_PIX_FMT_RGB32;
        break;
    /* 24  BGR-8-8-8 */
    case V4L2_PIX_FMT_BGR24:
        camera->av_pixel_format = AV_PIX_FMT_BGR24;
        break;
    /* 24  RGB-8-8-8 */
    case V4L2_PIX_FMT_RGB24:
        camera->av_pixel_format = AV_PIX_FMT_RGB24;
        break;
    /* 16  xxxxrrrr ggggbbbb */
    case V4L2_PIX_FMT_RGB444:
        return false;
    /* 16  RGB-5-5-5 */
    case V4L2_PIX_FMT_RGB555:
        camera->av_pixel_format = AV_PIX_FMT_RGB555;
        break;
    /* 16  RGB-5-6-5 */
    case V4L2_PIX_FMT_RGB565:
        camera->av_pixel_format = AV_PIX_FMT_RGB565;
        break;
    /* 16  RGB-5-5-5 BE */
    case V4L2_PIX_FMT_RGB555X:
        return false;
    /* 16  RGB-5-6-5 BE */
    case V4L2_PIX_FMT_RGB565X:
        return false;
    /*  8  RGB-3-3-2 */
    case V4L2_PIX_FMT_RGB332:
        return false;

    /* Palette formats */
    /*  8  8-bit palette */
    case V4L2_PIX_FMT_PAL8:
        camera->av_pixel_format = AV_PIX_FMT_PAL8;
        break;

    /* Luminance+Chrominance formats */
    /* 32  YUV-8-8-8-8 */
    case V4L2_PIX_FMT_YUV32:
        return false;
    /* 16  xxxxyyyy uuuuvvvv */
    case V4L2_PIX_FMT_YUV444:
        return false;
    /* 16  YUV-5-5-5 */
    case V4L2_PIX_FMT_YUV555:
        return false;
    /* 16  YUV-5-6-5 */
    case V4L2_PIX_FMT_YUV565:
        return false;
    /* 16  YUV 4:2:2 */
    case V4L2_PIX_FMT_YUYV:
        camera->av_pixel_format = AV_PIX_FMT_YUYV422;
        break;
    /* 16  YUV 4:2:2 */
    case V4L2_PIX_FMT_YYUV:
        return false;
    /* 16  YVU 4:2:2 */
    case V4L2_PIX_FMT_YVYU:
        return false;
    /* 16  YUV 4:2:2 */
    case V4L2_PIX_FMT_UYVY:
        camera->av_pixel_format = AV_PIX_FMT_UYVY422;
        break;
    /* 16  YUV 4:2:2 */
    case V4L2_PIX_FMT_VYUY:
        return false;
    /* 16  YVU422 planar */
    case V4L2_PIX_FMT_YUV422P:
        camera->av_pixel_format = AV_PIX_FMT_YUV422P;
        break;
    /* 16  YVU411 planar */
    case V4L2_PIX_FMT_YUV411P:
        camera->av_pixel_format = AV_PIX_FMT_YUV411P;
        break;
    /* 12  YVU 4:2:0 */
    case V4L2_PIX_FMT_YVU420:
        return false;
    /* 12  YUV 4:1:1 */
    case V4L2_PIX_FMT_Y41P:
        camera->av_pixel_format = AV_PIX_FMT_UYYVYY411;
        break;
    /* 12  YUV 4:2:0 */
    case V4L2_PIX_FMT_YUV420:
        camera->av_pixel_format = AV_PIX_FMT_YUV420P;
        break;
    /*  9  YVU 4:1:0 */
    case V4L2_PIX_FMT_YVU410:
        return false;
    /*  9  YUV 4:1:0 */
    case V4L2_PIX_FMT_YUV410:
        camera->av_pixel_format = AV_PIX_FMT_YUV410P;
        break;
    /*  8  8-bit color */
    case V4L2_PIX_FMT_HI240:
        return false;
    /*  8  YUV 4:2:0 16x16 macroblocks */
    case V4L2_PIX_FMT_HM12:
        return false;

    /* two planes -- one Y: one Cr + Cb interleaved */
    /* 12  Y/CbCr 4:2:0 */
    case V4L2_PIX_FMT_NV12:
        camera->av_pixel_format = AV_PIX_FMT_NV12;
        break;
    /* 12  Y/CrCb 4:2:0 */
    case V4L2_PIX_FMT_NV21:
        camera->av_pixel_format = AV_PIX_FMT_NV21;
        break;
    /* 16  Y/CbCr 4:2:2 */
    case V4L2_PIX_FMT_NV16:
        camera->av_pixel_format = AV_PIX_FMT_YUV422P;
        break;
    /* 16  Y/CrCb 4:2:2 */
    case V4L2_PIX_FMT_NV61:
        return false;

    /* Grey formats */
    /* 16  Greyscale */
    case V4L2_PIX_FMT_Y16:
        camera->av_pixel_format = AV_PIX_FMT_GRAY16LE;
        break;
    /*  8  Greyscale */
    case V4L2_PIX_FMT_GREY:
        camera->av_pixel_format = AV_PIX_FMT_GRAY8;
        break;

    default:
        MSG("Unknown pixel format %c%c%c%c",
            (uint8_t)(pixfmt->pixelformat>> 0),
            (uint8_t)(pixfmt->pixelformat>> 8),
            (uint8_t)(pixfmt->pixelformat>>16),
            (uint8_t)(pixfmt->pixelformat>>24));
        return false;
    }

    ENSURE(NULL !=
           (camera->sws_context =
            sws_getContext(// source
                           pixfmt->width, pixfmt->height, camera->av_pixel_format,

                           // destination
                           pixfmt->width, pixfmt->height,
                           want_color ? AV_PIX_FMT_RGB24 : AV_PIX_FMT_GRAY8,

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

               // We either ask fora specific pixel format from the camera, or
               // we pick the best one we've got
               dv4l_pixelformat_choice_t pixelformat_requested,

               const struct v4l2_control* controls,
               const int Ncontrols)
{
    // for ENSURE()
    void cleanup(void)
    {
        dv4l_deinit(camera);
    }


    *camera = (dv4l_t){};

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

    // If asked for the biggest possible images, I ask the driver for an
    // unreasonable upper bound of what I want. The driver will change the
    // passed-in parameters to whatever it is actually capable of
    camera->format =
        (struct v4l2_format)
        {.type    = V4L2_BUF_TYPE_VIDEO_CAPTURE,
         .fmt.pix = {.width       = width_requested  <= 0 ? 100000 : width_requested,
                     .height      = height_requested <= 0 ? 100000 : height_requested,
                     .field       = V4L2_FIELD_NONE} }; // no interlacing

    if(pixelformat_requested.choice == USE_REQUESTED_PIXELFORMAT)
    {
        camera->want_color = is_pixelformat_color(pixelformat_requested.pixelformat);
        camera->format.fmt.pix.pixelformat = pixelformat_requested.pixelformat;
    }
    else
    {
        camera->want_color = pixelformat_requested.choice == BEST_COLOR_PIXELFORMAT;
        ENSURE(pixfmt_select(&camera->format.fmt.pix.pixelformat,
                             camera->fd,
                             camera->want_color));
    }

    uint32_t pixelformat_did_set = camera->format.fmt.pix.pixelformat;
    ENSURE_IOCTL(camera->fd, VIDIOC_S_FMT, &camera->format);
    if(pixelformat_did_set != camera->format.fmt.pix.pixelformat)
    {
        MSG("Warning: asked for pixel format %c%c%c%c but V4L2 gave us %c%c%c%c instead. Continuing",
            (uint8_t)(pixelformat_did_set>> 0),
            (uint8_t)(pixelformat_did_set>> 8),
            (uint8_t)(pixelformat_did_set>>16),
            (uint8_t)(pixelformat_did_set>>24),
            (uint8_t)(camera->format.fmt.pix.pixelformat>> 0),
            (uint8_t)(camera->format.fmt.pix.pixelformat>> 8),
            (uint8_t)(camera->format.fmt.pix.pixelformat>>16),
            (uint8_t)(camera->format.fmt.pix.pixelformat>>24));
    }

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
                        &camera->format.fmt.pix,
                        camera->want_color));

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
                                   camera->want_color ? AV_PIX_FMT_RGB24 : AV_PIX_FMT_GRAY8,

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
                                        camera->av_pixel_format,
                                        camera->format.fmt.pix.width,
                                        camera->format.fmt.pix.height,
                                        1) );

        scale_source = (const uint8_t*const*)scale_source_value;
        scale_stride = scale_stride_value;
    }


    // This is ALWAYS needed
    sws_scale(camera->sws_context,
              // source
              scale_source, scale_stride, 0, camera->format.fmt.pix.height,
              // destination buffer, stride
              (uint8_t*const*)&image,
              camera->want_color ?
              &(int){3*camera->format.fmt.pix.width} :
              &(int){  camera->format.fmt.pix.width});

    cleanup();

    return true;
}
