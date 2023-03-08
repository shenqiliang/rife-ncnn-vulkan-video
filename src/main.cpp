// rife implemented with ncnn library

#include <stdio.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <clocale>

#if _WIN32
// image decoder and encoder with wic
#include "wic_image.h"
#else // _WIN32
// image decoder and encoder with stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_STDIO
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // _WIN32
#include <codecvt>
#include <memory>
#include <thread>

#include "webp_image.h"
#include "FFmpegVideoDecoder.h"
#include "FFmpegVideoEncoder.h"


std::string wstr2str(std::wstring string_to_convert)
{

    //setup converter
    using convert_type = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_type, wchar_t> converter;

    //use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
    std::string converted_str = converter.to_bytes( string_to_convert );
    return converted_str;
}


#if _WIN32
#include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t* const argv[], const wchar_t* optstring)
{
    if (optind >= argc || argv[optind][0] != L'-')
        return -1;

    wchar_t opt = argv[optind][1];
    const wchar_t* p = wcschr(optstring, opt);
    if (p == NULL)
        return L'?';

    optarg = NULL;

    if (p[1] == L':')
    {
        optind++;
        if (optind >= argc)
            return L'?';

        optarg = argv[optind];
    }

    optind++;

    return opt;
}

static std::vector<int> parse_optarg_int_array(const wchar_t* optarg)
{
    std::vector<int> array;
    array.push_back(_wtoi(optarg));

    const wchar_t* p = wcschr(optarg, L',');
    while (p)
    {
        p++;
        array.push_back(_wtoi(p));
        p = wcschr(p, L',');
    }

    return array;
}
#else // _WIN32
#include <unistd.h> // getopt()

static std::vector<int> parse_optarg_int_array(const char* optarg)
{
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char* p = strchr(optarg, ',');
    while (p)
    {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}
#endif // _WIN32

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"
#include "benchmark.h"

#include "rife.h"

#include "filesystem_utils.h"

static void print_usage()
{
    fprintf(stderr, "Usage: rife-ncnn-vulkan -0 infile -1 infile1 -o outfile [options]...\n");
    fprintf(stderr, "       rife-ncnn-vulkan -i indir -o outdir [options]...\n\n");
    fprintf(stderr, "  -h                   show this help\n");
    fprintf(stderr, "  -v                   verbose output\n");
    fprintf(stderr, "  -0 input0-path       input image0 path (jpg/png/webp)\n");
    fprintf(stderr, "  -1 input1-path       input image1 path (jpg/png/webp)\n");
    fprintf(stderr, "  -i input-path        input image directory (jpg/png/webp)\n");
    fprintf(stderr, "  -o output-path       output image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -n num-frame         target frame count (default=N*2)\n");
    fprintf(stderr, "  -s time-step         time step (0~1, default=0.5)\n");
    fprintf(stderr, "  -m model-path        rife model path (default=rife-v2.3)\n");
    fprintf(stderr, "  -g gpu-id            gpu device to use (-1=cpu, default=auto) can be 0,1,2 for multi-gpu\n");
    fprintf(stderr, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stdout, "  -x                   enable spatial tta mode\n");
    fprintf(stdout, "  -z                   enable temporal tta mode\n");
    fprintf(stdout, "  -u                   enable UHD mode\n");
    fprintf(stderr, "  -f pattern-format    output image filename pattern format (%%08d.jpg/png/webp, default=ext/%%08d.png)\n");
}

static int decode_image(const path_t& imagepath, ncnn::Mat& image, int* webp)
{
    *webp = 0;

    unsigned char* pixeldata = 0;
    int w;
    int h;
    int c;

#if _WIN32
    FILE* fp = _wfopen(imagepath.c_str(), L"rb");
#else
    FILE* fp = fopen(imagepath.c_str(), "rb");
#endif
    if (fp)
    {
        // read whole file
        unsigned char* filedata = 0;
        int length = 0;
        {
            fseek(fp, 0, SEEK_END);
            length = ftell(fp);
            rewind(fp);
            filedata = (unsigned char*)malloc(length);
            if (filedata)
            {
                fread(filedata, 1, length, fp);
            }
            fclose(fp);
        }

        if (filedata)
        {
            pixeldata = webp_load(filedata, length, &w, &h, &c);
            if (pixeldata)
            {
                *webp = 1;
            }
            else
            {
                // not webp, try jpg png etc.
#if _WIN32
                pixeldata = wic_decode_image(imagepath.c_str(), &w, &h, &c);
#else // _WIN32
                pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 3);
                c = 3;
#endif // _WIN32
            }

            free(filedata);
        }
    }

    if (!pixeldata)
    {
#if _WIN32
        fwprintf(stderr, L"decode image %ls failed\n", imagepath.c_str());
#else // _WIN32
        fprintf(stderr, "decode image %s failed\n", imagepath.c_str());
#endif // _WIN32

        return -1;
    }

    image = ncnn::Mat(w, h, (void*)pixeldata, (size_t)3, 3);

    return 0;
}

static int encode_image(const path_t& imagepath, const ncnn::Mat& image)
{
    int success = 0;

    path_t ext = get_file_extension(imagepath);

    if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
    {
        success = webp_save(imagepath.c_str(), image.w, image.h, image.elempack, (const unsigned char*)image.data);
    }
    else if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
    {
#if _WIN32
        success = wic_encode_image(imagepath.c_str(), image.w, image.h, image.elempack, image.data);
#else
        success = stbi_write_png(imagepath.c_str(), image.w, image.h, image.elempack, image.data, 0);
#endif
    }
    else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
    {
#if _WIN32
        success = wic_encode_jpeg_image(imagepath.c_str(), image.w, image.h, image.elempack, image.data);
#else
        success = stbi_write_jpg(imagepath.c_str(), image.w, image.h, image.elempack, image.data, 100);
#endif
    }

    if (!success)
    {
#if _WIN32
        fwprintf(stderr, L"encode image %ls failed\n", imagepath.c_str());
#else
        fprintf(stderr, "encode image %s failed\n", imagepath.c_str());
#endif
    }

    return success ? 0 : -1;
}


class Task
{
public:
    int id;

    int step = 1;

    double time0;
    double time1;

    bool processed = false;

    ncnn::Mat in0image;
    ncnn::Mat in1image;
    ncnn::Mat out0image;
    ncnn::Mat out1image;
    ncnn::Mat out2image;

};

template <typename T>
class TaskQueue
{
public:
    const int cap;
    
    TaskQueue(const int cap): cap(cap)
    {
    }
    
    void put(const T& v)
    {
        lock.lock();

        while (tasks.size() >= cap) // FIXME hardcode queue length
            {
            condition_not_full.waitTime(lock, 3000);
            }

        
        tasks.push(v);

        lock.unlock();

        condition_has_data.broadcast();
    }

    void pop(T& v)
    {
        lock.lock();

        while (tasks.size() == 0)
        {
            condition_has_data.waitTime(lock, 3000);
        }

        v = tasks.front();
        tasks.pop();

        lock.unlock();

        condition_not_full.broadcast();
    }

    int pop_no_block(T& v)
    {
        int ret = 0;
        lock.lock();

        if (tasks.size() == 0)
        {
            ret = -1;
        }
        else
        {
            v = tasks.front();
            tasks.pop();
            ret = 1;
        }


        lock.unlock();
        
        return ret;
    }


private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition_has_data;
    ncnn::ConditionVariable condition_not_full;
    std::queue<T> tasks;
};

TaskQueue<std::shared_ptr<Task>> toproc(8);
TaskQueue<std::shared_ptr<Task>> tosave(40);
TaskQueue<FMediaFrame> audioqueue(INT32_MAX);
FFmpegVideoDecoder *VideoDecoder;


class LoadThreadParams
{
public:
    int jobs_load;
    int step;
};

void* load(void* args)
{
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;

    FMediaFrame FirstFrame;
    VideoDecoder->DecodeMedia(FirstFrame);
    FMediaFrame NewFrame;
    int index = 0;
    while (VideoDecoder->DecodeMedia(NewFrame) == 0)
    {
        if (NewFrame.Type == FFrameType::Video)
        {
            unsigned char* pixeldata0 = FirstFrame.Buffer;
            unsigned char* pixeldata1 = NewFrame.Buffer;
            if (pixeldata0 != nullptr && pixeldata1 != nullptr)
            {
                auto v = std::make_shared<Task>();
                v->id = index++;
                v->time0 = FirstFrame.Time;
                v->time1 = NewFrame.Time;
                v->step = ltp->step;

                fprintf(stderr, "decode video time: %f\n", v->time1);

                v->in0image = ncnn::Mat(FirstFrame.Width, FirstFrame.Height, pixeldata0, (size_t)3, 3);
                v->in1image = ncnn::Mat(NewFrame.Width, NewFrame.Height, pixeldata1, (size_t)3, 3);
                v->out0image = ncnn::Mat(NewFrame.Width, NewFrame.Height, (size_t)3, 3);
                v->out1image = ncnn::Mat(NewFrame.Width, NewFrame.Height, (size_t)3, 3);
                v->out2image = ncnn::Mat(NewFrame.Width, NewFrame.Height, (size_t)3, 3);
                v->processed = false;
                
                toproc.put(v);
                tosave.put(v);

            }
            FirstFrame = NewFrame;
        }
        else if (NewFrame.Type == FFrameType::Audio)
        {
            audioqueue.put(NewFrame);
        }

    }

    return 0;
}

class ProcThreadParams
{
public:
    const RIFE* rife;
};

void* proc(void* args)
{
    const ProcThreadParams* ptp = (const ProcThreadParams*)args;
    const RIFE* rife = ptp->rife;

    for (;;)
    {
        std::shared_ptr<Task> v;

        toproc.pop(v);

        if (v->id == -233)
            break;

        const int step = v->step;
        if (step >= 1)
        {
            rife->process(v->in0image, v->in1image, 1.0f/(1.0f+step), v->out0image);
        }
        if (step >= 2)
        {
            rife->process(v->in0image, v->in1image, 1.0f/(1.0f+step) * 2, v->out1image);
        }
        if (step >= 3)
        {
            rife->process(v->in0image, v->in1image, 1.0f/(1.0f+step) * 3, v->out2image);
        }

        v->processed = true;
    }

    return 0;
}

class SaveThreadParams
{
public:
    int verbose;
    int width;
    int height;
    std::string save_path;
};

void* save(void* args)
{
    const SaveThreadParams* stp = (const SaveThreadParams*)args;
    FFmpegVideoEncoder encoder(stp->width, stp->height, stp->save_path);
    bool first_encoded = false;
    
    for (;;)
    {
        std::shared_ptr<Task> v;

        tosave.pop(v);

        if (v->id == -233)
            break;

        while (!v->processed)
        {
            encoder.Flush();
            if (!v->processed)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        
        FMediaFrame audio_frame;
        while (audioqueue.pop_no_block(audio_frame) > 0)
        {
            encoder.AddAudioBuffer(audio_frame.Buffer, audio_frame.Size);
            ResetMediaFrame(audio_frame);
        }

        if (!first_encoded)
        {
            encoder.AddVideoBuffer((const char*)v->in0image.data, v->in0image.w, v->in0image.h, v->time0);
            first_encoded = true;
        }

        const int step = v->step;
        const float start_time = v->time0;
        const float step_time = (v->time1 - v->time0)/(step+1);

        if (step >= 1)
        {
            encoder.AddVideoBuffer((const char*)v->out0image.data, v->out0image.w, v->out0image.h, start_time+step_time);
        }
        if (step >= 2)
        {
            encoder.AddVideoBuffer((const char*)v->out0image.data, v->out0image.w, v->out0image.h, start_time+step_time*2);
        }
        if (step >= 3)
        {
            encoder.AddVideoBuffer((const char*)v->out0image.data, v->out0image.w, v->out0image.h, start_time+step_time*3);
        }

        
        encoder.AddVideoBuffer((const char*)v->in1image.data, v->in1image.w, v->in1image.h, v->time1);
        v->out0image.release();
        void *buffer = v->in0image.data;
        v->in0image.release();
        if (buffer != nullptr)
        {
            //free(buffer);
        }
    }
    encoder.EndEncoder();

    return 0;
}


#if _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    auto begin_time = std::chrono::system_clock::now();
    path_t inputpath;
    path_t outputpath;
    int numframe = 0;
    float timestep = 0.5f;
    path_t model = PATHSTR("rife-v2.3");
    std::vector<int> gpuid;
    int jobs_load = 1;
    std::vector<int> jobs_proc;
    int jobs_save = 1;
    int verbose = 0;
    int tta_mode = 0;
    int tta_temporal_mode = 0;
    int uhd_mode = 0;
    path_t pattern_format = PATHSTR("%08d.png");

#if _WIN32
    setlocale(LC_ALL, "");
    wchar_t opt;
    while ((opt = getopt(argc, argv, L"i:o:n:s:m:g:j:f:vxzuh")) != (wchar_t)-1)
    {
        switch (opt)
        {
        case L'i':
            inputpath = optarg;
            break;
        case L'o':
            outputpath = optarg;
            break;
        case L'n':
            numframe = _wtoi(optarg);
            break;
        case L's':
            timestep = _wtof(optarg);
            break;
        case L'm':
            model = optarg;
            break;
        case L'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case L'j':
            swscanf(optarg, L"%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(wcschr(optarg, L':') + 1);
            break;
        case L'f':
            pattern_format = optarg;
            break;
        case L'v':
            verbose = 1;
            break;
        case L'x':
            tta_mode = 1;
            break;
        case L'z':
            tta_temporal_mode = 1;
            break;
        case L'u':
            uhd_mode = 1;
            break;
        case L'h':
        default:
            print_usage();
            return -1;
        }
    }
#else // _WIN32
    int opt;
    while ((opt = getopt(argc, argv, "i:o:n:s:m:g:j:f:vxzuh")) != -1)
    {
        switch (opt)
        {
        case 'i':
            inputpath = optarg;
            break;
        case 'o':
            outputpath = optarg;
            break;
        case 'n':
            numframe = atoi(optarg);
            break;
        case 's':
            timestep = atof(optarg);
            break;
        case 'm':
            model = optarg;
            break;
        case 'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case 'j':
            sscanf(optarg, "%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
            break;
        case 'f':
            pattern_format = optarg;
            break;
        case 'v':
            verbose = 1;
            break;
        case 'x':
            tta_mode = 1;
            break;
        case 'z':
            tta_temporal_mode = 1;
            break;
        case 'u':
            uhd_mode = 1;
            break;
        case 'h':
        default:
            print_usage();
            return -1;
        }
    }
#endif // _WIN32

    if (inputpath.empty() || outputpath.empty())
    {
        print_usage();
        return -1;
    }

    if (inputpath.empty() && (timestep <= 0.f || timestep >= 1.f))
    {
        fprintf(stderr, "invalid timestep argument, must be 0~1\n");
        return -1;
    }

    if (!inputpath.empty() && numframe < 0)
    {
        fprintf(stderr, "invalid numframe argument, must not be negative\n");
        return -1;
    }

    if (jobs_load < 1 || jobs_save < 1)
    {
        fprintf(stderr, "invalid thread count argument\n");
        return -1;
    }

    if (jobs_proc.size() != (gpuid.empty() ? 1 : gpuid.size()) && !jobs_proc.empty())
    {
        fprintf(stderr, "invalid jobs_proc thread count argument\n");
        return -1;
    }

    for (int i=0; i<(int)jobs_proc.size(); i++)
    {
        if (jobs_proc[i] < 1)
        {
            fprintf(stderr, "invalid jobs_proc thread count argument\n");
            return -1;
        }
    }
    
    bool rife_v2 = false;
    bool rife_v4 = false;
    if (model.find(PATHSTR("rife-v2")) != path_t::npos)
    {
        // fine
        rife_v2 = true;
    }
    else if (model.find(PATHSTR("rife-v3")) != path_t::npos)
    {
        // fine
        rife_v2 = true;
    }
    else if (model.find(PATHSTR("rife-v4")) != path_t::npos)
    {
        // fine
        rife_v4 = true;
    }
    else if (model.find(PATHSTR("rife")) != path_t::npos)
    {
        // fine
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

    if (!rife_v4 && (numframe != 0 || timestep != 0.5))
    {
        fprintf(stderr, "only rife-v4 model support custom numframe and timestep\n");
        return -1;
    }
    

    path_t modeldir = sanitize_dirpath(model);

#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    ncnn::create_gpu_instance();

    if (gpuid.empty())
    {
        gpuid.push_back(ncnn::get_default_gpu_index());
    }

    const int use_gpu_count = (int)gpuid.size();

    if (jobs_proc.empty())
    {
        jobs_proc.resize(use_gpu_count, 2);
    }

    int cpu_count = std::max(1, ncnn::get_cpu_count());
    jobs_load = std::min(jobs_load, cpu_count);

    int gpu_count = ncnn::get_gpu_count();
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] < -1 || gpuid[i] >= gpu_count)
        {
            fprintf(stderr, "invalid gpu device\n");

            ncnn::destroy_gpu_instance();
            return -1;
        }
    }

    int total_jobs_proc = 0;
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] == -1)
        {
            jobs_proc[i] = std::min(jobs_proc[i], cpu_count);
            total_jobs_proc += 1;
        }
        else
        {
            total_jobs_proc += jobs_proc[i];
        }
    }

    {
        std::vector<RIFE*> rife(use_gpu_count);

        for (int i=0; i<use_gpu_count; i++)
        {
            int num_threads = gpuid[i] == -1 ? jobs_proc[i] : 1;

            rife[i] = new RIFE(gpuid[i], tta_mode, tta_temporal_mode, uhd_mode, num_threads, rife_v2, rife_v4);

            rife[i]->load(modeldir);
        }

        // main routine
        {
            // load image
            LoadThreadParams ltp;
            ltp.jobs_load = jobs_load;
            ltp.step = std::min(std::max((1.0/timestep)-1.0, 1.0), 3.0);
#if _WIN32
            VideoDecoder = new FFmpegVideoDecoder(wstr2str(inputpath));
#else
            VideoDecoder = new FFmpegVideoDecoder(inputpath);
#endif

            ncnn::Thread load_thread(load, (void*)&ltp);

            // rife proc
            std::vector<ProcThreadParams> ptp(use_gpu_count);
            for (int i=0; i<use_gpu_count; i++)
            {
                ptp[i].rife = rife[i];
            }

            std::vector<ncnn::Thread*> proc_threads(total_jobs_proc);
            {
                int total_jobs_proc_id = 0;
                for (int i=0; i<use_gpu_count; i++)
                {
                    if (gpuid[i] == -1)
                    {
                        proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                    }
                    else
                    {
                        for (int j=0; j<jobs_proc[i]; j++)
                        {
                            proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                        }
                    }
                }
            }

            // save image
            SaveThreadParams stp;
            stp.verbose = verbose;
            stp.width = VideoDecoder->GetWidth();
            stp.height = VideoDecoder->GetHeight();
#if _WIN32
            stp.save_path = wstr2str(outputpath);
#else
            stp.save_path = outputpath;
#endif

            jobs_save = 1;
            std::vector<ncnn::Thread*> save_threads(jobs_save);
            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i] = new ncnn::Thread(save, (void*)&stp);
            }

            // end
            load_thread.join();

            auto end = std::make_shared<Task>();
            end->id = -233;

            for (int i=0; i<total_jobs_proc; i++)
            {
                toproc.put(end);
            }

            for (int i=0; i<total_jobs_proc; i++)
            {
                proc_threads[i]->join();
                delete proc_threads[i];
            }

            for (int i=0; i<jobs_save; i++)
            {
                tosave.put(end);
            }

            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i]->join();
                delete save_threads[i];
            }
        }

        for (int i=0; i<use_gpu_count; i++)
        {
            delete rife[i];
        }
        rife.clear();
    }

    ncnn::destroy_gpu_instance();
    auto end_time = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time);

    fprintf(stderr, "Convert Total Cost: %lld\n", diff.count());

    return 0;
}
