#include <assert.h>
#include <chrono>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"
#include <cnpy.h>

#include <opencv2/opencv.hpp>
// 重要！
#include <opencv2/dnn.hpp>
// dpcpp alexnet.cpp -ldnnl `pkg-config opencv4 --cflags` `pkg-config opencv4 --libs` -lcnpy -o test
using namespace dnnl;
using namespace cv;

constexpr int IH = 224;
constexpr int IW = 224;
constexpr int IC = 3;
constexpr int IN = 1;

void read_img(std::string ImgName, std::vector<float> &data_buffer)
{
        // 读取图片，其中1表示以彩色图像的方式读取
        Mat image = imread(ImgName, 1);
        // 预处理（尺寸变换、通道变换、归一化）
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        cv::resize(image, image, cv::Size(224, 224));
        image.convertTo(image, CV_32FC3, 1.0 / 255.0);
        // cv::Scalar mean(0.5, 0.5, 0.5);
        // // 不一样的地方
        // cv::Scalar std(0.25, 0.5, 0.5);

        cv::Scalar mean(0.485, 0.456, 0.406);
        cv::Scalar std(0.229, 0.224, 0.225);

        cv::subtract(image, mean, image);
        cv::divide(image, std, image);

        // blobFromImage操作顺序：swapRB交换通道 -> scalefactor比例缩放 -> mean求减 -> size进行resize;
        // mean操作时，ddepth不能选取CV_8U;
        // crop=True时，先等比缩放，直到宽高之一率先达到对应的size尺寸，另一个大于或等于对应的size尺寸，然后从中心裁剪;
        // 返回4-D Mat维度顺序：NCHW
        // cv::Mat blob = cv::dnn::blobFromImage(image, 1., cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);
        cv::Mat blob = cv::dnn::blobFromImage(image);
        cv::Mat flatBlob = blob.reshape(1, 1);
        data_buffer.assign((float *)flatBlob.data, (float *)flatBlob.data + flatBlob.total() * flatBlob.channels());
}

std::vector<float> load_net_data(std::string key)
{
        cnpy::NpyArray arr = cnpy::npy_load("netarg/" + key + ".npy");
        float *res = arr.data<float>();
        std::vector<float> data(res, res + arr.num_vals);
        return data;
}
void simple_net(engine::kind engine_kind, int times = 100, std::string img_name = "1.jpg")
{
        using tag = memory::format_tag;
        using dt = memory::data_type;

        //[Initialize engine and stream]
        engine eng(engine_kind, 0);
        stream s(eng);
        //[Initialize engine and stream]

        //[Create network]
        std::vector<primitive> net;
        std::vector<std::unordered_map<int, memory>> net_args;
        //[Create network]

        const memory::dim batch = IN;

        // conv1
        // shape
        memory::dims conv1_src_tz = {batch, 3, IW, IH};
        memory::dims conv1_weights_tz = {64, 3, 11, 11};
        memory::dims conv1_bias_tz = {64};
        memory::dims conv1_dst_tz = {batch, 64, 55, 55};
        memory::dims conv1_strides = {4, 4};
        memory::dims conv1_padding = {2, 2};

        //[Allocate buffers]
        std::vector<float> user_src(batch * 3 * IW * IH);
        std::vector<float> user_dst(batch * 3);
        std::vector<float> conv1_weights = load_net_data("conv1-weight");
        std::vector<float> conv1_bias = load_net_data("conv1-bias");
        //[Allocate buffers]

        //[Create user memory]
        auto user_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
        write_to_dnnl_memory(user_src.data(), user_src_memory);
        auto user_weights_memory = memory({{conv1_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv1_weights.data(), user_weights_memory);
        auto conv1_user_bias_memory = memory({{conv1_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv1_bias.data(), conv1_user_bias_memory);
        //[Create user memory]

        //[Create convolution memory descriptors]
        auto conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
        auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
        auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
        auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);
        //[Create convolution memory descriptors]

        // post-ops
        // relu1
        post_ops conv_ops;
        conv_ops.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
        primitive_attr conv_attr;
        conv_attr.set_post_ops(conv_ops);

        //[Create convolution primitive descriptor]
        auto conv1_prim_desc = convolution_forward::primitive_desc(eng,
                                                                   prop_kind::forward_inference, algorithm::convolution_direct,
                                                                   conv1_src_md, conv1_weights_md, conv1_bias_md, conv1_dst_md,
                                                                   conv1_strides, conv1_padding, conv1_padding, conv_attr);
        //[Create convolution primitive descriptor]

        //[Reorder data and weights]
        auto conv1_src_memory = user_src_memory;
        if (conv1_prim_desc.src_desc() != user_src_memory.get_desc())
        {
                conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
                net.push_back(reorder(user_src_memory, conv1_src_memory));
                net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                                    {DNNL_ARG_TO, conv1_src_memory}});
        }

        auto conv1_weights_memory = user_weights_memory;
        if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc())
        {
                conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
                reorder(user_weights_memory, conv1_weights_memory)
                    .execute(s, user_weights_memory, conv1_weights_memory);
        }
        //[Reorder data and weights]

        //[Create memory for output]
        auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);
        //[Create memory for output]

        //[Create convolution primitive]
        net.push_back(convolution_forward(conv1_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv1_src_memory},
                            {DNNL_ARG_WEIGHTS, conv1_weights_memory},
                            {DNNL_ARG_BIAS, conv1_user_bias_memory},
                            {DNNL_ARG_DST, conv1_dst_memory}});
        //[Create convolution primitive]

        // lrn1
        const memory::dim local1_size = 5;
        const float alpha1 = 0.0001f;
        const float beta1 = 0.75f;
        const float k1 = 1.0f;

        // create lrn primitive and add it to net
        auto lrn1_prim_desc = lrn_forward::primitive_desc(eng,
                                                          prop_kind::forward_inference, algorithm::lrn_across_channels,
                                                          conv1_dst_memory.get_desc(), conv1_dst_memory.get_desc(),
                                                          local1_size, alpha1, beta1, k1);
        auto lrn1_dst_memory = memory(lrn1_prim_desc.dst_desc(), eng);

        net.push_back(lrn_forward(lrn1_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv1_dst_memory},
                            {DNNL_ARG_DST, lrn1_dst_memory}});

        // pool1
        memory::dims pool1_dst_tz = {batch, 64, 27, 27};
        memory::dims pool1_kernel = {3, 3};
        memory::dims pool1_strides = {2, 2};
        // new api
        memory::dims pool_dilation = {0, 0};
        memory::dims pool_padding = {0, 0};

        auto pool1_dst_md = memory::desc({pool1_dst_tz}, dt::f32, tag::any);

        //[Create pooling primitive]
        auto pool1_pd = pooling_forward::primitive_desc(eng,
                                                        prop_kind::forward_inference, algorithm::pooling_max,
                                                        lrn1_dst_memory.get_desc(), pool1_dst_md, pool1_strides,
                                                        pool1_kernel, pool_dilation, pool_padding, pool_padding);

        auto pool1_dst_memory = memory(pool1_pd.dst_desc(), eng);

        net.push_back(pooling_forward(pool1_pd));
        net_args.push_back({{DNNL_ARG_SRC, lrn1_dst_memory},
                            {DNNL_ARG_DST, pool1_dst_memory}});
        //[Create pooling primitive]

        // conv2
        memory::dims conv2_src_tz = {batch, 64, 27, 27};
        memory::dims conv2_weights_tz = {192, 64, 5, 5};
        memory::dims conv2_bias_tz = {192};
        memory::dims conv2_dst_tz = {batch, 192, 27, 27};
        memory::dims conv2_strides = {1, 1};
        memory::dims conv2_padding = {2, 2};

        std::vector<float> conv2_weights = load_net_data("conv2-weight");
        std::vector<float> conv2_bias = load_net_data("conv2-bias");

        // create memory for user data
        auto conv2_user_weights_memory = memory({{conv2_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv2_weights.data(), conv2_user_weights_memory);
        auto conv2_user_bias_memory = memory({{conv2_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv2_bias.data(), conv2_user_bias_memory);

        // create memory descriptors for convolution data w/ no specified format
        auto conv2_src_md = memory::desc({conv2_src_tz}, dt::f32, tag::any);
        auto conv2_bias_md = memory::desc({conv2_bias_tz}, dt::f32, tag::any);
        auto conv2_weights_md = memory::desc({conv2_weights_tz}, dt::f32, tag::any);
        auto conv2_dst_md = memory::desc({conv2_dst_tz}, dt::f32, tag::any);

        // relu2
        post_ops conv_ops2;
        conv_ops2.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
        primitive_attr conv_attr2;
        conv_attr2.set_post_ops(conv_ops2);

        // create a convolution
        auto conv2_prim_desc = convolution_forward::primitive_desc(eng,
                                                                   prop_kind::forward_inference, algorithm::convolution_direct,
                                                                   conv2_src_md, conv2_weights_md, conv2_bias_md, conv2_dst_md,
                                                                   conv2_strides, conv2_padding, conv2_padding, conv_attr2);

        auto conv2_src_memory = pool1_dst_memory;
        if (conv2_prim_desc.src_desc() != conv2_src_memory.get_desc())
        {
                conv2_src_memory = memory(conv2_prim_desc.src_desc(), eng);
                net.push_back(reorder(pool1_dst_memory, conv2_src_memory));
                net_args.push_back({{DNNL_ARG_FROM, pool1_dst_memory},
                                    {DNNL_ARG_TO, conv2_src_memory}});
        }

        auto conv2_weights_memory = conv2_user_weights_memory;
        if (conv2_prim_desc.weights_desc() != conv2_user_weights_memory.get_desc())
        {
                conv2_weights_memory = memory(conv2_prim_desc.weights_desc(), eng);
                reorder(conv2_user_weights_memory, conv2_weights_memory)
                    .execute(s, conv2_user_weights_memory, conv2_weights_memory);
        }

        auto conv2_dst_memory = memory(conv2_prim_desc.dst_desc(), eng);

        // create convolution primitive and add it to net
        net.push_back(convolution_forward(conv2_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv2_src_memory},
                            {DNNL_ARG_WEIGHTS, conv2_weights_memory},
                            {DNNL_ARG_BIAS, conv2_user_bias_memory},
                            {DNNL_ARG_DST, conv2_dst_memory}});

        // lrn2
        const memory::dim local2_size = 5;
        const float alpha2 = 0.0001f;
        const float beta2 = 0.75f;
        const float k2 = 1.0f;

        // create lrn primitive and add it to net
        auto lrn2_prim_desc = lrn_forward::primitive_desc(eng, prop_kind::forward_inference,
                                                          algorithm::lrn_across_channels, conv2_prim_desc.dst_desc(),
                                                          conv2_prim_desc.dst_desc(), local2_size, alpha2, beta2, k2);
        auto lrn2_dst_memory = memory(lrn2_prim_desc.dst_desc(), eng);

        net.push_back(lrn_forward(lrn2_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv2_dst_memory},
                            {DNNL_ARG_DST, lrn2_dst_memory}});

        // pool2
        memory::dims pool2_dst_tz = {batch, 192, 13, 13};
        memory::dims pool2_kernel = {3, 3};
        memory::dims pool2_strides = {2, 2};
        memory::dims pool2_dilation = {0, 0};
        memory::dims pool2_padding = {0, 0};

        auto pool2_dst_md = memory::desc({pool2_dst_tz}, dt::f32, tag::any);

        // create a pooling
        auto pool2_pd = pooling_forward::primitive_desc(eng,
                                                        prop_kind::forward_inference, algorithm::pooling_max,
                                                        lrn2_dst_memory.get_desc(), pool2_dst_md, pool2_strides,
                                                        pool2_kernel, pool2_dilation, pool2_padding, pool2_padding);
        auto pool2_dst_memory = memory(pool2_pd.dst_desc(), eng);

        // create pooling primitive an add it to net
        net.push_back(pooling_forward(pool2_pd));
        net_args.push_back({{DNNL_ARG_SRC, lrn2_dst_memory},
                            {DNNL_ARG_DST, pool2_dst_memory}});

        // conv3
        memory::dims conv3_src_tz = {batch, 192, 13, 13};
        memory::dims conv3_weights_tz = {384, 192, 3, 3};
        memory::dims conv3_bias_tz = {384};
        memory::dims conv3_dst_tz = {batch, 384, 13, 13};
        memory::dims conv3_strides = {1, 1};
        memory::dims conv3_padding = {1, 1};

        std::vector<float> conv3_weights = load_net_data("conv3-weight");
        std::vector<float> conv3_bias = load_net_data("conv3-bias");

        // create memory for user data
        auto conv3_user_weights_memory = memory({{conv3_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv3_weights.data(), conv3_user_weights_memory);
        auto conv3_user_bias_memory = memory({{conv3_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv3_bias.data(), conv3_user_bias_memory);

        // create memory descriptors for convolution data w/ no specified format
        auto conv3_src_md = memory::desc({conv3_src_tz}, dt::f32, tag::any);
        auto conv3_bias_md = memory::desc({conv3_bias_tz}, dt::f32, tag::any);
        auto conv3_weights_md = memory::desc({conv3_weights_tz}, dt::f32, tag::any);
        auto conv3_dst_md = memory::desc({conv3_dst_tz}, dt::f32, tag::any);

        post_ops conv_ops3;
        conv_ops3.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
        primitive_attr conv_attr3;
        conv_attr3.set_post_ops(conv_ops3);

        // create a convolution
        auto conv3_prim_desc = convolution_forward::primitive_desc(eng,
                                                                   prop_kind::forward_inference, algorithm::convolution_direct,
                                                                   conv3_src_md, conv3_weights_md, conv3_bias_md, conv3_dst_md,
                                                                   conv3_strides, conv3_padding, conv3_padding, conv_attr3);

        auto conv3_src_memory = pool2_dst_memory;
        if (conv3_prim_desc.src_desc() != conv3_src_memory.get_desc())
        {
                conv3_src_memory = memory(conv3_prim_desc.src_desc(), eng);
                net.push_back(reorder(pool2_dst_memory, conv3_src_memory));
                net_args.push_back({{DNNL_ARG_FROM, pool2_dst_memory},
                                    {DNNL_ARG_TO, conv3_src_memory}});
        }

        auto conv3_weights_memory = conv3_user_weights_memory;
        if (conv3_prim_desc.weights_desc() != conv3_user_weights_memory.get_desc())
        {
                conv3_weights_memory = memory(conv3_prim_desc.weights_desc(), eng);
                reorder(conv3_user_weights_memory, conv3_weights_memory)
                    .execute(s, conv3_user_weights_memory, conv3_weights_memory);
        }

        auto conv3_dst_memory = memory(conv3_prim_desc.dst_desc(), eng);

        // create convolution primitive and add it to net
        net.push_back(convolution_forward(conv3_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv3_src_memory},
                            {DNNL_ARG_WEIGHTS, conv3_weights_memory},
                            {DNNL_ARG_BIAS, conv3_user_bias_memory},
                            {DNNL_ARG_DST, conv3_dst_memory}});

        // conv4
        memory::dims conv4_src_tz = {batch, 384, 13, 13};
        memory::dims conv4_weights_tz = {256, 384, 3, 3};
        memory::dims conv4_bias_tz = {256};
        memory::dims conv4_dst_tz = {batch, 256, 13, 13};
        memory::dims conv4_strides = {1, 1};
        memory::dims conv4_padding = {1, 1};

        std::vector<float> conv4_weights = load_net_data("conv4-weight");
        std::vector<float> conv4_bias = load_net_data("conv4-bias");

        // create memory for user data
        auto conv4_user_weights_memory = memory({{conv4_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv4_weights.data(), conv4_user_weights_memory);
        auto conv4_user_bias_memory = memory({{conv4_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv4_bias.data(), conv4_user_bias_memory);

        // create memory descriptors for convolution data w/ no specified format
        auto conv4_src_md = memory::desc({conv4_src_tz}, dt::f32, tag::any);
        auto conv4_bias_md = memory::desc({conv4_bias_tz}, dt::f32, tag::any);
        auto conv4_weights_md = memory::desc({conv4_weights_tz}, dt::f32, tag::any);
        auto conv4_dst_md = memory::desc({conv4_dst_tz}, dt::f32, tag::any);

        post_ops conv_ops4;
        conv_ops4.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
        primitive_attr conv_attr4;
        conv_attr4.set_post_ops(conv_ops4);

        // create a convolution
        auto conv4_prim_desc = convolution_forward::primitive_desc(eng,
                                                                   prop_kind::forward_inference, algorithm::convolution_direct,
                                                                   conv4_src_md, conv4_weights_md, conv4_bias_md, conv4_dst_md,
                                                                   conv4_strides, conv4_padding, conv4_padding, conv_attr4);

        auto conv4_src_memory = conv3_dst_memory;
        if (conv4_prim_desc.src_desc() != conv4_src_memory.get_desc())
        {
                conv4_src_memory = memory(conv4_prim_desc.src_desc(), eng);
                net.push_back(reorder(conv3_dst_memory, conv4_src_memory));
                net_args.push_back({{DNNL_ARG_FROM, conv3_dst_memory},
                                    {DNNL_ARG_TO, conv4_src_memory}});
        }

        auto conv4_weights_memory = conv4_user_weights_memory;
        if (conv4_prim_desc.weights_desc() != conv4_user_weights_memory.get_desc())
        {
                conv4_weights_memory = memory(conv4_prim_desc.weights_desc(), eng);
                reorder(conv4_user_weights_memory, conv4_weights_memory)
                    .execute(s, conv4_user_weights_memory, conv4_weights_memory);
        }

        auto conv4_dst_memory = memory(conv4_prim_desc.dst_desc(), eng);

        // create convolution primitive and add it to net
        net.push_back(convolution_forward(conv4_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv4_src_memory},
                            {DNNL_ARG_WEIGHTS, conv4_weights_memory},
                            {DNNL_ARG_BIAS, conv4_user_bias_memory},
                            {DNNL_ARG_DST, conv4_dst_memory}});

        // conv5
        memory::dims conv5_src_tz = {batch, 256, 13, 13};
        memory::dims conv5_weights_tz = {256, 256, 3, 3};
        memory::dims conv5_bias_tz = {256};
        memory::dims conv5_dst_tz = {batch, 256, 13, 13};
        memory::dims conv5_strides = {1, 1};
        memory::dims conv5_padding = {1, 1};

        std::vector<float> conv5_weights = load_net_data("conv5-weight");
        std::vector<float> conv5_bias = load_net_data("conv5-bias");

        //         std::vector<float> conv5_weights(product(conv5_weights_tz));
        // std::vector<float> conv5_bias(product(conv5_bias_tz));

        // create memory for user data
        auto conv5_user_weights_memory = memory({{conv5_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv5_weights.data(), conv5_user_weights_memory);
        auto conv5_user_bias_memory = memory({{conv5_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv5_bias.data(), conv5_user_bias_memory);

        // create memory descriptors for convolution data w/ no specified format
        auto conv5_src_md = memory::desc({conv5_src_tz}, dt::f32, tag::any);
        auto conv5_weights_md = memory::desc({conv5_weights_tz}, dt::f32, tag::any);
        auto conv5_bias_md = memory::desc({conv5_bias_tz}, dt::f32, tag::any);
        auto conv5_dst_md = memory::desc({conv5_dst_tz}, dt::f32, tag::any);

        post_ops conv_ops5;
        conv_ops5.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
        primitive_attr conv_attr5;
        conv_attr5.set_post_ops(conv_ops5);

        // create a convolution
        auto conv5_prim_desc = convolution_forward::primitive_desc(eng,
                                                                   prop_kind::forward_inference, algorithm::convolution_direct,
                                                                   conv5_src_md, conv5_weights_md, conv5_bias_md, conv5_dst_md,
                                                                   conv5_strides, conv5_padding, conv5_padding, conv_attr5);

        auto conv5_src_memory = conv4_dst_memory;
        if (conv5_prim_desc.src_desc() != conv5_src_memory.get_desc())
        {
                conv5_src_memory = memory(conv5_prim_desc.src_desc(), eng);
                net.push_back(reorder(conv4_dst_memory, conv5_src_memory));
                net_args.push_back({{DNNL_ARG_FROM, conv4_dst_memory},
                                    {DNNL_ARG_TO, conv5_src_memory}});
        }

        auto conv5_weights_memory = conv5_user_weights_memory;
        if (conv5_prim_desc.weights_desc() != conv5_user_weights_memory.get_desc())
        {
                conv5_weights_memory = memory(conv5_prim_desc.weights_desc(), eng);
                reorder(conv5_user_weights_memory, conv5_weights_memory)
                    .execute(s, conv5_user_weights_memory, conv5_weights_memory);
        }

        auto conv5_dst_memory = memory(conv5_prim_desc.dst_desc(), eng);

        // create convolution primitive and add it to net
        net.push_back(convolution_forward(conv5_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv5_src_memory},
                            {DNNL_ARG_WEIGHTS, conv5_weights_memory},
                            {DNNL_ARG_BIAS, conv5_user_bias_memory},
                            {DNNL_ARG_DST, conv5_dst_memory}});

        // pool5
        memory::dims pool5_dst_tz = {batch, 256, 6, 6};
        memory::dims pool5_kernel = {3, 3};
        memory::dims pool5_strides = {2, 2};
        memory::dims pool5_dilation = {0, 0};
        memory::dims pool5_padding = {0, 0};

        auto pool5_dst_md = memory::desc({pool5_dst_tz}, dt::f32, tag::any);

        // create a pooling
        auto pool5_pd = pooling_forward::primitive_desc(eng,
                                                        prop_kind::forward_inference, algorithm::pooling_max,
                                                        conv5_dst_memory.get_desc(), pool5_dst_md, pool5_strides,
                                                        pool5_kernel, pool5_dilation, pool5_padding, pool5_padding);

        auto pool5_dst_memory = memory(pool5_pd.dst_desc(), eng);

        // create pooling primitive an add it to net
        net.push_back(pooling_forward(pool5_pd));
        net_args.push_back({{DNNL_ARG_SRC, conv5_dst_memory},
                            {DNNL_ARG_DST, pool5_dst_memory}});

        // fc6
        memory::dims fc6_src_tz = {batch, 256, 6, 6};
        memory::dims fc6_weights_tz = {4096, 256, 6, 6};
        memory::dims fc6_bias_tz = {4096};
        memory::dims fc6_dst_tz = {batch, 4096};

        std::vector<float> fc6_weights = load_net_data("fc6-weight");
        std::vector<float> fc6_bias = load_net_data("fc6-bias");

        // std::vector<float> fc6_weights(product(fc6_weights_tz));
        // std::vector<float> fc6_bias(product(fc6_dst_tz));
        // create memory for user data
        auto fc6_user_weights_memory = memory({{fc6_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(fc6_weights.data(), fc6_user_weights_memory);
        auto fc6_user_bias_memory = memory({{fc6_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(fc6_bias.data(), fc6_user_bias_memory);

        // create memory descriptors for convolution data w/ no specified format
        auto fc6_src_md = memory::desc({fc6_src_tz}, dt::f32, tag::any);
        auto fc6_bias_md = memory::desc({fc6_bias_tz}, dt::f32, tag::any);
        auto fc6_weights_md = memory::desc({fc6_weights_tz}, dt::f32, tag::any);
        auto fc6_dst_md = memory::desc({fc6_dst_tz}, dt::f32, tag::any);

        // create a inner_product
        auto fc6_prim_desc = inner_product_forward::primitive_desc(eng,
                                                                   prop_kind::forward_inference, fc6_src_md, fc6_weights_md,
                                                                   fc6_bias_md, fc6_dst_md);
        auto fc6_src_memory = pool5_dst_memory;

        if (fc6_prim_desc.src_desc() != fc6_src_memory.get_desc())
        {
                fc6_src_memory = memory(fc6_prim_desc.src_desc(), eng);
                net.push_back(reorder(pool5_dst_memory, fc6_src_memory));
                net_args.push_back({{DNNL_ARG_FROM, pool5_dst_memory},
                                    {DNNL_ARG_TO, fc6_src_memory}});
        }

        auto fc6_weights_memory = fc6_user_weights_memory;
        if (fc6_prim_desc.weights_desc() != fc6_user_weights_memory.get_desc())
        {
                fc6_weights_memory = memory(fc6_prim_desc.weights_desc(), eng);
                reorder(fc6_user_weights_memory, fc6_weights_memory)
                    .execute(s, fc6_user_weights_memory, fc6_weights_memory);
        }

        auto fc6_dst_memory = memory(fc6_prim_desc.dst_desc(), eng);

        // create convolution primitive and add it to net
        net.push_back(inner_product_forward(fc6_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc6_src_memory},
                            {DNNL_ARG_WEIGHTS, fc6_weights_memory},
                            {DNNL_ARG_BIAS, fc6_user_bias_memory},
                            {DNNL_ARG_DST, fc6_dst_memory}});

        // fc7
        memory::dims fc7_weights_tz = {4096, 4096};
        memory::dims fc7_bias_tz = {4096};
        memory::dims fc7_dst_tz = {batch, 4096};

        std::vector<float> fc7_weights = load_net_data("fc7-weight");
        std::vector<float> fc7_bias = load_net_data("fc7-bias");
        //             std::vector<float> fc7_weights(product(fc7_weights_tz));
        //     std::vector<float> fc7_bias(product(fc7_bias_tz));

        //     // initializing non-zero values for weights and bias
        //     for (size_t i = 0; i < fc7_weights.size(); ++i)
        //         fc7_weights[i] = sinf((float)i);
        //     for (size_t i = 0; i < fc7_bias.size(); ++i)
        //         fc7_bias[i] = sinf((float)i);

        // create memory for user data
        auto fc7_user_weights_memory = memory({{fc7_weights_tz}, dt::f32, tag::nc}, eng);
        write_to_dnnl_memory(fc7_weights.data(), fc7_user_weights_memory);

        auto fc7_user_bias_memory = memory({{fc7_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(fc7_bias.data(), fc7_user_bias_memory);

        // create memory descriptors for convolution data w/ no specified format
        auto fc7_bias_md = memory::desc({fc7_bias_tz}, dt::f32, tag::any);
        auto fc7_weights_md = memory::desc({fc7_weights_tz}, dt::f32, tag::any);
        auto fc7_dst_md = memory::desc({fc7_dst_tz}, dt::f32, tag::any);

        // create a inner_product
        auto fc7_prim_desc = inner_product_forward::primitive_desc(eng,
                                                                   prop_kind::forward_inference, fc6_dst_memory.get_desc(),
                                                                   fc7_weights_md, fc7_bias_md, fc7_dst_md);

        auto fc7_weights_memory = fc7_user_weights_memory;
        if (fc7_prim_desc.weights_desc() != fc7_user_weights_memory.get_desc())
        {
                fc7_weights_memory = memory(fc7_prim_desc.weights_desc(), eng);
                reorder(fc7_user_weights_memory, fc7_weights_memory)
                    .execute(s, fc7_user_weights_memory, fc7_weights_memory);
        }

        auto fc7_dst_memory = memory(fc7_prim_desc.dst_desc(), eng);

        // create convolution primitive and add it to net
        net.push_back(inner_product_forward(fc7_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc6_dst_memory},
                            {DNNL_ARG_WEIGHTS, fc7_weights_memory},
                            {DNNL_ARG_BIAS, fc7_user_bias_memory},
                            {DNNL_ARG_DST, fc7_dst_memory}});

        // fc8
        memory::dims fc8_weights_tz = {3, 4096};
        memory::dims fc8_bias_tz = {3};
        memory::dims fc8_dst_tz = {batch, 3};

        std::vector<float> fc8_weights = load_net_data("fc8-weight");
        std::vector<float> fc8_bias = load_net_data("fc8-bias");
        // std::vector<float> fc8_weights(product(fc8_weights_tz));
        // std::vector<float> fc8_bias(product(fc8_dst_tz));

        // create memory for user data
        auto fc8_user_weights_memory = memory({{fc8_weights_tz}, dt::f32, tag::nc}, eng);
        write_to_dnnl_memory(fc8_weights.data(), fc8_user_weights_memory);
        auto fc8_user_bias_memory = memory({{fc8_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(fc8_bias.data(), fc8_user_bias_memory);
        auto user_dst_memory = memory({{fc8_dst_tz}, dt::f32, tag::nc}, eng);
        write_to_dnnl_memory(user_dst.data(), user_dst_memory);

        // create memory descriptors for convolution data w/ no specified format
        auto fc8_bias_md = memory::desc({fc8_bias_tz}, dt::f32, tag::any);
        auto fc8_weights_md = memory::desc({fc8_weights_tz}, dt::f32, tag::any);
        auto fc8_dst_md = memory::desc({fc8_dst_tz}, dt::f32, tag::any);

        // create a inner_product
        auto fc8_prim_desc = inner_product_forward::primitive_desc(eng,
                                                                   prop_kind::forward_inference, fc7_dst_memory.get_desc(),
                                                                   fc8_weights_md, fc8_bias_md, fc8_dst_md);

        auto fc8_weights_memory = fc8_user_weights_memory;
        if (fc8_prim_desc.weights_desc() != fc8_user_weights_memory.get_desc())
        {
                fc8_weights_memory = memory(fc8_prim_desc.weights_desc(), eng);
                reorder(fc8_user_weights_memory, fc8_weights_memory)
                    .execute(s, fc8_user_weights_memory, fc8_weights_memory);
        }

        auto fc8_dst_memory = memory(fc8_prim_desc.dst_desc(), eng);

        // create convolution primitive and add it to net
        net.push_back(inner_product_forward(fc8_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc7_dst_memory},
                            {DNNL_ARG_WEIGHTS, fc8_weights_memory},
                            {DNNL_ARG_BIAS, fc8_user_bias_memory},
                            {DNNL_ARG_DST, fc8_dst_memory}});

        // softmax 8
        const int axis = 1;
        auto softmax_pd = softmax_forward::primitive_desc(eng, prop_kind::forward_training,
                                                          algorithm::softmax_accurate, fc8_dst_memory.get_desc(), fc8_dst_memory.get_desc(), axis);
        net.push_back(softmax_forward(softmax_pd));
        net_args.push_back({{DNNL_ARG_SRC, fc8_dst_memory},
                            {DNNL_ARG_DST, fc8_dst_memory}});

        // create reorder between internal and user data if it is needed and
        // add it to net after pooling
        if (fc8_dst_memory != user_dst_memory)
        {
                net.push_back(reorder(fc8_dst_memory, user_dst_memory));
                net_args.push_back({{DNNL_ARG_FROM, fc8_dst_memory},
                                    {DNNL_ARG_TO, user_dst_memory}});
        }

        //[Execute model]

        std::vector<float> inputs(user_src.size());
        read_img(img_name, inputs);
        write_to_dnnl_memory(inputs.data(), net_args.at(0).find(DNNL_ARG_SRC)->second);

        // 先运行一次
        for (int j = 0; j < times; ++j)
        {
                assert(net.size() == net_args.size() && "something is missing");
                for (size_t i = 0; i < net.size(); ++i)
                {
                        net.at(i).execute(s, net_args.at(i));
                }
        }
        s.wait();
        uint64_t dur_time = 0;
        auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
        for (int j = 0; j < times; ++j)
        {
                assert(net.size() == net_args.size() && "something is missing");
                for (size_t i = 0; i < net.size(); ++i)
                {
                        net.at(i).execute(s, net_args.at(i));
                }
        }
        s.wait();
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
        dur_time = end - begin;
        //[Execute model]

        std::vector<float> output(IN *3);
        read_from_dnnl_memory(output.data(), net_args.at(net.size() - 1).find(DNNL_ARG_DST)->second);

        std::vector<std::string> class_names = {"cloudy", "rainy", "snow"};
        std::cout << "==================================================" << std::endl;
        std::cout << "==============="
                  << " alexnet "
                  << "===============" << std::endl;
        std::cout << times << " th Iteration, Total dur time :: " << dur_time << " milliseconds" << std::endl;
        int max_index = max_element(output.begin(), output.end()) - output.begin();
        std::cout << "Index : " << max_index << ", Probability : " << output[max_index] << ", Class Name : " << class_names[max_index] << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << "layer count : " << net.size() << std::endl;
}

// int main(int argc, char **argv)
// {
//         int times = 100;
//         engine::kind engine_kind = parse_engine_kind(argc, argv);
//         simple_net(engine_kind, times);
//         return 0;
// }
int main(int argc, char **argv)
{
        int times = 100;
        engine::kind engine_kind = parse_engine_kind(argc, argv, 1);
        simple_net(engine_kind, times, argv[2]);
        return 0;
}