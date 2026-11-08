// Optional real tiny-metal-nn integration. Sampling and training share one GPU queue.
#include "image_io.h"
#include <metalrt/runtime.h>
#include <tiny-metal-nn/network_with_input_encoding.h>
#include <tiny-metal-nn/trainer.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace metalrt;
namespace {
struct SamplingParams {
    uint32_t count, seed;
};
struct EvaluationParams {
    uint32_t width, height;
};
double elapsed(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
double mse(const tcnn::GPUMatrix<float>& prediction, const tcnn::GPUMatrix<float>& target) {
    double sum = 0;
    for (size_t i = 0; i < target.n_elements(); ++i) {
        const double error = double(prediction.data()[i]) - target.data()[i];
        if (!std::isfinite(error))
            throw std::runtime_error("non-finite network output");
        sum += error * error;
    }
    return sum / target.n_elements();
}
} // namespace
int main(int argc, char** argv) {
    try {
        std::string path = "data/images/Albert_Einstein_Head.jpg", out = "build/image-training";
        uint32_t steps = 1000, batch_size = 16384;
        DispatchPolicy policy = DispatchPolicy::Direct;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--image" && i + 1 < argc)
                path = argv[++i];
            else if (arg == "--steps" && i + 1 < argc)
                steps = metalrt_examples::positive_count(argv[++i]);
            else if (arg == "--batch-size" && i + 1 < argc)
                batch_size = metalrt_examples::positive_count(argv[++i]);
            else if (arg == "--out" && i + 1 < argc)
                out = argv[++i];
            else if (arg == "--indirect")
                policy = DispatchPolicy::Indirect;
            else
                throw std::invalid_argument("usage: image_training [--image path] [--steps N] "
                                            "[--batch-size N] [--out directory] [--indirect]");
        }
        if (!steps || !batch_size || batch_size % 128)
            throw std::invalid_argument(
                "steps must be positive; batch-size must be a positive multiple of 128");
        auto image = metalrt_examples::load_image(path);
        auto device = Device::system_default();
        tcnn::metal::set_default_device(device);
        tcnn::metal::Stream nn_stream{device, policy};
        auto& stream = nn_stream.runtime();
        const auto upload_start = std::chrono::steady_clock::now();
        auto texture = device.upload_texture({image.width, image.height, PixelFormat::RGBA8Unorm},
                                             image.rgba.data(), image.rgba.size());
        const double upload_seconds = elapsed(upload_start);
        auto sampler = device.sampler();
        const std::string shader =
            std::string{METALRT_EXAMPLE_SHADER_DIR} + "/image_training.metal";
        auto sample = device.kernel_file(shader, "sample_training_image");
        auto eval_sample = device.kernel_file(shader, "sample_evaluation_image");
        using Precision = tcnn::network_precision_t;
        const tcnn::json encoding = {{"otype", "HashGrid"},       {"n_levels", 12},
                                     {"n_features_per_level", 2}, {"log2_hashmap_size", 17},
                                     {"base_resolution", 16},     {"per_level_scale", 1.5}};
        const tcnn::json network = {{"otype", "FullyFusedMLP"},
                                    {"n_neurons", 64},
                                    {"n_hidden_layers", 2},
                                    {"activation", "ReLU"},
                                    {"output_activation", "None"}};
        auto model =
            std::make_shared<tcnn::NetworkWithInputEncoding<Precision>>(2, 3, encoding, network);
        auto optimizer = std::shared_ptr<tcnn::Optimizer<Precision>>{
            tcnn::create_optimizer<Precision>({{"otype", "Adam"}, {"learning_rate", .001f}})};
        auto loss =
            std::shared_ptr<tcnn::Loss<Precision>>{tcnn::create_loss<Precision>({{"otype", "L2"}})};
        tcnn::Trainer<float, Precision, Precision> trainer{model, optimizer, loss, 17};
        tcnn::GPUMatrix<float> inputs{2, batch_size}, targets{3, batch_size};
        const uint32_t eval_width = 256;
        const uint32_t eval_height =
            std::max(1u, uint32_t(std::round(double(image.height) * eval_width / image.width)));
        const uint32_t eval_count = eval_width * eval_height;
        tcnn::GPUMatrix<float> eval_input{2, eval_count}, eval_target{3, eval_count},
            prediction{3, eval_count};
        launch_linear(&stream, eval_sample, eval_count,
                      {Argument::texture(0, texture), Argument::sampler(0, sampler),
                       Argument::buffer(0, eval_input.buffer_view(), Access::Write),
                       Argument::buffer(1, eval_target.buffer_view(), Access::Write),
                       Argument::scalar(2, EvaluationParams{eval_width, eval_height})});
        model->inference(nn_stream, eval_input, prediction);
        stream.synchronize();
        const double initial_mse = mse(prediction, eval_target);
        std::cout << "GPU: " << device.capabilities().name << "\nImage: " << path << " ("
                  << image.width << 'x' << image.height << ")\n"
                  << "GPU texture -> (2,N) UV + (3,N) RGB matrices -> tiny-metal-nn HashGrid + "
                     "FullyFusedMLP + Adam\n"
                  << "Upload: " << upload_seconds * 1000 << " ms; batch: " << batch_size
                  << "; parameters: " << model->n_params()
                  << "; initial evaluation MSE: " << initial_mse << std::endl;
        std::vector<std::pair<uint32_t, float>> loss_history;
        const auto start = std::chrono::steady_clock::now();
        for (uint32_t step = 0; step < steps; ++step) {
            std::unique_ptr<tcnn::Trainer<float, Precision, Precision>::ForwardContext> context;
            {
                auto batch = stream.batch();
                launch_linear(&stream, sample, batch_size,
                              {Argument::texture(0, texture), Argument::sampler(0, sampler),
                               Argument::buffer(0, inputs.buffer_view(), Access::Write),
                               Argument::buffer(1, targets.buffer_view(), Access::Write),
                               Argument::scalar(
                                   2, SamplingParams{batch_size, 23u + step * batch_size * 2u})});
                context = trainer.training_step(nn_stream, inputs, targets);
                batch.finish();
            }
            if ((step + 1) % 100 == 0 || step + 1 == steps) {
                const float value = trainer.loss(nn_stream, *context);
                if (!std::isfinite(value))
                    throw std::runtime_error("non-finite training loss");
                loss_history.emplace_back(step + 1, value);
                std::cout << "step " << step + 1 << ": loss=" << value << std::endl;
            }
        }
        stream.synchronize();
        const double training_seconds = elapsed(start);
        model->inference(nn_stream, eval_input, prediction);
        stream.synchronize();
        const double final_mse = mse(prediction, eval_target);
        const double psnr = -10 * std::log10(std::max(final_mse, 1e-20));
        if (final_mse >= initial_mse)
            throw std::runtime_error("training did not reduce fixed-grid evaluation error");
        std::filesystem::create_directories(out);
        metalrt_examples::write_rgb_png(out + "/target.png", eval_width, eval_height,
                                        eval_target.data());
        metalrt_examples::write_rgb_png(out + "/prediction.png", eval_width, eval_height,
                                        prediction.data());
        tcnn::json history = tcnn::json::array();
        for (const auto& point : loss_history)
            history.push_back({{"step", point.first}, {"loss", point.second}});
        tcnn::json result = {{"device", device.capabilities().name},
                             {"image", path},
                             {"width", image.width},
                             {"height", image.height},
                             {"steps", steps},
                             {"batch_size", batch_size},
                             {"texture_upload_seconds", upload_seconds},
                             {"training_seconds_including_first_step", training_seconds},
                             {"initial_evaluation_mse", initial_mse},
                             {"final_evaluation_mse", final_mse},
                             {"psnr_db", psnr},
                             {"evaluation_width", eval_width},
                             {"evaluation_height", eval_height},
                             {"encoding", encoding},
                             {"network", network},
                             {"loss_history", history}};
        std::ofstream report{out + "/results.json"};
        report << result.dump(2) << '\n';
        if (!report)
            throw std::runtime_error("could not write results.json");
        std::cout << "Fixed-grid MSE: " << initial_mse << " -> " << final_mse << " (" << psnr
                  << " dB)\n"
                  << "Training wall time (includes first-step compilation): " << training_seconds
                  << " s\n"
                  << "Saved " << out << "/{target.png,prediction.png,results.json}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "IMAGE TRAINING FAILURE: " << error.what() << '\n';
        return 1;
    }
}
