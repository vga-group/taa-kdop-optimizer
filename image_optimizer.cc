#include <glm/glm.hpp>
#include "kdop_volume.hh"
#include <vector>
#include <cstdio>
#include <cmath>
#include <clocale>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace glm;

uint pcg(uint& seed)
{
    seed = seed * 747796405u + 2891336453u;
    seed = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737u;
    seed ^= seed >> 22;
    return seed;
}

float generate_uniform_random(uint& seed)
{
    return pcg(seed) * 2.3283064365386963e-10f;
}

vec3 sample_sphere(vec2 u)
{
    float cos_theta = 2.0f * u.x - 1.0f;
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    float phi = u.y * 2.0f * M_PI;
    return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

vec3 sample_sphere(uint& seed)
{
    vec2 u;
    u.x = generate_uniform_random(seed);
    u.y = generate_uniform_random(seed);
    return sample_sphere(u);
}

float find_kdop_volume(
    const vec3* points,
    const vec3* axes,
    size_t axis_count
){
    vec2 axis_extents[32];
    for(size_t i = 0; i < axis_count; ++i)
        axis_extents[i] = vec2(1e9, -1e9);

    for(size_t i = 0; i < 9; ++i)
    {
        vec3 p = points[i];
        for(size_t j = 0; j < axis_count; ++j)
        {
            auto& pair = axis_extents[j];
            float d = dot(p, axes[j]);
            pair.x = std::min(pair.x, d);
            pair.y = std::max(pair.y, d);
        }
    }
    return calc_kdop_volume(axis_count, axes, axis_extents);
}

float evaluate_axes_cost(
    int w,
    int h,
    const uint8_t* image_data,
    const vec3* axes,
    size_t axis_count,
    uint seed,
    size_t attempt_count = 10000
){
    float sum_volume = 0;
    const float gamma = 2.2f;

    #pragma omp parallel for
    for(size_t a = 0; a < attempt_count; ++a)
    {
        uint cur_seed = seed+a;
        int x = clamp(int(generate_uniform_random(cur_seed) * (w-2)+1), 1, w-2);
        int y = clamp(int(generate_uniform_random(cur_seed) * (h-2)+1), 1, h-2);
        vec3 neighborhood[9];
        for(int i = -1; i <= 1; ++i)
        for(int j = -1; j <= 1; ++j)
        {
            int xi = x+i;
            int yi = y+j;
            uint8_t ri = image_data[xi*3+yi*w*3];
            uint8_t gi = image_data[xi*3+1+yi*w*3];
            uint8_t bi = image_data[xi*3+2+yi*w*3];
            float r = pow(ri / 255.0f, gamma);
            float g = pow(gi / 255.0f, gamma);
            float b = pow(bi / 255.0f, gamma);
            neighborhood[i+1+3*(j+1)] = vec3(r, g, b);
        }
        float volume = find_kdop_volume(neighborhood, axes, axis_count);
        #pragma omp critical
        sum_volume += volume;
    }

    sum_volume /= attempt_count;
    return sum_volume;
}

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        printf("Usage: %s <filename> <axis_count> [forced axes...]\n", argv[0]);
        return 1;
    }

    // Make atoi / atof behave predictably
    setlocale(LC_ALL, "C");

    const char* filename = argv[1];
    int axis_count = atoi(argv[2]);

    std::vector<vec3> best_axes(axis_count, vec3(0));
    uint seed = 0;

    int locked_axes = 0;
    for(int i = 0; i < argc-3; ++i)
    {
        int component_index = i%3;
        if(component_index == 0)
            locked_axes++;
        best_axes[locked_axes-1][component_index] = atof(argv[3+i]);
    }
    for(int i = 0; i < locked_axes; ++i)
        best_axes[i] = normalize(best_axes[i]);
    for(int i = locked_axes; i < axis_count; ++i)
        best_axes[i] = sample_sphere(seed);

    int no_improvement = 0;

    int w, h, n;
    unsigned char* data = stbi_load(filename, &w, &h, &n, 3);

    int fail_count = 0;
    float temperature = 1;
    float best_score = 1e9f;
    while(temperature > FLT_MIN)
    {
        std::vector<vec3> axes = best_axes;
        for(int i = locked_axes; i < axis_count; ++i)
            axes[i] = normalize(axes[i] + temperature * sample_sphere(seed));

        float cur_score = evaluate_axes_cost(
            w,
            h,
            data,
            axes.data(),
            axes.size(),
            0
        );
        printf("%f: %e vs %e\n", temperature, cur_score, best_score);

        //float acceptance =
        //    cur_score < best_score ? 1 : exp(-(cur_score - best_score)/temperature);
        //if(acceptance > generate_uniform_random(seed))
        //if(generate_uniform_random(seed) < acceptance)
        if(cur_score < best_score)
        {
            printf("Picked new best axes\n");
            best_axes = axes;
            best_score = cur_score;
            fail_count = 0;
            for(int i = 0; i < axis_count; ++i)
                printf("    vec3(%f, %f, %f),\n", best_axes[i].x, best_axes[i].y, best_axes[i].z);
        }
        else
        {
            fail_count++;
            if(fail_count > 100)
            {
                printf("Shrinking step size\n");
                fail_count = 0;
                temperature *= 0.5;
            }
        }
    }

    printf("Finished axis optimization\n");
    for(int i = 0; i < axis_count; ++i)
        printf("    vec3(%f, %f, %f),\n", best_axes[i].x, best_axes[i].y, best_axes[i].z);

    stbi_image_free(data);

    return 0;
}


