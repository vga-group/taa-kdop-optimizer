// Copyright 2024 Julius Ikkala
// 
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#extension GL_EXT_control_flow_attributes : enable

const int neighborhood_size = 9;
const int axis_count = /* Number of k-DOP axes (k/2) */;
const vec3 axes[] = vec3[](
    // k-DOP axes go here
);

vec3 kdop_clipping(
    vec3 cur_color,
    vec3 prev_color,
    vec3 colors[neighborhood_size]
){
    const float epsilon = 1e-5f;

    vec3 dir = prev_color - cur_color;
    float near = -1e9f, far = 1e9f;
    [[unroll]] for(int a = 0; a < axis_count; ++a)
    {
        vec3 axis = axes[a];
        vec2 extent = vec2(1e9f, -1e9f);
        [[unroll]] for(int n = 0; n < neighborhood_size; ++n)
        {
            float t = dot(colors[n], axis);
            extent.x = min(t, extent.x);
            extent.y = max(t, extent.y);
        }
        extent += vec2(-epsilon +epsilon);
        float proj_pos = dot(cur_color, axis);
        float inv_dir = 1.0f / dot(dir, axis);
        float t0 = (extent.x - proj_pos) * inv_dir;
        float t1 = (extent.y - proj_pos) * inv_dir;
        near = max(near, min(t0, t1));
        far = min(far, max(t0, t1));
    }
    if(near <= far && (near > 0.0f || far > 0.0f))
    {
        float t = clamp(near > 0.0f ? near : far, 0.0f, 1.0f);
        return cur_color + t * dir;
    }
    return cur_color;
}

vec3 kdop_variance_clipping(
    vec3 cur_color,
    vec3 prev_color,
    float gamma,
    vec3 colors[neighborhood_size]
){
    vec3 dir = prev_color - cur_color;
    float near = -1e9f, far = 1e9f;
    [[unroll]] for(int a = 0; a < axis_count; ++a)
    {
        vec3 axis = axes[a];
        vec2 moments = vec2(0);
        [[unroll]] for(int n = 0; n < neighborhood_size; ++n)
        {
            float t = dot(colors[n], axis);
            moments += vec2(t, t*t);
        }
        float proj_pos = dot(cur_color, axis);
        moments /= vec2(neighborhood_size);
        float mu = moments.x;
        float sigma = sqrt(moments.y - mu*mu);
        vec2 extent = vec2(
            min(mu - gamma * sigma, proj_pos),
            max(mu + gamma * sigma, proj_pos)
        );
        float inv_dir = 1.0f / dot(dir, axis);
        float t0 = (extent.x - proj_pos) * inv_dir;
        float t1 = (extent.y - proj_pos) * inv_dir;
        near = max(near, min(t0, t1));
        far = min(far, max(t0, t1));
    }
    if(near <= far && (near > 0.0f || far > 0.0f))
    {
        float t = clamp(near > 0.0f ? near : far, 0.0f, 1.0f);
        return cur_color + t * dir;
    }
    return cur_color;
}
