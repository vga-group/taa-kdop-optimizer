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

// For [[unroll]]
#extension GL_EXT_control_flow_attributes : enable

// The default axis set here is the 32-DOP from the paper
// "k-DOP Clipping: Robust Ghosting Mitigation in Temporal Antialiasing"
// (https://doi.org/10.1145/3681758.3697996), but it may be a bit too expensive
// for low-end GPUs. Benchmark and find out. You can find other sets using the
// tools in this repo: https://github.com/vga-group/taa-kdop-optimizer
// Performance is affected by axis_count and how many components in 'axes' are
// zeroed (assuming your compiler unrolls the loops below and simplifies the
// relevant dot products).
const vec3 axes[] = vec3[](
    // k-DOP axes go here
    vec3(1, 0, 0),
    vec3(0, 1, 0),
    vec3(0, 0, 1),
    vec3(0.820081, 0.456727, -0.344773),
    vec3(0.540295, 0.829202, 0.143195),
    vec3(0.255800, 0.841084, -0.476597),
    vec3(-0.406935, -0.389062, 0.826459),
    vec3(-0.826708, -0.382923, -0.412219),
    vec3(0.260942, -0.577482, 0.773578),
    vec3(0.254398, 0.637821, 0.726957),
    vec3(0.310900, -0.728083, -0.610930),
    vec3(0.798513, -0.556827, -0.228738),
    vec3(0.673383, -0.163602, -0.720964),
    vec3(-0.813922, 0.369658, -0.448201),
    vec3(0.477650, -0.853722, 0.207384),
    vec3(-0.554854, -0.041550, -0.830910)
);

// If you wanted to use a +-sized neighborhood instead, set this to 5. The
// default assumes a 3x3 window. Window shape doesn't actually matter for this
// algorithm, just the count. Smaller neighborhoods are faster and more likely
// to ghost less, but also more likely flicker more.
const int neighborhood_size = 9;

// This function is a drop-in replacement for RGB or YCoCg color clipping in
// temporal anti-aliasing. It reduces ghosting by rectifying a reprojected
// color by approximately limiting it to the range of colors that could
// legitimately appear as anti-aliasing edges, based on a given color
// neighborhood. The k-DOP (Discrete Oriented Polytope) approximation is more
// accurate than AABB-based RGB or YCoCg clipping.
//
// It works by constructing a k-DOP in color space around the color neighborhood
// (approximating a convex hull around the neighborhood), then casting a ray
// from the current color towards the previous color and using the intersection
// point. This _approximately_ limits history colors to what is achievable with
// a weighted average of the neighborhood's colors, which is what you would
// expect from anti-aliased edges.
//
// Parameters:
//     cur_color: own pixel color from the current frame
//     prev_color: reprojected history color from previous frame
//     colors: neighborhood colors from current frame
//     return value: rectified history color
//
// Optimization notes:
//     When 2 * axis_count < 3 * neighborhood_size, you could save registers
//     with an alternate formulation, where you calculate the extents when
//     colors are read and pass them to this function instead of keeping colors
//     around. Also, this form gets 'cur_color' twice: once in 'cur_color', and
//     a second time somewhere in 'colors'. So, if your compiler misses that,
//     you may also be able to claw some registers back there too.
//
//     Unrolling these loops is absolutely vital. You really don't want to
//     dynamically index an array on a GPU ;)
vec3 kdop_clipping(
    vec3 cur_color,
    vec3 prev_color,
    vec3 colors[neighborhood_size]
){
    const float epsilon = 1e-5f;

    vec3 dir = prev_color - cur_color;
    float near = -1e9f, far = 1e9f;
    [[unroll]] for(int a = 0; a < axes.length(); ++a)
    {
        vec3 axis = axes[a];
        // Construct color extent along this axis
        vec2 extent = vec2(1e9f, -1e9f);
        [[unroll]] for(int n = 0; n < neighborhood_size; ++n)
        {
            float t = dot(colors[n], axis);
            extent.x = min(t, extent.x);
            extent.y = max(t, extent.y);
        }
        // Some extra padding to prevent issues with otherwise zero-volume
        // k-DOPs in corner cases.
        extent += vec2(-epsilon, +epsilon);

        // Compute intersections to the planes of the current slab
        float proj_pos = dot(cur_color, axis);
        float inv_dir = 1.0f / dot(dir, axis);
        float t0 = (extent.x - proj_pos) * inv_dir;
        float t1 = (extent.y - proj_pos) * inv_dir;

        // Accumulate with previous plane intersections.
        near = max(near, min(t0, t1));
        far = min(far, max(t0, t1));
    }
    if(near <= far && (near > 0.0f || far > 0.0f))
    { // Hit
        float t = clamp(near > 0.0f ? near : far, 0.0f, 1.0f);
        return cur_color + t * dir;
    }
    // Missed??? This shouldn't happen as the ray should start from inside the
    // k-DOP hull, but just in case.
    return cur_color;
}

// This function is a drop-in replacement for RGB or YCoCg variance clipping in
// temporal anti-aliasing. It's similar to the function above (kdop_clipping()),
// but calculates the k-DOP extents from color variance instead. This can make
// tighter bounding volumes and thus remove more ghosting, but it also starts
// rejecting valid colors too, causing more flickering.
//
// Honestly, I wouldn't really bother with variance clipping for k-DOPs, as it's
// slightly slower and introduces more annoying flickering in edges. Use the
// normal clipping (above function) instead. But if you _really_ want a blunt
// hammer approach to ghosting and don't care about flicker or AA quality, maybe
// it's useful?
//
// Parameters:
//     cur_color: own pixel color from the current frame
//     prev_color: reprojected history color from previous frame
//     gamma: parameter that controls how aggressive the variance clipping is
//            (default is typically 1.0 in earlier variance clipping literature)
//     colors: neighborhood colors from current frame
//     return value: rectified history color
//
// Optimization notes:
//     Same as for kdop_clipping().
vec3 kdop_variance_clipping(
    vec3 cur_color,
    vec3 prev_color,
    float gamma,
    vec3 colors[neighborhood_size]
){
    vec3 dir = prev_color - cur_color;
    float near = -1e9f, far = 1e9f;
    [[unroll]] for(int a = 0; a < axes.length(); ++a)
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
