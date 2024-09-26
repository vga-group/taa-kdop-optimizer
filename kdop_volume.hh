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

// NOTE: There's now a better way to do this: https://dl.acm.org/doi/abs/10.1145/3675391
// "SAH-Optimized k-DOP Hierarchies for Ray Tracing" by Káčerik and Bittner 2024.
// However, that paper was published after I wrote this volume calculator. This
// one is "fast enough" to demonstrate what is needed.
//
// This code is not good and I'm not proud of it, although it seems to produce
// correct results. The idea is to find intersection lines between planes,
// then check where those lines intersect with the other sides of the k-DOP.
// Those intersection points are the vertices, which we then just sort in
// clockwise order for each side, and then compute the volume as tetrahedrons.
// This is rather easy because k-DOPs are convex.
//
// Issues:
// * kdop_trace_range() can somehow return values that are outside of the k-DOP,
//   so that's separately fixed with kdop_distance(). This is overhead that
//   could be avoided if kdop_trace_range() worked correctly, but I can't figure
//   out the math.
// * Memory allocations could be avoided for more performance
// * The algorithm is O(n^3) (n^2 from iterating all plane pairs, the last n for
//   tracing a ray for each pair)
//
// Still, it's faster than CGAL with reasonable axis counts, and doesn't crash
// like CGAL and doesn't slow down compile times to several minutes like CGAL.
//
// It should also be pretty easy to calculate the area too, if you want that.
// Near the end of calc_kdop_volume(), after vertices have been de-duplicated,
// si.vertices contains a clockwise list of vertices for each face. You can form
// a triangle fan from that and then calculate the area from that. Mesh
// generation should also be similarly easy.
#ifndef KDOP_VOLUME_HH
#define KDOP_VOLUME_HH
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>
using namespace glm;

inline std::pair<double, double> kdop_trace_range(
    dvec3 pos,
    dvec3 dir,
    size_t axis_count,
    const vec3* axes,
    const vec2* ranges,
    int excluded_axes[2]
){
    double near = -FLT_MAX;
    double far = FLT_MAX;
    for(int a = 0; a < axis_count; ++a)
    {
        if(a == excluded_axes[0] || a == excluded_axes[1]) continue;

        const dvec3 axis = axes[a];
        const dvec2 range = ranges[a];
        double proj_pos = dot(pos, axis);

        double d = dot(dir, axis);
        if(abs(d) < 1e-7) continue;
        double inv_dir = 1.0f / d;

        double t0 = (range.x - proj_pos) * inv_dir;
        double t1 = (range.y - proj_pos) * inv_dir;

        double t_min = min(t0, t1);
        double t_max = max(t0, t1);

        near = max(near, t_min);
        far = min(far, t_max);
    }
    return {near, far};
}

inline double kdop_distance(
    dvec3 pos,
    size_t axis_count,
    const vec3* axes,
    const vec2* ranges
){
    double max_dist = 0;
    for(int a = 0; a < axis_count; ++a)
    {
        const dvec3 axis = axes[a];
        dvec2 range = ranges[a];
        double proj_pos = dot(pos, axis);

        double distance =
            proj_pos < range.x ? range.x - proj_pos :
            proj_pos > range.y ? proj_pos - range.y :
            0;
        if(distance > max_dist) max_dist = distance;
    }
    return max_dist;
}

#define M_1_SQRT3 0.57735026918962576451
inline dmat3 create_tangent_space(dvec3 normal)
{
    dvec3 major;
    if(abs(normal.x) < M_1_SQRT3) major = dvec3(1,0,0);
    else if(abs(normal.y) < M_1_SQRT3) major = dvec3(0,1,0);
    else major = dvec3(0,0,1);

    dvec3 tangent = normalize(cross(normal, major));
    dvec3 bitangent = cross(normal, tangent);
    return dmat3(tangent, bitangent, normal);
}

inline double signed_angle(dvec3 p, dvec3 pivot, const dmat3& tbn)
{
    dvec3 delta = p-pivot;
    return atan2(dot(tbn[0], delta), dot(tbn[1], delta));
}

inline double calc_kdop_volume(
    size_t axis_count,
    const vec3* axes,
    const vec2* ranges
){
    // Algorithm:
    //
    // Find all edges between planes. (N^2)
    //
    // Trace rays along each edge, with related planes removed.
    //     Extent gained from intersection points. No intersections means that
    //     the edge does not exist.
    //
    //  Calculate midpoint on each side, then sort vertices into CCW, then
    //  compute volume based on tetrahedrons to volume midpoint.

    constexpr double epsilon = 1e-5f;
    struct side_info
    {
        std::vector<dvec3> vertices;
    };
    std::vector<side_info> sides(axis_count * 2);

    for(int a = 0; a < axis_count; ++a)
    for(int b = 0; b < axis_count; ++b)
    {
        if(b == a) continue;
        const dvec3 a_axis = axes[a];
        const dvec3 b_axis = axes[b];

        dvec3 dir = cross(a_axis, b_axis);
        double d = dot(a_axis, b_axis);
        double inv = 1.0f / (1-d*d);

        int excluded[2] = {a, b};

        for(int i = 0; i < 4; ++i)
        {
            int a_high = i&1;
            int b_high = i>>1;

            int a_side = a*2+a_high;
            int b_side = b*2+b_high;

            double h1 = ranges[a][a_high];
            double h2 = ranges[b][b_high];
            double c1 = (h1 - h2 * d) * inv;
            double c2 = (h2 - h1 * d) * inv;
            dvec3 point = c1 * a_axis + c2 * b_axis;

            side_info& a_info = sides[a_side];
            side_info& b_info = sides[b_side];

            auto hits = kdop_trace_range(
                point, dir, axis_count, axes, ranges, excluded
            );
            if(hits.first > hits.second) continue;

            dvec3 va = point + hits.first * dir;
            dvec3 vb = point + hits.second * dir;

            double dist = kdop_distance(va, axis_count, axes, ranges);
            if(dist < epsilon)
            {
                //printf("va good\n");
                a_info.vertices.push_back(va);
                b_info.vertices.push_back(va);
            }
            //else printf("va dist: %f\n", dist);

            dist = kdop_distance(vb, axis_count, axes, ranges);
            if(dist < epsilon)
            {
                //printf("vb good\n");
                a_info.vertices.push_back(vb);
                b_info.vertices.push_back(vb);
            }
            //else printf("vb dist: %f\n", dist);
        }
    }

    dvec3 ref_center = dvec3(0);
    // Find first vertex that exists.
    for(side_info& si: sides)
    {
        if(si.vertices.size() > 2)
        {
            ref_center = si.vertices[0];
            break;
        }
    }

    double total_volume = 0;
    for(int i = 0; i < sides.size(); ++i)
    {
        dvec3 axis = axes[i/2];
        side_info& si = sides[i];
        if(si.vertices.size() <= 2) continue;

        dmat3 tbn = create_tangent_space(axis);

        dvec3 ref = vec3(0);
        for(dvec3 v: si.vertices)
            ref += v;

        ref /= si.vertices.size();

        // Sort
        std::sort(
            si.vertices.begin(),
            si.vertices.end(),
            [&](dvec3 a, dvec3 b)
            {
                return signed_angle(a, ref, tbn) < signed_angle(b, ref, tbn);
            }
        );

        // De-duplicate vertices
        dvec3 prev = si.vertices.back();
        for(auto it = si.vertices.begin(); it != si.vertices.end();)
        {
            dvec3 cur = *it;
            dvec3 delta = prev - cur;
            if(dot(delta, delta) < epsilon * epsilon)
            {
                it = si.vertices.erase(it);
                continue;
            }

            prev = cur;
            ++it;
        }

        if(si.vertices.size() <= 2) continue;

        //printf("Axis: %f, %f, %f\n", axis.x, axis.y, axis.z);

        dvec3 ref2 = si.vertices[0]; // May have changed after sorting.
        // Iterate over unique points.
        dvec3 va = si.vertices[1];
        //printf("\tPoint: %f, %f, %f (%f)\n", ref2.x, ref2.y, ref2.z, signed_angle(ref2, ref, tbn));
        //printf("\tPoint: %f, %f, %f (%f)\n", va.x, va.y, va.z, signed_angle(va, ref, tbn));
        for(int i = 2; i < si.vertices.size(); ++i)
        {
            dvec3 vb = si.vertices[i];
            //printf("\tPoint: %f, %f, %f (%f)\n", vb.x, vb.y, vb.z, signed_angle(vb, ref, tbn));
            dmat4 m = mat4(
                dvec4(va, 1),
                dvec4(vb, 1),
                dvec4(ref2, 1),
                dvec4(ref_center, 1)
            );
            double volume = abs(determinant(m))/6;
            //printf("%f\n", volume);
            total_volume += volume;
            va = vb;
        }
    }
    return total_volume;
}

#endif

