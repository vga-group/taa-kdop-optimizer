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
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <vector>
#include <cstdio>
#include <cmath>
#include <clocale>
#include "kdop_volume.hh"
using namespace glm;

// The k-DOP axes are slightly different in the paper with the same parameters,
// this is due to swapping from CGAL to our own volume solver. For some k-DOPs,
// small rounding errors can cause noticeable differences in the result axes.
// The differences should be fairly minimal, but if you want to replicate the
// exact paper numbers, the function below _should_ do it.
/*
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Convex_hull_3/dual/halfspace_intersection_3.h>
#include <CGAL/polygon_mesh_processing.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/polygon_mesh_io.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel   K;
typedef K::Plane_3                                            Plane;
typedef K::Point_3                                            Point;
typedef K::Vector_3                                           Vector;
typedef CGAL::Surface_mesh<Point>                             Surface_mesh;

Point vtop(vec3 v) { return Point(v.x, v.y, v.z); }
Vector vtov(vec3 v) { return Vector(v.x, v.y, v.z); }

float evaluate_volume(const std::vector<vec3>& axes)
{
    std::vector<Plane> planes(axes.size()*2);
    for(int i = 0; i < axes.size(); ++i)
    {
        planes[i*2+0] = Plane(vtop(axes[i]), vtov(axes[i]));
        planes[i*2+1] = Plane(vtop(-axes[i]), vtov(-axes[i]));
    }

    Surface_mesh enclosed;
    CGAL::halfspace_intersection_3(planes.begin(), planes.end(), enclosed);

    CGAL::Polygon_mesh_processing::triangulate_faces(enclosed);

    float volume = CGAL::Polygon_mesh_processing::volume(enclosed);

    return volume;
}
*/

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("Usage: %s <axis-count> [forced axes...]\n", argv[0]);
        return 1;
    }

    // Make atoi / atof behave predictably
    setlocale(LC_ALL, "C");

    int axis_count = atoi(argv[1]);
    float best_volume = 1e99;
    std::vector<vec3> best_axes(axis_count, vec3(0));
    std::vector<vec2> extents(axis_count, vec2(-1, 1));
    int locked_axes = 0;
    for(int i = 0; i < argc-2; ++i)
    {
        int component_index = i%3;
        if(component_index == 0)
            locked_axes++;
        best_axes[locked_axes-1][component_index] = atof(argv[2+i]);
    }
    for(int i = 0; i < locked_axes; ++i)
        best_axes[i] = normalize(best_axes[i]);

    int no_improvement = 0;
    float perturbation = 2;

    for(int j = 0; perturbation > 1e-5; ++j)
    {
        std::vector<vec3> axes = best_axes;
        for(int i = locked_axes; i < axis_count; ++i)
            axes[i] = normalize(axes[i]+sphericalRand(perturbation));

        float volume = calc_kdop_volume(axes.size(), axes.data(), extents.data());
        //float volume = evaluate_volume(axes);

        if(volume < best_volume)
        {
            best_volume = volume;
            best_axes = axes;
            no_improvement = 0;
            printf("Best so far on try %d: %f\n", j, volume);
        }
        else
        {
            no_improvement++;
            if(no_improvement > 1000)
            {
                perturbation *= 0.5f;
                no_improvement = 0;
                printf("Adjusted perturbation to %f\n", perturbation);
            }
        }
    }

    printf("Finished with best volume = %f\n", best_volume);
    for(int i = 0; i < axis_count; ++i)
    {
        if(fabs(best_axes[i].x) < 5e-3) best_axes[i].x = 0;
        if(fabs(best_axes[i].y) < 5e-3) best_axes[i].y = 0;
        if(fabs(best_axes[i].z) < 5e-3) best_axes[i].z = 0;
        best_axes[i] = normalize(best_axes[i]);
        printf("    vec3(%f, %f, %f),\n", best_axes[i].x, best_axes[i].y, best_axes[i].z);
    }

    return 0;
}


