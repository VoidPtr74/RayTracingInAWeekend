# Ray Tracing in One Weekend 

<img align="center" src="https://github.com/VoidPtr74/RayTracingInOneWeekend/blob/master/title_image.jpg">

## Overview

Yet another implementation of a simple ray-tracer based on Peter Shirley's "[Ray Tracing in One Weekend](https://github.com/petershirley/raytracinginoneweekend)" written in Rust.

This implementation goes a little further from the first weekend by also using the BVH tree from [the next week](https://github.com/RayTracing/TheNextWeek) to improve rendering speed.

## Building

This implementation uses intrinsics for SSE instructions. To compile, run the following: 

`cargo rustc --release -- --C target-cpu=native`

## Running

For best performance, I recommend building for and running on a cpu that supports FMA AVX instructions. The picture at the top was rendered in about 16 minutes on a laptop running an Intel i9-8950HK CPU @ 2.90GHz (boosting as inconsistently as one might expect). The image was rendered at 3840x2160 with 1024 samples per pixel, running 24 worker threads with a maximum of 20 bounces per ray.

## Notes

The implementation for vec3 and enhancements to AABB were translated from [GPSnoopy](https://github.com/GPSnoopy)'s C++ implementation of [the next week](https://github.com/GPSnoopy/RayTracingTheNextWeek) 
