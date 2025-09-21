# DODRocketsRaylib

This is a RayLib implementation of my CppCon 2025 keynote demo.

[<img src="keynote.png">](https://www.youtube.com/watch?v=SzjJfKHygaQ)

[**More Speed & Simplicity: Practical Data-Oriented Design in C++**](https://www.youtube.com/watch?v=SzjJfKHygaQ)

## How to build and run

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
ninja # or make
cd ..
./build/DODRocketsRaylib.exe
```

Untick "Enable Rendering" and maximize rocket spawn rate + simulation time to compare the update time difference between the various memory layouts.
