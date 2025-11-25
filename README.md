# ibbv
ibbv (Indexed Block Bit Vector) utilize SIMD technology to accelerate common set operations.

Requires C++17 and AVX-512.

## How to use

1. Add following lines to `CMakeLists.txt`:
```txt
find_package(ibbv REQUIRED CONFIG)
link_libraries(ibbv) # or target_link_libraries
```
2. Set enviornment variable `ibbv_DIR` to the path to this repo.
3. Use AdapativeBitVector:
```c++
#include <AdaptiveBitVector.hpp>
template <unsigned _unused = 128>
using SparseBitVector = ibbv::AdaptiveBitVector<>;
```

The implementation can be changed by passing `-DIBBV_IMPL=<impl_to_use>` when configuring project. `ibbv::AdaptiveBitVector<>` will be routed to different implementations.

Some implementations doesn't support `find_first` and/or `find_last`.

Available implementations: "ibbv" "abv" "roaring" "wah" "concise" "ewah"

 - ibbv: Indexed Block Bit Vector.
 - abv: Adaptive Bit Vector. Use array to store few elements and ibbv otherwise.
 - roaring: CRoaring implementation (on 2025-09-08T13:52:04Z).
  
