#pragma once

namespace aiu {

#define CMAX(a,b) ((a > b) ? a : b)
#define CMIN(a,b) ((a < b) ? a : b)
#define SIZE_OF_VREG 4
#define DOUBLE_VREG 2

//constexpr int WARP_SIZE = 32;

constexpr int inline GetTileThreads(int TB_M, int TB_N, int WARP_M, int WARP_N) {
  return (WARP_SIZE * TB_M * TB_N) / (WARP_M * WARP_N);
}

constexpr int inline GetTileVregPerThread(int WARP_M, int WARP_N, int MMA_K, int SizeOfAB,
                                          int SizeOfACC) {
  return (DOUBLE_VREG * (WARP_M * MMA_K + WARP_N * MMA_K) * SizeOfAB +
          WARP_M * WARP_N * SizeOfACC) /
         (WARP_SIZE * SIZE_OF_VREG);
}

constexpr int inline GetWarpCount(int TB_M, int TB_N, int WARP_M, int WARP_N) {
  return (TB_M * TB_N) / (WARP_M * WARP_N);
}

constexpr int inline GetTileTsmInKB(int TB_M, int TB_N, int TB_K, int SizeOfAB, int Stages) {
  return (TB_M * TB_K + TB_N * TB_K) * SizeOfAB * Stages / 1024;
}

template <typename ElementAB, typename ElementAccumulator>
constexpr int inline GetVregOccupancy(int WARP_M, int WARP_N, int MMA_K, int WARP_COUNT) {
  return (512 * 8) /
         ( GetTileVregPerThread(WARP_M, WARP_N, MMA_K, sizeof(ElementAB), sizeof(ElementAccumulator)) * WARP_COUNT);
}

template <typename ElementAB>
constexpr int inline GetTsmOccupancy(int TB_M, int TB_N, int TB_K, int Stages) {
  return 256 / GetTileTsmInKB(TB_M, TB_N, TB_K, sizeof(ElementAB), Stages);
}

template <typename ElementAB, typename ElementAccumulator>
constexpr int inline GetTileThreadsWithOccupancy(int TB_M, int TB_N, int TB_K, int WARP_M,
                                                 int WARP_N, int MMA_K, int Stages) {
  return GetTileThreads(TB_M, TB_N, WARP_M, WARP_N) *
         CMIN((GetVregOccupancy<ElementAB, ElementAccumulator>(WARP_M, WARP_N, MMA_K, GetWarpCount(TB_M, TB_N, WARP_M, WARP_N))),
              (GetTsmOccupancy<ElementAB>(TB_M, TB_N, TB_K, Stages)));
}

template <typename ElementAB,
          typename ElementAccumulator,
          typename ThreadblockShape,
          typename WarpShape,
          typename MmaShape,
          int Stages>
class LaunchBound {
public:
  static constexpr int value = GetTileThreadsWithOccupancy<ElementAB, ElementAccumulator>(
      ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK, WarpShape::kM,
      WarpShape::kN, MmaShape::kK, Stages);
};

} /// namespace aiu