#pragma once

#include "cutlass2/half.h"
#include "cutlass2/bfloat16.h"

// convert half to cutlass::half_t
//      or bfloat16 to cutlass::bfloat16_t
template <typename Element> class ToCutlassType {
public:
  using Element_if_bf16 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, nv_bfloat16>::value,
                                                    cutlass::bfloat16_t, Element>::type;
  using Element_if_fp16 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, half>::value,
                                                    cutlass::half_t, Element_if_bf16>::type;
  using type = Element_if_fp16;
};

// convert cutlass::half_t to half
//      or cutlass::bfloat16_t to bfloat16
template <typename Element> class FromCutlassType {
public:
  using Element_if_bf16 =
      typename cutlass::platform::conditional<cutlass::platform::is_same<Element, cutlass::bfloat16_t>::value, nv_bfloat16,
                                Element>::type;
  using Element_if_fp16 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, cutlass::half_t>::value,
                                                    half, Element_if_bf16>::type;
  using type = Element_if_fp16;
};

// convert float to tf32 for WmmaFragABType
template <typename Element> class ToTF32 {
public:
  using type = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, float>::value,
                                         awmma::precision::tf32, Element>::type;
};