#include <optional>
#include "c10/util/DimVector.h"
#include "torch/torch.h"

namespace my_ops {

at::Tensor sum_dim(const at::Tensor &self,
                   at::OptionalIntArrayRef dim,
                   bool keepdim,
                   ::std::optional<at::ScalarType> dtype);
}
