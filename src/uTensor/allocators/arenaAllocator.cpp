#include "arenaAllocator.hpp"
namespace uTensor {

DEFINE_EVENT(MetaHeaderNotFound);
DEFINE_EVENT(localCircularArenaAllocatorRebalancing);
DEFINE_EVENT(localCircularArenaAllocatorConstructed);
DEFINE_ERROR(InvalidBoundRegionState);
DEFINE_ERROR(InvalidAlignmentAllocation);
DEFINE_ERROR(MetaHeaderNotBound);

}  // namespace uTensor
