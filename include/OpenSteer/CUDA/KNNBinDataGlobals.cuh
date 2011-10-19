#ifndef OPENSTEER_KNNBINDATAGLOBALS_CUH
#define OPENSTEER_KNNBINDATAGLOBALS_CUH

namespace OpenSteer
{

struct bin_cell
{
	size_t	index;		// Index of this cell.
	float3	position;
	float3	minBound;	// Minimum bounds of this cell.
	float3	maxBound;	// Maximum bounds of this cell.
};

}	// namespace OpenSteer


#endif
