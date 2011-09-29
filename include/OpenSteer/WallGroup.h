#ifndef OPENSTEER_WALLGROUP_H
#define OPENSTEER_WALLGROUP_H

#include "BaseGroup.h"

#include "CUDA/WallGroupData.cuh"

#include "CUDA/KNNBinData.cuh"

namespace OpenSteer
{
class WallGroup : public BaseGroup
{
protected:
	WallGroupData		m_wallGroupData;

public:
	WallGroup( uint3 const& worldCells, uint const knw );
	~WallGroup( void )
	{}

	bool LoadFromFile( char const* szFilename );
	void SplitWalls( std::vector< bin_cell > const& cells );

	WallGroupData &		GetWallGroupData( void ){ return m_wallGroupData; }

	void SyncDevice( void );

	// Overloaded pure virtuals.
	virtual uint		Size( void ) const		{ return m_wallGroupData.size(); }
	virtual float3 *	pdPosition( void )		{ return m_wallGroupData.pdLineMid(); }
	virtual float3 *	pdDirection( void )		{ return NULL; }
	virtual float *		pdSpeed( void )			{ return NULL; }
	virtual float *		pdRadius( void )		{ return NULL; }

	virtual void		SetSyncHost( void )		{}
};	// class WallGroup
}	// namespace OpenSteer


#endif
