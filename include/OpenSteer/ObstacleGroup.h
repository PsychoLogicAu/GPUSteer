#ifndef OPENSTEER_OBSTACLEGROUP_H
#define OPENSTEER_OBSTACLEGROUP_H

#include "BaseGroup.h"

#include "CUDA/ObstacleGroupData.cuh"

namespace OpenSteer
{
class ObstacleGroup : public BaseGroup
{
protected:
	// Copy the host data to the device.
	void	SyncDevice( void );

	// Obstacle data.
	ObstacleGroupData	m_obstacleGroupData;

	// Number of obstacles.
	uint				m_nCount;

public:
	ObstacleGroup( uint3 const& worldCells, uint const kno );
	virtual ~ObstacleGroup( void );

	void		Clear( void );
	void		AddObstacle( ObstacleData const& od );
	void		RemoveObstacle( uint const index );

	bool GetDataForObstacle( uint const index, ObstacleData & od );

	// Returns the minimum distance to any obstacle within the given radius through the distance parameter.  Returns true or false signifying the success of the search.
	bool MinDistanceToObstacle( const float3 &position, const float radius, float &distance );

	// Overloaded pure virtuals.
	virtual uint		Size( void ) const		{ return m_nCount; }
	virtual float3 *	pdPosition( void )		{ return m_obstacleGroupData.pdPosition(); }
};	// class ObstacleGroup
}	// namespace OpenSteer
#endif
