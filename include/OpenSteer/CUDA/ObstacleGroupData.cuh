#ifndef OPENSTEER_OBSTACLEGROUPDATA_CUH
#define OPENSTEER_OBSTACLEGROUPDATA_CUH

#include "dev_vector.cuh"

#include <vector>

namespace OpenSteer
{
typedef struct obstacle_data
{
	float4	position;
	float	radius;
} ObstacleData;

class obstacle_group_data
{
	friend class ObstacleGroup;
private:
	bool	m_bSyncDevice;
	uint	m_nSize;

	//
	// Device vectors
	//
	dev_vector< float4 >	m_dvPosition;
	dev_vector< float >		m_dvRadius;

	//
	// Host vectors
	//
	std::vector< float4 >	m_hvPosition;
	std::vector< float >	m_hvRadius;

public:
	obstacle_group_data( void )
		:	m_nSize( 0 ),
			m_bSyncDevice( false )
	{ }

	~obstacle_group_data( void ) { }

	// Accessors for the device data.
	float4 *	pdPosition( void )							{ return m_dvPosition.begin(); }
	float *		pdRadius( void )							{ return m_dvRadius.begin(); }

	// Accessors for the host data.
	std::vector< float4 > const& hvPosition( void ) const	{ return m_hvPosition; }
	std::vector< float4 > & hvPosition( void )				{ m_bSyncDevice = true; return m_hvPosition; }

	std::vector< float > const& hvRadius( void ) const		{ return m_hvRadius; }
	std::vector< float > & hvRadius( void )					{ m_bSyncDevice = true; return m_hvRadius; }

	uint size( void ) const									{ return m_nSize; }

	/// Adds an obstacle from an obstacle_data structure.
	void addObstacle( obstacle_data const& od )
	{
		m_hvPosition.push_back( od.position );
		m_hvRadius.push_back( od.radius );

		m_nSize++;
		m_bSyncDevice = true;
	}

	/// Removes the obstacle structure at index.
	void removeObstacle( uint const index )
	{
		if( index < m_nSize )
		{
			m_hvPosition.erase( m_hvPosition.begin() + index );
			m_hvRadius.erase( m_hvRadius.begin() + index );
			
			m_nSize--;
			m_bSyncDevice = true;
		}
	}

	/// Get the data for the obstacle at index.
	void getObstacleData( uint const index, obstacle_data & od )
	{
		if( index < m_nSize )
		{
			// The obstacles don't move, no need to sync the host.
			od.position	= m_hvPosition[ index ];
			od.radius	= m_hvRadius[ index ];
		}
	}

	/// Copy the host data to the device.
	void syncDevice( void )
	{
		if( m_bSyncDevice )
		{
			m_dvPosition	= m_hvPosition;
			m_dvRadius		= m_hvRadius;

			m_bSyncDevice = false;
		}
	}

	/// Clear all obstacles.
	void clear( void )
	{
		m_nSize = 0;
		m_bSyncDevice = false;

		m_dvPosition.clear();
		m_dvRadius.clear();

		m_hvPosition.clear();
		m_hvRadius.clear();
	}
};	// class ObstacleGroupData
typedef obstacle_group_data ObstacleGroupData;
}	// namespace OpenSteer

#endif
