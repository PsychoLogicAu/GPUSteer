#ifndef OPENSTEER_OBSTACLEGROUP_H
#define OPENSTEER_OBSTACLEGROUP_H

#include "Obstacle.h"
#include "Proximity.h"

namespace OpenSteer
{
	typedef LQProximityDatabase<SphericalObstacleData*>					SphericalObstaclePDB;
	typedef AbstractTokenForProximityDatabase<SphericalObstacleData*>	SphericalObstaclePT;
	typedef std::vector<SphericalObstaclePT*>							SphericalObstaclePTVec;
	typedef SphericalObstaclePTVec::iterator							SphericalObstaclePTIt;

	class ObstacleGroup
	{
	protected:
		unsigned int				m_nCount;
		SphericalObstaclePTVec		m_proximityTokens;
		// Proximity database.
		SphericalObstaclePDB		*m_proximityDatabase;
		// Token used for proximity lookups.
		SphericalObstaclePT			*m_lookupToken;

	public:
		//SphericalObstacleDataVec	m_vObstacleData;
		std::vector<SphericalObstacleData>	m_vObstacleData;

		ObstacleGroup(const float3 &center, const float3 &dimensions, const float3 &divisions);

		~ObstacleGroup(void);

		void Clear(void);
		void AddObstacle(SphericalObstacleData* pData);

		// Returns a collection of objects within the given sphere.
		void FindNearObstacles(const float3 &center, const float radius, SphericalObstacleDataVec &results);

		// Returns the minimum distance to any obstacle within the given radius through the distance parameter.  Returns true or false signifying the success of the search.
		bool MinDistanceToObstacle(const float3 &position, const float radius, float &distance);

		unsigned int Size(void);

		//SphericalObstacleDataVec& GetObstacles(void);

		void OutputDataToFile(const char *filename);
	};
};
#endif
