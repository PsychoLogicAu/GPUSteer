#include "OpenSteer/ObstacleGroup.h"

//#include <iostream>
//#include <fstream>
//using std::endl;

using namespace OpenSteer;

ObstacleGroup::ObstacleGroup( const float3 &center, const float3 &dimensions, const uint3 &divisions)
:	m_nCount(0),
	m_proximityDatabase(NULL),
	m_lookupToken(NULL)
{
	// Create the proximity database.
	m_proximityDatabase = new SphericalObstaclePDB(center, dimensions, divisions);

	// Create the token to use for lookups.
	SphericalObstacleData temp(1.0f, make_float3(0.0f, 0.0f, 0.0f));
	m_lookupToken = m_proximityDatabase->allocateToken(&temp);

	m_vObstacleData.reserve(100);
}

ObstacleGroup::~ObstacleGroup(void)
{
	Clear();

	delete m_lookupToken;
	delete m_proximityDatabase;
}

void ObstacleGroup::Clear(void)
{
	//// Clear the obstacle data vector.
	//for(SphericalObstacleDataIt it = m_vObstacleData.begin(); it != m_vObstacleData.end(); it++)
	//{
	//	delete *it;
	//}
	m_vObstacleData.clear();

	// Delete all of the proximity tokens.
	for(SphericalObstaclePTIt it = m_proximityTokens.begin(); it != m_proximityTokens.end(); it++)
	{
		delete *it;
	}
	m_proximityTokens.clear();

	m_nCount = 0;
}

void ObstacleGroup::AddObstacle(SphericalObstacleData* pData)
{
	// Set the id of the obstacle to its index in the array.
	pData->id = m_nCount;

	// Add a copy of the obstacle into the vector.
	m_vObstacleData.push_back(*pData);

	// Allocate a token for the obstacle.
	SphericalObstaclePT *pToken = m_proximityDatabase->allocateToken(&m_vObstacleData.back());

	// Set the position of the object in the db.
	pToken->updateForNewPosition(pData->center);
	m_proximityTokens.push_back(pToken);

	m_nCount++;
}

// Returns a collection of objects within the given sphere.
void ObstacleGroup::FindNearObstacles(const float3 &center, const float radius, SphericalObstacleDataVec &results)
{
	//m_lookupToken->updateForNewPosition(center); // TODO: this line needed?

	// Get the indices of obstacles within the sphere.
	m_lookupToken->findNeighbors(center, radius, results);
}

// Returns the minimum distance to any obstacle within the given radius through the distance parameter.  Returns true or false signifying the success of the search.
bool ObstacleGroup::MinDistanceToObstacle(const float3 &position, const float radius, float &distance)
{
	SphericalObstacleDataVec neighbours;
	m_lookupToken->findNeighbors(position, radius, neighbours);

	// No neighbours were found within the given sphere.
	if(neighbours.size() == 0)
		return false;

	distance = FLT_MAX;

	// For each neighbour found.
	for(unsigned int i = 0; i < neighbours.size(); i++)
	{
		float dist = float3_distance(position, m_vObstacleData[i].center);

		if(dist < distance)
			distance = dist;
	}

	return true;
}

unsigned int ObstacleGroup::Size(void)
{
	return m_nCount;
}

//void ObstacleGroup::OutputDataToFile(const char *filename)
//{
//	std::ofstream out;
//	out.open(filename);
//	if(out.is_open())
//	{
//		for(unsigned int i = 0; i < Size(); i++)
//		{
//			out << "number: " << i + 1 << endl;
//			out << "center: " << m_vObstacleData[i].center << endl;
//			out << "radius: " << m_vObstacleData[i].radius << endl;
//
//			out << endl;
//		}
//
//		out.close();
//	}
//}
