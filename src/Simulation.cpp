#include "OpenSteer/Simulation.h"

#include <cassert>

using namespace OpenSteer;

void GroupParams::load( std::ifstream & fs )
{
	std::string str, key;

	fs >> str;
	while( str != "</group>" )
	{
		// Handle comments.
		if( str[0] == '#' )
		{
			std::getline( fs, str );
			fs >> str;
			continue;
		}

		key = str;
		fs >> str;	// Remove delimiter.

		if( key == "numAgents" )
			fs >> m_nNumAgents;
		else if( key == "startPosition" )
			fs >> m_f3StartPosition.x >> m_f3StartPosition.y >> m_f3StartPosition.z;
		else if( key == "minStartRadius" )
			fs >> m_fMinStartRadius;
		else if( key == "maxStartRadius" )
			fs >> m_fMaxStartRadius;
		else if( key == "maxSpeed" )
			fs >> m_fMaxSpeed;
		else if( key == "maxForce" )
			fs >> m_fMaxForce;
		else if( key == "<path>" )
		{
			while( str != "</path>" )
			{
				// Handle comments.
				if( str[0] == '#' )
				{
					std::getline( fs, str );
					fs >> str;
					continue;
				}

				key = str;
				fs >> str;	// Remove delimiter.

				if( key == "point" )
				{
					float3 point;
					fs >> point.x >> point.y >> point.z;
					m_vecPathPoints.push_back( point );
				}
				else if( key == "cyclic" )
				{
					fs >> m_bPathIsCyclic;
				}
				else if( key == "radius" )
				{
					fs >> m_fPathRadius;
				}

				fs >> str;
			}
		}
		else
		{
			assert( 0 );
		}

		fs >> str;
	}
}

void WorldParams::load( std::ifstream & fs )
{
	std::string str, key;

	fs >> str;
	while( str != "</world>" )
	{
		// Handle comments.
		if( str[0] == '#' )
		{
			std::getline( fs, str );
			fs >> str;
			continue;
		}

		key = str;
		fs >> str;	// Remove delimiter.

		if( key == "dimensions" )
		{
			fs >> m_f3Dimensions.x >> m_f3Dimensions.y >> m_f3Dimensions.z;
		}
		else if( key == "cells" )
		{
			fs >> m_u3Cells.x >> m_u3Cells.y >> m_u3Cells.z;
		}
		else if( key == "obstacleCount" )
		{
			fs >> m_nObstacleCount;
		}
		else if( key == "mapFilename" )
		{
			fs >> m_strMapFilename;
		}
		else
		{
			assert( 0 );
		}

		fs >> str;
	}
}

void SimulationParams::load( char const* szFilename )
{
	std::ifstream fs( szFilename );

	if( ! fs.is_open() )
		return;

	std::string str, key;

	fs >> str;
	while( str != "</simulation>" )
	{
		// Handle comments.
		if( str[0] == '#' )
		{
			std::getline( fs, str );
			fs >> str;
			continue;
		}

		key = str;

		// Load the world.
		if( key == "<world>" )
		{
			m_WorldParams.load( fs );

			fs >> str;
			continue;
		}

		// Load the group.
		if( key == "<group>" )
		{
			GroupParams gp;

			gp.load( fs );
			m_vecGroupParams.push_back( gp );

			fs >> str;
			continue;
		}

		fs >> str;	// Remove delimiter.

		if( key == "seed" )
			fs >> m_nSeed;
		else if( key == "knn" )
			fs >> m_nKNN;
		else if( key == "kno" )
			fs >> m_nKNO;
		else if( key == "knw" )
			fs >> m_nKNW;
		else if( key == "searchRadius" )
			fs >> m_nSearchRadius;
		else if( key == "maxPursuitPredictionTime" )
			fs >> m_fMaxPursuitPredictionTime;
		else if( key == "minTimeToCollision" )
			fs >> m_fMinTimeToCollision;
		else if( key == "minTimeToObstacle" )
			fs >> m_fMinTimeToObstacle;
		else if( key == "minTimeToWall" )
			fs >> m_fMinTimeToWall;
		else if( key == "pathPredictionTime" )
			fs >> m_fPathPredictionTime;
		else if( key == "minSeparationDistance" )
			fs >> m_fMinSeparationDistance;
		else if( key == "maxSeparationDistance" )
			fs >> m_fMaxSeparationDistance;
		else if( key == "minFlockingDistance" )
			fs >> m_fMinFlockingDistance;
		else if( key == "maxFlockingDistance" )
			fs >> m_fMaxFlockingDistance;
		else if( key == "cosMaxFlockingAngle" )
			fs >> m_fCosMaxFlockingAngle;
		// Weights.
		else if( key == "weightAlignment" )
			fs >> m_fWeightAlignment;
		else if( key == "weightCohesion" )
			fs >> m_fWeightCohesion;
		else if( key == "weightSeparation" )
			fs >> m_fWeightSeparation;
		else if( key == "weightPursuit" )
			fs >> m_fWeightPursuit;
		else if( key == "weightSeek" )
			fs >> m_fWeightSeek;
		else if( key == "weightFollowPath" )
			fs >> m_fWeightFollowPath;
		else if( key == "weightAvoidObstacles" )
			fs >> m_fWeightAvoidObstacles;
		else if( key == "weightAvoidWalls" )
			fs >> m_fWeightAvoidWalls;
		else if( key == "weightAvoidNeighbors" )
			fs >> m_fWeightAvoidNeighbors;
		// Masks.
		else if( key == "maskAlignment" )
			fs >> m_nMaskAlignment;
		else if( key == "maskAntiPenetrationAgents" )
			fs >> m_nMaskAntiPenetrationAgents;
		else if( key == "maskCohesion" )
			fs >> m_nMaskCohesion;
		else if( key == "maskSeparation" )
			fs >> m_nMaskSeparation;
		else if( key == "maskSeek" )
			fs >> m_nMaskSeek;
		else if( key == "maskFlee" )
			fs >> m_nMaskFlee;
		else if( key == "maskPursuit" )
			fs >> m_nMaskPursuit;
		else if( key == "maskEvade" )
			fs >> m_nMaskEvade;
		else if( key == "maskFollowPath" )
			fs >> m_nMaskFollowPath;
		else if( key == "maskAvoidObstacles" )
			fs >> m_nMaskAvoidObstacles;
		else if( key == "maskAvoidWalls" )
			fs >> m_nMaskAvoidWalls;
		else if( key == "maskAvoidNeighbors" )
			fs >> m_nMaskAvoidNeighbors;
		else
			assert( 0 );

		fs >> str;
	}
}
