#include "OpenSteer/Simulation.h"

using namespace OpenSteer;

void GroupParams::open( std::ifstream & fs )
{
	std::string str, key;

	fs >> str;
	while( str != "</group>" )
	{
		// Handle comments.
		if( str == "#" )
		{
			std::getline( fs, str );
			continue;
		}

		key = str;
		fs >> str;	// Remove delimiter.

		if( key == "numAgents" )
		{
			fs >> m_nNumAgents;
		}
		else if( key == "<path>" )
		{
			fs >> str;
			while( str != "</path>" )
			{
				// Handle comments.
				if( str == "#" )
				{
					std::getline( fs, str );
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

				fs >> str;
			}
		}
		else
		{
		}

		fs >> str;
	}
}

void WorldParams::open( std::ifstream & fs )
{
	std::string str, key;

	fs >> str;
	while( str != "</world>" )
	{
		// Handle comments.
		if( str == "#" )
		{
			std::getline( fs, str );
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

		}

		fs >> str;
	}
}

void Simulation::open( char const* szFilename )
{
	std::ifstream fs( szFilename );

	if( ! fs.is_open() )
		return;

	std::string str, key;

	fs >> str;
	while( str != "</simulation>" )
	{
		// Handle comments.
		if( str == "#" )
		{
			std::getline( fs, str );
			continue;
		}

		key = str;
		fs >> str;	// Remove delimiter.
		
		// Load the world.
		if( key == "<world>" )
		{
			m_WorldParams.open( fs );
		}

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


/*
	# Weights

	# Masks

*/




		fs >> str;
	}




}