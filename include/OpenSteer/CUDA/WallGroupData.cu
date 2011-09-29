#include "WallGroupData.cuh"

#include "KNNBinData.cuh"

#include "CUDAKernelGlobals.cuh"

#include <algorithm>
#include <list>

#include <fstream>

using namespace OpenSteer;

WallGroupData::WallGroupData( void )
{
}

void WallGroupData::syncDevice( void )
{
	m_dvLineStart = m_hvLineStart;
	m_dvLineMid = m_hvLineMid;
	m_dvLineEnd = m_hvLineEnd;
	m_dvLineNormal = m_hvLineNormal;
}

// Following is based on: http://www.garagegames.com/community/blogs/view/309
bool WallGroupData::intersects(	float3 const& start, float3 const& end,		// Start and end of line segment.
								float3 const& cellMin, float3 const& cellMax,		// Min and max of cell.
								float3 & intersectPoint								// The point of intersection with the cell.
								)
{
	float st, et, fst = 0, fet = 1;
	float const* bmin = &cellMin.x;
	float const* bmax = &cellMax.x;
	float const* si = &start.x;
	float const* ei = &end.x;

	for (int i = 0; i < 3; i++)
	{
		if (*si < *ei)
		{
			if (*si > *bmax || *ei < *bmin)
				return false;
			float di = *ei - *si;
			st = (*si < *bmin)? (*bmin - *si) / di: 0;
			et = (*ei > *bmax)? (*bmax - *si) / di: 1;
		}
		else
		{
			if (*ei > *bmax || *si < *bmin)
				return false;
			float di = *ei - *si;
			st = (*si > *bmax)? (*bmax - *si) / di: 0;
			et = (*ei < *bmin)? (*bmin - *si) / di: 1;
		}

		if (st > fst)
			fst = st;
		if (et < fet)
			fet = et;
		if (fet < fst)
			return false;
		bmin++; bmax++;
		si++; ei++;
	}

	intersectPoint = float3_add( start, float3_scalar_multiply( float3_subtract(end, start), fst ) );
	//*time = fst;
	return true;
}

bool WallGroupData::LoadFromFile( char const* szFilename )
{
	std::ifstream inFile( szFilename );

	if( ! inFile.is_open() )
		return false;

	while( ! inFile.eof() )
	{
		float3 start, mid, end, normal;

		inFile >> start.x >> start.y >> start.z;
		inFile >> end.x >> end.y >> end.z;
		inFile >> normal.x >> normal.y >> normal.z;
		mid.x = 0.5f * (start.x + end.x);
		mid.y = 0.5f * (start.y + end.y);
		mid.z = 0.5f * (start.z + end.z);

		if( ! inFile.good() )
			break;

		m_hvLineStart.push_back( start );
		m_hvLineEnd.push_back( end );
		m_hvLineNormal.push_back( normal );
		m_hvLineMid.push_back( mid );
	}
	inFile.close();

	return true;
}

void WallGroupData::SplitWalls( std::vector< bin_cell > const& cells )
{
	size_t count = m_hvLineStart.size();

	// Copy the vectors to lists as the vectors don't like what I have planned for them.
	std::list< float3 > startList;
	std::list< float3 > midList;
	std::list< float3 > endList;
	std::list< float3 > normalList;
	startList.resize( count );
	midList.resize( count );
	endList.resize( count );
	normalList.resize( count );
	std::copy( m_hvLineStart.begin(), m_hvLineStart.end(), startList.begin() );
	std::copy( m_hvLineMid.begin(), m_hvLineMid.end(), midList.begin() );
	std::copy( m_hvLineEnd.begin(), m_hvLineEnd.end(), endList.begin() );
	std::copy( m_hvLineNormal.begin(), m_hvLineNormal.end(), normalList.begin() );

	std::list< float3 >::iterator itStart	= startList.begin();
	std::list< float3 >::iterator itMid		= midList.begin();
	std::list< float3 >::iterator itEnd		= endList.begin();
	std::list< float3 >::iterator itNormal	= normalList.begin();


	// For each line...
	while( itStart != startList.end() )
	{
		bool intersected = false;

		// For each cell...
		for( std::vector< bin_cell >::const_iterator itCell = cells.begin(); itCell != cells.end(); ++itCell )
		{
			// Does the line segment intersect the cell?
			float3 intersectPoint;

			if( intersected = intersects( *itStart, *itEnd, itCell->minBound, itCell->maxBound, intersectPoint ) )
			{
				float3 midPoint;

				// Add the two new line segments (but only if they are of >EPSILON length).
				// start - intersectPoint
				if( float3_distance( *itStart, intersectPoint ) > EPSILON && float3_distance( intersectPoint, *itEnd ) > EPSILON )
				{
					midPoint = float3_add( *itStart, float3_scalar_multiply( float3_subtract( intersectPoint, *itStart ), 0.5f ) );
					
					startList.push_back( *itStart );
					midList.push_back( midPoint );
					endList.push_back( intersectPoint );
					normalList.push_back( *itNormal );

					midPoint = float3_add( intersectPoint, float3_scalar_multiply( float3_subtract( *itEnd, intersectPoint ), 0.5f ) );

					startList.push_back( intersectPoint );
					midList.push_back( midPoint );
					endList.push_back( *itEnd );
					normalList.push_back( *itNormal );

					// Remove the current line segment.
					itStart		= startList.erase( itStart );
					itMid		= midList.erase( itMid );
					itEnd		= endList.erase( itEnd );
					itNormal	= normalList.erase( itNormal );

					// Break from the inner loop.
					break;
				}
				else
					intersected = false;


			}
		}

		if( intersected )
		{
			// The iterators will have been incremented by the erase.
			continue;
		}

		++itStart;
		++itMid;
		++itEnd;
		++itNormal;
	}

	count = startList.size();

	m_hvLineStart.resize( count );
	m_hvLineMid.resize( count );
	m_hvLineEnd.resize( count );
	m_hvLineNormal.resize( count );

	// Copy the lists back into the vectors.
	std::copy( startList.begin(), startList.end(), m_hvLineStart.begin() );
	std::copy( midList.begin(), midList.end(), m_hvLineMid.begin() );
	std::copy( endList.begin(), endList.end(), m_hvLineEnd.begin() );
	std::copy( normalList.begin(), normalList.end(), m_hvLineNormal.begin() );


	syncDevice();
}
