#include "OpenSteer/VehicleGroup.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// For writing the data to an output file.
#include <iostream>
#include <fstream>
using std::endl;

#include "OpenSteer/Utilities.h"

using namespace OpenSteer;

void VehicleGroup::OutputDataToFile(const char *filename)
{
	std::ofstream out;
	out.open(filename);
	if(out.is_open())
	{
		for(unsigned int i = 0; i < Size(); i++)
		{
			const VehicleData &vdata = m_vehicleData[i];
			const VehicleConst &vconst = m_vehicleConst[i];

			out << "id: " << vconst.id << endl;
			out << "Data:" << endl;
			out << "forward: " << vdata.forward << endl;
			out << "position: " << vdata.position << endl;
			out << "side: " << vdata.side << endl;
			out << "speed: " << vdata.speed << endl;
			out << "steering: " << vdata.steering << endl;
			out << "up: " << vdata.up << endl;

			out << "Const:" << endl;
			out << "id: " << vconst.id << endl;
			out << "mass: " << vconst.mass << endl;
			out << "maxForce: " << vconst.maxForce << endl;
			out << "maxSpeed: " << vconst.maxSpeed << endl;
			out << "radius: " << vconst.radius << endl;
			out << endl;
		}

		out.close();
	}
}

VehicleGroup::VehicleGroup(void)
: m_nCount(0)
{
}

VehicleGroup::~VehicleGroup(void)
{
	Clear();
}

// Gets the index of a vehicle in the vector from its id number.
int VehicleGroup::GetVehicleIndex(unsigned int _id) const
{
	int index = -1;

	// Check if m_idToIndexMap contains the key.
	idToIndexMap::const_iterator it = m_idToIndexMap.find(_id);
	if(it != m_idToIndexMap.end())
	{
		index = it->second;
	}

	//for(unsigned int i = 0; i < m_vehicleData.size(); i++)
	//{
	//	if(m_vehicleData[i].id == _id)
	//	{
	//		index = i;
	//		break;
	//	}
	//}

	return index;
}

// Adds a vehicle to the group.
bool VehicleGroup::AddVehicle(VehicleData _data, VehicleConst _const)
{
	// Add the vehicle if it is not already contained.
	if(GetVehicleIndex(_const.id) == -1)
	{
		m_vehicleData.push_back(_data);
		m_vehicleConst.push_back(_const);

		// Add the id and index to the idToIndexMap.
		m_idToIndexMap[_const.id] = m_nCount++;

		return true;
	}

	return false;
}

// Removes a vehicle from the group using the supplied id number.
void VehicleGroup::RemoveVehicle(const unsigned int _id)
{
	// Get the vehicle's index.
	int index = GetVehicleIndex(_id);

	if(index > -1) // Found.
	{
		m_vehicleData.erase(m_vehicleData.begin() + index);
		m_vehicleConst.erase(m_vehicleConst.begin() + index);

		m_nCount--;
	}
}

// Clears all vehicles from the group.
void VehicleGroup::Clear(void)
{
	// Clear m_vehicleData
	m_vehicleData.clear();

	// Clear m_vehicleConst
	m_vehicleConst.clear();

	m_idToIndexMap.clear();

	m_nCount = 0;
}

// Gets the data for a vehicle from the supplied id.
bool VehicleGroup::GetDataForVehicle(const unsigned int _id, VehicleData &_data, VehicleConst &_const) const
{
	// Get the index of the vehicle.
	int index = GetVehicleIndex(_id);

	if(index == -1) // Not found.
		return false;

	_data = m_vehicleData[index];
	_const = m_vehicleConst[index];

	return true;
}
