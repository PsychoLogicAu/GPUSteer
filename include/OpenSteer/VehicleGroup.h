#ifndef OPENSTEER_VEHICLEGROUP_H
#define OPENSTEER_VEHICLEGROUP_H

#include "VehicleData.cu"
#include <map>

namespace OpenSteer
{
	class VehicleGroup
	{
		typedef std::map<unsigned int, unsigned int> idToIndexMap;

	protected:
		std::vector<vehicle_data>		m_vehicleData;
		std::vector<vehicle_const>		m_vehicleConst;
		idToIndexMap					m_idToIndexMap;

		unsigned int					m_nCount;

		int GetVehicleIndex(unsigned int _id) const;

	public:
		VehicleGroup(void);
		virtual ~VehicleGroup(void);

		bool AddVehicle(VehicleData _data, VehicleConst _const);
		void RemoveVehicle(const unsigned int _id);
		void Clear(void);

		/// Get the size of the collection.
		unsigned int Size() const { return m_nCount; }

		/// Retrieve a pointer to the VehicleData array.
		VehicleData* GetVehicleData() { if(m_nCount > 0) return &m_vehicleData[0]; else return NULL; }

		/// Retrieve a pointer to the VehicleConst array.
		VehicleConst* GetVehicleConst() { if(m_nCount > 0) return &m_vehicleConst[0]; else return NULL; }

		/// Use to extract data for an individual vehicle
		bool GetDataForVehicle(const unsigned int _id, VehicleData &_data, VehicleConst &_const) const;

		void OutputDataToFile(const char *filename);
	};//class VehicleGroup
}//namespace OpenSteer
#endif
