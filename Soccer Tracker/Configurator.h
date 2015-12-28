#pragma once

#include <cv.h>

using namespace std;
using namespace cv;

namespace st {

//*************************************************************************************************
// ----- This class is used for reading and writing OpenCV objects from and to xml file.
// ----- Used for working with configuration file that stores all ST settings
//*************************************************************************************************
class Configurator {
	
	//_____________________________________________________________________________________________
	private:
		string fileName;

	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		Configurator () {}

		//=========================================================================================
		Configurator(string fileName) {
			this->fileName = fileName;
		}

		//=========================================================================================
		template <class objType>
		objType readObject (string objName) {
			objType parsedObject;
			FileStorage storage(fileName, FileStorage::READ);
			storage[objName] >> parsedObject;
			storage.release();
			return parsedObject;
		}

		//=========================================================================================
		template <class objType>
		void writeObject (string objName, objType obj) {
			FileStorage storage(fileName, FileStorage::APPEND);
			storage << objName << obj;
			storage.release();
		}

		//=========================================================================================
		~Configurator(void) {}
};

}