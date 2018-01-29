# Multi-Camera Soccer Tracker

We developed a system for the tracking of players and ball in a soccer video. Work done at the Advanced Digital Sciences Center under Stefan Winkler. Details can be found in the attached report. 

# Contents
------------
  * [Requirements](#requirements)
  * [Brief Project Structure](#brief-project-structure)
  * [Results](#results)
  * [Usage](#usage)

# Requirements
------------
What we used to develop the system

  * OpenCV 3.3
  * Microsoft Visual Studio Community 2015
  * Windows 10
  
# Brief Project Structure
------------

    ├── Soccer Tracker                 : Directory containing the source and header files of the project
    ├── dataset                        : Directory containing the dataset and the results folder
    ├── others                         : Directory containing images for instructions
    ├── Soccer Tracker.sln             : MSVC Project Solution
    |── Report.pdf                     : Report
 
# Results
------------
 
[![Vid](/others/Soccer.png)](https://www.youtube.com/watch?v=hiLK2klFtQI)
 
# Usage
------------

 * Download the dataset at [http://www.issia.cnr.it/wp/dataset-cnr-fig/](http://www.issia.cnr.it/wp/dataset-cnr-fig/) and place them in the dataset directory
 
 <img src="/others/dataset.png" width="90%" height="90%">

 * Launch `Soccer Tracker.sln`.
  
 * Set solution configuration and platform to `Release` and `x64`.
  
<img src="/others/Configurations.png">
 
 * Set project properties (see images below) if using a different version of OpenCV.
 
<img src="/others/CGeneral.png" width="90%" height="90%">
<img src="/others/LinkerGeneral.png" width="90%" height="90%">
<img src="/others/LinkerInput.png" width="90%" height="90%">

* Enable OpenMP support

<img src="/others/OpenMP.png" width="90%" height="90%">

* Build the solution
* Ensure that the files `opencv_ffmpeg330_64.dll` and `opencv_world330.dll` are in the release directory (created only after building the solution). Both files can be found at `<your_opencv_folder>\build\bin`.

<img src="/others/dll.png" width="80%" height="80%">

* Run the program via the `Start Without Debugging` option (Ctrl + F5)
* The video will be saved in `dataset/results`
